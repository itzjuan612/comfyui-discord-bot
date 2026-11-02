import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from core import config

log = logging.getLogger("job_queue")


class JobCancelled(Exception):
    """Raised when a job was cancelled before it produced a result."""


class JobQueuedTimeout(JobCancelled):
    """Raised when a job expired while still waiting in line."""


class JobRunTimeout(JobCancelled):
    """Raised when a running job exceeded its execution budget."""


def job_error_text(prefix: str, exc: Exception) -> str:
    """Failure line for a generation path.

    A cancelled or timed-out job is not a failure, so it gets its own wording
    instead of the caller's "❌ … failed:" prefix.
    """
    if isinstance(exc, (JobQueuedTimeout, JobRunTimeout)):
        return f"\u23f2\ufe0f {exc}."
    if isinstance(exc, JobCancelled):
        return "\u23f9\ufe0f Cancelled."
    return f"{prefix} {exc}"


def log_job_error(logger: logging.Logger, message: str, exc: Exception) -> None:
    """Log a job failure: cancellations are a normal outcome, not an error.

    Call it from the ``except`` block so ``logger.exception`` still captures
    the traceback for real failures.
    """
    if isinstance(exc, JobCancelled):
        logger.info("%s: %s", message, exc)
    else:
        logger.exception(message)


def _timeout_setting(key: str, default: float | None) -> float | None:
    """Read a ``queueing.<key>`` seconds value; 0 or None disables it."""
    value = config.get("queueing", {}).get(key)
    if value is None:
        return default
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        log.warning("Invalid queueing.%s=%r; using %s", key, value, default)
        return default
    return seconds if seconds > 0 else None


# A queued job this old is dropped: its Discord interaction can no longer be
# answered anyway (followup tokens expire after 15 minutes).
DEFAULT_QUEUED_TIMEOUT = 900.0
# No default run budget: ComfyUI's own poll timeout already bounds a prompt.
DEFAULT_RUN_TIMEOUT: float | None = None

# Resolved once at import: config is read at startup and /reload does not
# re-read it, so a malformed value is warned about once, not per job.
QUEUED_TIMEOUT = _timeout_setting("queued_timeout", DEFAULT_QUEUED_TIMEOUT)
RUN_TIMEOUT = _timeout_setting("run_timeout", DEFAULT_RUN_TIMEOUT)


@dataclass(eq=False)
class Job:
    """A unit of work waiting for, or occupying, a queue lane.

    Await the job to get its result, exactly like the Future ``submit`` used
    to return. ``cancel()`` asks the worker to drop it: a queued job is never
    started, a running one is cancelled and ``on_cancel`` (if set) is awaited
    so the external resource — ComfyUI sampling, an LLM request — stops too.
    """

    coro: Any
    name: str = "job"
    queued_timeout: float | None = None
    run_timeout: float | None = None
    on_cancel: Callable[[], Awaitable[None]] | None = None
    created_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        # Binds the job to the running loop: ``submit`` is async-only, so the
        # Job cannot be created (or awaited) from synchronous code.
        loop = asyncio.get_running_loop()
        self._future = loop.create_future()
        self._cancel_event = asyncio.Event()
        # Mark abandoned results as retrieved so a cancelled or timed-out job
        # whose caller already went away does not log "never retrieved".
        self._future.add_done_callback(_retrieve)

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    @property
    def done(self) -> bool:
        """True once the job has a result, failed, timed out or was cancelled."""
        return self._future.done()

    def cancel(self) -> None:
        """Mark this job for cancellation. Safe to call from any task."""
        self._cancel_event.set()

    def __await__(self):
        return self._future.__await__()


def _retrieve(future: asyncio.Future) -> None:
    if not future.cancelled():
        future.exception()


class JobQueue:
    """A single-slot, FIFO queue for long-running, resource-heavy jobs.

    Jobs are coroutines. They are executed strictly one at a time, in the
    order they were submitted. ``submit`` returns a ``Job`` that resolves to
    the job's return value (or raises its exception), so callers can ``await``
    it to get the result. ``wait_drained`` blocks until every submitted job
    has finished.
    """

    # How long a cancel hook (e.g. ComfyUI /interrupt) may take before it is
    # abandoned, so an unreachable server cannot stall the queue.
    CANCEL_HOOK_TIMEOUT = 15.0

    def __init__(self):
        self._queue = asyncio.Queue()
        self._worker_task = None
        self._current: Job | None = None
        self._pending = 0
        self._active = 0
        self._drained = asyncio.Event()
        self._drained.set()
        self._listeners = []

    def _ensure_worker(self) -> None:
        """Start the single worker task if it is not already running."""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.ensure_future(self._worker())

    async def _worker(self) -> None:
        """Consume jobs from the queue one at a time, resolving each job."""
        while True:
            job: Job = await self._queue.get()
            # Job moves from the queue to the running slot.
            self._pending -= 1
            self._active += 1
            self._current = job
            for callback in list(self._listeners):
                asyncio.create_task(self._fire(callback, "start", job.coro))
            try:
                await self._run(job)
            finally:
                self._current = None
                self._active -= 1
                if self._pending == 0 and self._active == 0:
                    self._drained.set()
                for callback in list(self._listeners):
                    asyncio.create_task(self._fire(callback, "finish", job.coro))

    async def _run(self, job: Job) -> None:
        """Run one job, honouring cancellation and the queued/run budgets."""
        if job.cancelled:
            job.coro.close()
            log.info("job %s cancelled before it started", job.name)
            self._finish(job, exc=JobCancelled("Cancelled while waiting in the queue"))
            return
        waited = time.monotonic() - job.created_at
        if job.queued_timeout is not None and waited > job.queued_timeout:
            job.coro.close()
            log.info("job %s waited %.0fs in queue; dropping it", job.name, waited)
            self._finish(job, exc=JobQueuedTimeout("Timed out waiting in the queue"))
            return

        work = asyncio.ensure_future(job.coro)
        cancel_wait = asyncio.ensure_future(job._cancel_event.wait())
        try:
            done, _ = await asyncio.wait(
                {work, cancel_wait},
                timeout=job.run_timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if work in done:
                self._finish(job, result=work.result())
            elif cancel_wait in done:
                await self._stop(job, work, JobCancelled("Cancelled"))
            else:
                await self._stop(
                    job,
                    work,
                    JobRunTimeout(f"Stopped after exceeding the {job.run_timeout:.0f}s time limit"),
                )
        except Exception as exc:  # noqa: BLE001 - the worker must never die
            self._finish(job, exc=exc)
        finally:
            cancel_wait.cancel()
            if not work.done():
                work.cancel()
            # Collect both tasks so nothing is left pending on the loop.
            await asyncio.gather(work, cancel_wait, return_exceptions=True)

    async def _stop(self, job: Job, work: asyncio.Future, exc: Exception) -> None:
        """Cancel the running coroutine and stop the resource it holds.

        The cancel hook fires first and runs while the coroutine unwinds: an
        external resource (ComfyUI sampling) must not keep working just because
        our own cleanup is still in flight. The hook is bounded so an
        unreachable server cannot wedge the worker.
        """
        work.cancel()
        if job.on_cancel is not None:
            try:
                await asyncio.wait_for(job.on_cancel(), timeout=self.CANCEL_HOOK_TIMEOUT)
            except asyncio.TimeoutError:
                log.warning("cancel hook for %s timed out", job.name)
            except Exception:
                log.exception("on_cancel callback failed for %s", job.name)
        await asyncio.gather(work, return_exceptions=True)
        log.info("job %s stopped: %s", job.name, exc)
        self._finish(job, exc=exc)

    @staticmethod
    def _finish(job: Job, *, result: Any = None, exc: BaseException | None = None) -> None:
        if job._future.done():
            return
        if exc is not None:
            job._future.set_exception(exc)
        else:
            job._future.set_result(result)

    async def _fire(self, callback, event: str, coro) -> None:
        """Invoke a listener callback, never letting its failure break the worker."""
        try:
            await callback(event, coro)
        except Exception:
            log.exception("queue listener failed")

    def register(self, callback) -> None:
        """Register an async ``callback(event, coro)`` notified when jobs start/finish."""
        self._listeners.append(callback)

    def unregister(self, callback) -> None:
        try:
            self._listeners.remove(callback)
        except ValueError:
            pass

    def submit(
        self,
        coro,
        name: str = "job",
        *,
        queued_timeout: float | None = None,
        run_timeout: float | None = None,
        on_cancel: Callable[[], Awaitable[None]] | None = None,
    ) -> Job:
        """Queue ``coro`` to run serially.

        Returns a ``Job`` that resolves to the coroutine's return value, or
        raises the coroutine's exception (or ``JobCancelled`` when it is
        cancelled or runs past its budget). The job runs only after any
        earlier submitted jobs have completed.
        """
        job = Job(
            coro,
            name=name,
            queued_timeout=QUEUED_TIMEOUT if queued_timeout is None else queued_timeout,
            run_timeout=RUN_TIMEOUT if run_timeout is None else run_timeout,
            on_cancel=on_cancel,
        )
        self._queue.put_nowait(job)
        self._pending += 1
        self._drained.clear()
        self._ensure_worker()
        return job

    async def wait_drained(self) -> None:
        """Await until no job is running and none are queued."""
        await self._drained.wait()

    async def shutdown(self, timeout: float = 5.0) -> None:
        """Cancel everything still queued or running (bot shutdown).

        Queued jobs are dropped without starting; the running job is cancelled
        through the normal path, so its cancel hook fires and the external
        resource (ComfyUI) stops before the process exits.
        """
        while not self._queue.empty():
            job: Job = self._queue.get_nowait()
            self._pending -= 1
            job.cancel()
            job.coro.close()
            self._finish(job, exc=JobCancelled("Bot is shutting down"))
        if self._pending == 0 and self._active == 0:
            self._drained.set()
        worker = self._worker_task
        if self._current is not None:
            self._current.cancel()
        if worker is not None and not worker.done():
            try:
                await asyncio.wait_for(self.wait_drained(), timeout=timeout)
            except asyncio.TimeoutError:
                log.warning("queue did not drain within %ss; stopping the worker", timeout)
            if not worker.done():
                worker.cancel()

    def stats(self) -> tuple[int, int]:
        """Return ``(active, pending)`` counts."""
        return self._active, self._pending


class JobQueueManager:
    """Routes jobs into serial lanes based on the configured mode.

    - ``unified``: one single serial queue for every job. A ComfyUI generation
      and an LLM prompt can never run at the same time (maximum 1 concurrent
      job). This is the safest mode against resource collapse.

    - ``separate``: two independent serial lanes — ``comfyui`` and ``llm``.
      Each lane runs its jobs one at a time, but the two lanes run in parallel,
      so a ComfyUI image generation and an LLM prompt may run concurrently
      (maximum 2 concurrent jobs, one per resource).
    """

    def __init__(self, mode: str = "unified"):
        self.mode = mode
        if mode == "separate":
            self._lanes = {"comfyui": JobQueue(), "llm": JobQueue()}
        else:
            self.mode = "unified"
            self._lanes = {"unified": JobQueue()}
        self._cancel_hooks: dict[str, Callable[[], Awaitable[None]]] = {}

    def _lane_key(self, lane: str) -> str:
        """Resolve which lane a job goes into for the current mode."""
        if self.mode == "separate" and lane in self._lanes:
            return lane
        # unified mode has a single lane; separate mode falls back for unknown lanes
        return next(iter(self._lanes))

    def submit(self, coro, lane: str = "comfyui", name: str = "job", **options) -> Job:
        """Queue ``coro`` on the given lane. Returns a Job for its result.

        ``on_cancel`` defaults to the hook registered for the *requested* lane,
        so an LLM job never stops a ComfyUI prompt even when both lanes share
        one queue in unified mode.
        """
        if options.get("on_cancel") is None:
            options.pop("on_cancel", None)
            hook = self._cancel_hooks.get(lane)
            if hook is not None:
                options["on_cancel"] = hook
        return self._lanes[self._lane_key(lane)].submit(coro, name, **options)

    def set_cancel_hook(self, lane: str, callback) -> None:
        """Set what ``on_cancel`` defaults to for jobs submitted on ``lane``.

        The hook is how a lane stops its external work (e.g. ComfyUI's
        currently sampling prompt) when a job is cancelled or times out.
        """
        self._cancel_hooks[lane] = callback

    async def wait_drained(self) -> None:
        """Await until every lane has no running or queued jobs."""
        for q in self._lanes.values():
            await q.wait_drained()

    async def shutdown(self, timeout: float = 5.0) -> None:
        """Stop every lane: drop queued jobs and interrupt the running one."""
        for q in self._lanes.values():
            await q.shutdown(timeout)

    def total_jobs(self) -> int:
        """Total number of jobs (running or queued) across all lanes."""
        total = 0
        for q in self._lanes.values():
            active, pending = q.stats()
            total += active + pending
        return total

    def position(self, lane: str) -> int:
        """1-based position of the next job among the waiting jobs on ``lane``.

        Only jobs queued *ahead* of the next job are counted (the currently
        running job does not occupy a waiting slot). The first waiting job is #1.
        """
        _, pending = self._lanes[self._lane_key(lane)].stats()
        return pending + 1

    def lane_stats(self, lane: str) -> tuple[int, int]:
        """Return ``(active, pending)`` counts for ``lane``."""
        return self._lanes[self._lane_key(lane)].stats()

    def waiting_prefix(self, lane: str) -> str:
        """Text to prepend to a progress message when the job must wait.

        Returns an empty string when the job will start immediately (nothing
        running and nothing queued). Otherwise shows the 1-based position in
        line, where #1 is the first waiting job.
        """
        active, pending = self._lanes[self._lane_key(lane)].stats()
        if active == 0 and pending == 0:
            return ""
        return f"\u23f3 You're #{pending + 1} in line.\n\n"

    def live_position(self, lane: str) -> int | None:
        """1-based position of a job that is *already queued* on ``lane``.

        ``pending`` counts waiting jobs including the job itself, so its
        position equals ``pending`` (jobs ahead + 1). Returns ``None`` when
        the lane has no waiting jobs.
        """
        _, pending = self._lanes[self._lane_key(lane)].stats()
        return pending if pending > 0 else None

    def register(self, lane: str, callback) -> None:
        """Register a listener on the lane, notified on job start/finish."""
        self._lanes[self._lane_key(lane)].register(callback)

    def unregister(self, lane: str, callback) -> None:
        self._lanes[self._lane_key(lane)].unregister(callback)


def _resolve_mode() -> str:
    mode = str(config.get("queueing", {}).get("mode", "unified")).strip().lower()
    if mode not in ("unified", "separate"):
        log.warning("Unknown queueing mode %r; falling back to 'unified'", mode)
        mode = "unified"
    return mode


# Global manager shared by all resource-heavy operations. Created once at
# import time from the config's ``queueing.mode`` (default "unified").
job_queue = JobQueueManager(_resolve_mode())
