import asyncio
import logging

from core import config

log = logging.getLogger("job_queue")


class JobQueue:
    """A single-slot, FIFO queue for long-running, resource-heavy jobs.

    Jobs are coroutines. They are executed strictly one at a time, in the
    order they were submitted. ``submit`` returns an ``asyncio.Future`` that
    resolves to the job's return value (or raises the job's exception), so
    callers can ``await`` the Future to get the result. ``wait_drained``
    blocks until every submitted job has finished.
    """

    def __init__(self):
        self._queue = asyncio.Queue()
        self._worker_task = None
        self._pending = 0
        self._active = 0
        self._drained = asyncio.Event()
        self._drained.set()

    def _ensure_worker(self) -> None:
        """Start the single worker task if it is not already running."""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.ensure_future(self._worker())

    async def _worker(self) -> None:
        """Consume jobs from the queue one at a time, resolving each future."""
        while True:
            name, coro, done = await self._queue.get()
            # Job moves from the queue to the running slot.
            self._pending -= 1
            self._active += 1
            try:
                result = await coro
            except Exception as exc:
                done.set_exception(exc)
            else:
                done.set_result(result)
            finally:
                self._active -= 1
                if self._pending == 0 and self._active == 0:
                    self._drained.set()

    def submit(self, coro, name: str = "job") -> asyncio.Future:
        """Queue ``coro`` to run serially.

        Returns a Future that resolves to the coroutine's return value, or
        raises the coroutine's exception. The job runs only after any earlier
        submitted jobs have completed.
        """
        loop = asyncio.get_running_loop()
        done = loop.create_future()
        self._queue.put_nowait((name, coro, done))
        self._pending += 1
        self._drained.clear()
        self._ensure_worker()
        return done

    async def wait_drained(self) -> None:
        """Await until no job is running and none are queued."""
        await self._drained.wait()

    def stats(self) -> tuple[int, int]:
        """Return ``(pending, active)`` counts."""
        return self._pending, self._active


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

    def _lane_key(self, lane: str) -> str:
        """Resolve which lane a job goes into for the current mode."""
        if self.mode == "separate" and lane in self._lanes:
            return lane
        # unified mode has a single lane; separate mode falls back for unknown lanes
        return next(iter(self._lanes))

    def submit(self, coro, lane: str = "comfyui", name: str = "job") -> asyncio.Future:
        """Queue ``coro`` on the given lane. Returns a Future for its result."""
        return self._lanes[self._lane_key(lane)].submit(coro, name)

    async def wait_drained(self) -> None:
        """Await until every lane has no running or queued jobs."""
        for q in self._lanes.values():
            await q.wait_drained()

    def position(self, lane: str) -> int:
        """1-based position the next job on ``lane`` would occupy.

        1 means it will start immediately (nothing running, nothing queued).
        """
        active, pending = self._lanes[self._lane_key(lane)].stats()
        return active + pending + 1

    def waiting_prefix(self, lane: str) -> str:
        """Text to prepend to a progress message when the job must wait.

        Returns an empty string when the job will start immediately.
        """
        pos = self.position(lane)
        if pos <= 1:
            return ""
        return f"\u23f3 You're #{pos} in line.\n\n"


def _resolve_mode() -> str:
    mode = str(config.get("queueing", {}).get("mode", "unified")).strip().lower()
    if mode not in ("unified", "separate"):
        log.warning("Unknown queueing mode %r; falling back to 'unified'", mode)
        mode = "unified"
    return mode


# Global manager shared by all resource-heavy operations. Created once at
# import time from the config's ``queueing.mode`` (default "unified").
job_queue = JobQueueManager(_resolve_mode())
