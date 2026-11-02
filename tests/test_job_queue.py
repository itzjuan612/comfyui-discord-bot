"""Tests for queue lanes: ordering, cancellation and time budgets."""

import asyncio
import logging

import pytest

from job_queue import (
    JobCancelled,
    JobQueue,
    JobQueuedTimeout,
    JobQueueManager,
    JobRunTimeout,
    job_error_text,
    log_job_error,
)


def test_jobs_run_serially_in_submission_order():
    async def scenario():
        queue = JobQueue()
        order = []

        async def work(n):
            order.append(("start", n))
            await asyncio.sleep(0.01)
            order.append(("end", n))
            return n

        jobs = [queue.submit(work(n), name=str(n)) for n in range(3)]
        assert await asyncio.gather(*jobs) == [0, 1, 2]
        assert order == [
            ("start", 0),
            ("end", 0),
            ("start", 1),
            ("end", 1),
            ("start", 2),
            ("end", 2),
        ]

    asyncio.run(scenario())


def test_job_returns_value_and_reraises_errors():
    async def scenario():
        queue = JobQueue()

        async def ok():
            return "done"

        async def boom():
            raise ValueError("nope")

        assert await queue.submit(ok()) == "done"
        with pytest.raises(ValueError, match="nope"):
            await queue.submit(boom())

    asyncio.run(scenario())


def test_cancelled_queued_job_never_runs():
    async def scenario():
        queue = JobQueue()
        started = []

        async def work(name):
            started.append(name)
            await asyncio.sleep(0.02)
            return name

        first = queue.submit(work("a"), name="a")
        second = queue.submit(work("b"), name="b")
        second.cancel()

        assert await first == "a"
        with pytest.raises(JobCancelled):
            await second
        assert started == ["a"]

    asyncio.run(scenario())


def test_cancelling_a_queued_job_frees_it_before_the_running_one_ends():
    async def scenario():
        queue = JobQueue()
        started = []
        dropped = asyncio.Event()

        async def long_job():
            started.append("long")
            await asyncio.sleep(0.3)

        async def queued_job():
            started.append("queued")

        async def listener(event, coro):
            # The jobs behind move up on this event, not on the worker's next tick.
            if event == "finish" and coro is queued.coro:
                dropped.set()

        queue.register(listener)
        running = queue.submit(long_job(), name="running")
        queued = queue.submit(queued_job(), name="queued")
        await asyncio.sleep(0.01)  # let the first job start

        queued.cancel()
        queued.cancel()  # idempotent

        # Fails now, while the first job is still running.
        with pytest.raises(JobCancelled):
            await asyncio.wait_for(queued, timeout=0.1)
        assert queue.stats() == (1, 0)
        assert started == ["long"]  # the cancelled job never started
        await asyncio.wait_for(dropped.wait(), timeout=1)
        assert queued.coro.cr_frame is None  # coroutine closed, not leaked

        await running

    asyncio.run(scenario())


def test_queued_timeout_drops_a_stale_job():
    async def scenario():
        queue = JobQueue()

        async def work():
            await asyncio.sleep(0.02)
            return "ok"

        first = queue.submit(work(), name="first")
        stale = queue.submit(work(), name="stale", queued_timeout=0.01)

        assert await first == "ok"
        with pytest.raises(JobQueuedTimeout):
            await stale

    asyncio.run(scenario())


def test_run_timeout_cancels_the_job_and_calls_on_cancel():
    async def scenario():
        queue = JobQueue()
        stopped = []

        async def forever():
            await asyncio.sleep(30)

        async def on_cancel():
            stopped.append(True)

        job = queue.submit(forever(), name="forever", run_timeout=0.02, on_cancel=on_cancel)
        with pytest.raises(JobRunTimeout):
            await job
        assert stopped == [True]

    asyncio.run(scenario())


def test_cancel_while_running_calls_on_cancel():
    async def scenario():
        queue = JobQueue()
        stopped = []

        async def forever():
            await asyncio.sleep(30)

        async def on_cancel():
            stopped.append(True)

        job = queue.submit(forever(), name="forever", on_cancel=on_cancel)
        await asyncio.sleep(0.01)
        job.cancel()
        with pytest.raises(JobCancelled):
            await job
        assert stopped == [True]

    asyncio.run(scenario())


def test_lane_cancel_hook_is_the_default_on_cancel():
    async def scenario():
        queue = JobQueueManager("unified")
        stopped = []

        async def forever():
            await asyncio.sleep(30)

        async def hook():
            stopped.append("comfyui")

        queue.set_cancel_hook("comfyui", hook)
        # An LLM job shares the unified lane but must not inherit that hook.
        llm_job = queue.submit(forever(), lane="llm", run_timeout=0.02)
        with pytest.raises(JobRunTimeout):
            await llm_job
        assert stopped == []

        comfy_job = queue.submit(forever(), lane="comfyui", run_timeout=0.02)
        with pytest.raises(JobRunTimeout):
            await comfy_job
        assert stopped == ["comfyui"]

    asyncio.run(scenario())


def test_failing_on_cancel_does_not_swallow_the_timeout():
    async def scenario():
        queue = JobQueue()

        async def forever():
            await asyncio.sleep(30)

        async def broken_hook():
            raise RuntimeError("interrupt failed")

        job = queue.submit(forever(), run_timeout=0.02, on_cancel=broken_hook)
        with pytest.raises(JobRunTimeout):
            await job

    asyncio.run(scenario())


def test_cancel_hook_offers_to_stop_the_resource_before_the_coroutine_is_collected():
    async def scenario():
        queue = JobQueue()
        order = []

        async def forever():
            try:
                await asyncio.sleep(30)
            finally:
                # Slow cleanup: the interrupt must not wait for this.
                await asyncio.sleep(0.05)
                order.append("cleanup")

        async def hook():
            order.append("interrupt")

        job = queue.submit(forever(), run_timeout=0.02, on_cancel=hook)
        with pytest.raises(JobRunTimeout):
            await job
        assert order == ["interrupt", "cleanup"]

    asyncio.run(scenario())


def test_cancel_hook_is_bounded():
    async def scenario():
        queue = JobQueue()
        queue.CANCEL_HOOK_TIMEOUT = 0.02

        async def forever():
            await asyncio.sleep(30)

        async def hanging_hook():
            await asyncio.sleep(30)

        job = queue.submit(forever(), run_timeout=0.01, on_cancel=hanging_hook)
        with pytest.raises(JobRunTimeout):
            await asyncio.wait_for(job, timeout=5)

    asyncio.run(scenario())


def test_shutdown_drops_queued_jobs_and_stops_the_running_one():
    async def scenario():
        queue = JobQueue()
        stopped = []

        async def forever():
            await asyncio.sleep(30)

        async def hook():
            stopped.append(True)

        running = queue.submit(forever(), on_cancel=hook)
        queued = queue.submit(forever())
        await asyncio.sleep(0.01)  # let the first job start

        await queue.shutdown(timeout=5)

        assert stopped == [True]
        with pytest.raises(JobCancelled):
            await running
        with pytest.raises(JobCancelled):
            await queued
        assert queue.stats() == (0, 0)

    asyncio.run(scenario())


def test_shutdown_on_an_idle_queue_returns_immediately():
    async def scenario():
        queue = JobQueue()
        await asyncio.wait_for(queue.shutdown(timeout=1), timeout=2)
        assert queue.stats() == (0, 0)

    asyncio.run(scenario())


def test_listeners_see_start_and_finish():
    async def scenario():
        queue = JobQueue()
        events = []
        finished = asyncio.Event()

        async def listener(event, coro):
            events.append(event)
            if event == "finish":
                finished.set()

        async def work():
            return 1

        queue.register(listener)
        await queue.submit(work())
        # Listeners are dispatched in their own task, so wait for the event
        # instead of guessing a delay.
        await asyncio.wait_for(finished.wait(), timeout=1)
        assert events == ["start", "finish"]

    asyncio.run(scenario())


def test_wait_drained_returns_when_idle():
    async def scenario():
        queue = JobQueue()

        async def work():
            await asyncio.sleep(0.01)

        await queue.submit(work())
        await queue.wait_drained()
        assert queue.stats() == (0, 0)

    asyncio.run(scenario())


def test_job_done_flag_tracks_completion():
    async def scenario():
        queue = JobQueue()
        gate = asyncio.Event()

        async def work():
            await gate.wait()
            return 1

        job = queue.submit(work())
        assert not job.done
        gate.set()
        assert await job == 1
        assert job.done

    asyncio.run(scenario())


def test_error_text_reports_cancellations_without_the_failure_prefix():
    prefix = "\u274c Generation failed:"
    assert job_error_text(prefix, JobCancelled("Cancelled")) == "\u23f9\ufe0f Cancelled."
    assert (
        job_error_text(prefix, JobQueuedTimeout("Timed out waiting in the queue"))
        == "\u23f2\ufe0f Timed out waiting in the queue."
    )
    assert (
        job_error_text(prefix, JobRunTimeout("Stopped after exceeding the 300s time limit"))
        == "\u23f2\ufe0f Stopped after exceeding the 300s time limit."
    )


def test_error_text_keeps_the_prefix_for_real_failures():
    assert (
        job_error_text("\u274c Generation failed:", ValueError("boom"))
        == "\u274c Generation failed: boom"
    )


def test_log_job_error_downgrades_cancellations(caplog):
    caplog.set_level(logging.INFO)
    try:
        raise JobCancelled("Cancelled")
    except JobCancelled as exc:
        log_job_error(logging.getLogger("job_queue"), "generation failed", exc)
    record = caplog.records[-1]
    assert record.levelno == logging.INFO
    assert record.getMessage() == "generation failed: Cancelled"


def test_log_job_error_keeps_the_traceback_for_real_failures(caplog):
    try:
        raise ValueError("boom")
    except ValueError as exc:
        log_job_error(logging.getLogger("job_queue"), "generation failed", exc)
    record = caplog.records[-1]
    assert record.levelno == logging.ERROR
    assert record.exc_info is not None
