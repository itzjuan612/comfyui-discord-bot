"""Tests for queue lanes: ordering, cancellation and time budgets."""

import asyncio

import pytest

from job_queue import (
    JobCancelled,
    JobQueue,
    JobQueuedTimeout,
    JobQueueManager,
    JobRunTimeout,
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
            ("start", 0), ("end", 0),
            ("start", 1), ("end", 1),
            ("start", 2), ("end", 2),
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

        job = queue.submit(forever(), name="forever", run_timeout=0.02,
                           on_cancel=on_cancel)
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
