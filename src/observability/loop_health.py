"""Bounded event-loop heartbeat, independent of request and transport latency."""

import asyncio
import logging
from contextlib import asynccontextmanager
from time import monotonic

from observability import emit_event


@asynccontextmanager
async def monitor_loop():
    async def monitor():
        samples = []
        while True:
            before = monotonic()
            await asyncio.sleep(0.1)
            samples.append(max(0, monotonic() - before - 0.1) * 1000)
            if len(samples) >= 300:
                values = sorted(samples)
                emit_event(
                    logging.getLogger(__name__),
                    "serving.loop_health",
                    result="success",
                    duration_ms=values[-1],
                    wait_ms=values[int(len(values) * 0.95) - 1],
                )
                samples.clear()

    task = asyncio.create_task(monitor())
    try:
        yield
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
