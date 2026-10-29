"""Bounded fair FIFO per owner. Waiting commands never occupy active-owner slots."""

import asyncio
import logging
import time
from collections import deque

from observability import emit_event


class QueueFull(ValueError):
    pass


class OwnerQueue:
    def __init__(self, *, workers=4, capacity=128, per_owner=8):
        self.limit, self.capacity, self.per_owner = workers, capacity, per_owner
        self.owners, self.ready, self.workers = {}, asyncio.Queue(), []
        self.count = 0
        self.closing = False
        self.idle = asyncio.Event()
        self.idle.set()

    def submit(self, owner, call):
        if (
            self.closing
            or self.count >= self.capacity
            or len(self.owners.get(owner, ())) >= self.per_owner
        ):
            raise QueueFull("Сейчас слишком много команд. Дождитесь ответа и повторите позже.")
        if not self.workers:
            self.workers = [asyncio.create_task(self._worker()) for _ in range(self.limit)]
        future = asyncio.get_running_loop().create_future()
        if owner not in self.owners:
            self.owners[owner] = deque()
            self.ready.put_nowait(owner)
        self.owners[owner].append((call, future, time.monotonic()))
        self.count += 1
        self.idle.clear()
        return future

    async def _worker(self):
        while True:
            owner = await self.ready.get()
            queue = self.owners[owner]
            call, future, queued_at = queue[0]
            started = time.monotonic()
            task = None
            try:
                if not future.cancelled():
                    task = asyncio.create_task(call())

                    def cancel_child(done, task=task):
                        if done.cancelled():
                            task.cancel()

                    future.add_done_callback(cancel_child)
                    value = await task
                    if not future.done():
                        future.set_result(value)
            except asyncio.CancelledError:
                future.cancel()
                if asyncio.current_task().cancelling():
                    raise
            except Exception as exc:
                if not future.done():
                    future.set_exception(exc)
            finally:
                if task is not None:
                    future.remove_done_callback(cancel_child)
                queue.popleft()
                self.count -= 1
                if queue:
                    self.ready.put_nowait(owner)
                else:
                    del self.owners[owner]
                if not self.count:
                    self.idle.set()
                emit_event(
                    logging.getLogger(__name__),
                    "serving.command_finished",
                    wait_ms=(started - queued_at) * 1000,
                    duration_ms=(time.monotonic() - started) * 1000,
                    result="cancelled" if future.cancelled() else "success",
                )

    async def close(self, grace=20):
        self.closing = True
        try:
            await asyncio.wait_for(self.idle.wait(), timeout=grace)
        except TimeoutError:
            pass
        finally:
            for worker in self.workers:
                worker.cancel()
            await asyncio.gather(*self.workers, return_exceptions=True)
            for queue in self.owners.values():
                for _, future, _ in queue:
                    future.cancel()
            self.owners.clear()
            self.count = 0
            self.idle.set()
