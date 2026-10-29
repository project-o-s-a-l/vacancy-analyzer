import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bot.rate_limits import TelegramLimits
from jobs.owner_queue import OwnerQueue, QueueFull
from jobs.serving_compute import compute


async def test_owner_fifo_fairness_and_bounded_mailboxes():
    queue = OwnerQueue(workers=2, capacity=5, per_owner=3)
    entered, release = asyncio.Event(), asyncio.Event()
    order = []

    async def slow():
        entered.set()
        await release.wait()
        order.append("upload")

    async def command(name):
        order.append(name)

    first = queue.submit(1, slow)
    await entered.wait()
    second = queue.submit(1, lambda: command("confirm"))
    third = queue.submit(1, lambda: command("latest"))
    with pytest.raises(QueueFull):
        queue.submit(1, lambda: command("overflow"))
    other = queue.submit(2, lambda: command("other"))
    await asyncio.wait_for(other, 1)
    assert order == ["other"] and not second.done()
    release.set()
    await asyncio.gather(first, second, third)
    await queue.close()
    assert order == ["other", "upload", "confirm", "latest"]
    assert not queue.owners and queue.count == 0


async def test_queue_shutdown_cancels_running_and_never_runs_waiting_commands():
    queue = OwnerQueue(workers=1)
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def slow():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    running = queue.submit(1, slow)
    waiting_call = AsyncMock()
    waiting = queue.submit(1, waiting_call)
    await entered.wait()
    await queue.close(grace=0.01)
    assert cancelled.is_set() and running.cancelled() and waiting.cancelled()
    waiting_call.assert_not_called()
    with pytest.raises(QueueFull):
        queue.submit(2, waiting_call)


async def test_cancel_waiting_command_does_not_break_next_command():
    queue = OwnerQueue(workers=1)
    release = asyncio.Event()
    first = queue.submit(1, release.wait)
    skipped = AsyncMock()
    second = queue.submit(1, skipped)
    third = queue.submit(1, AsyncMock(return_value=3))
    second.cancel()
    release.set()
    await first
    assert await third == 3
    skipped.assert_not_called()
    await queue.close()


async def test_cpu_cancel_keeps_slot_until_real_thread_finishes():
    import threading

    entered, release = threading.Event(), threading.Event()

    def work():
        entered.set()
        release.wait(2)
        return 3

    task = asyncio.create_task(compute(work))
    await asyncio.wait_for(asyncio.to_thread(entered.wait), 2)
    task.cancel()
    await asyncio.sleep(0.02)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await compute(lambda: 4) == 4


async def test_slow_chat_rate_wait_does_not_delay_another_chat():
    limits = TelegramLimits(rate=1000, chat_interval=0.2)
    sender = AsyncMock(return_value=True)
    await limits(sender, None, SimpleNamespace(chat_id=1))
    slow = asyncio.create_task(limits(sender, None, SimpleNamespace(chat_id=1)))
    await asyncio.sleep(0.005)
    before = time.monotonic()
    await limits(sender, None, SimpleNamespace(chat_id=2))
    assert time.monotonic() - before < 0.1 and not slow.done()
    await slow


@pytest.mark.postgres
async def test_polling_routes_ordered_form_commands_while_other_owner_runs(postgres_sessions):
    from aiogram import Bot
    from test_bot import TEST_TOKEN, RecordingSession, command_update

    from bot.handlers import build_dispatcher
    from jobs.users import UserService

    users = UserService(postgres_sessions)
    dispatcher = build_dispatcher(users)
    bot = Bot(TEST_TOKEN, session=RecordingSession())
    entered, release = asyncio.Event(), asyncio.Event()
    original = users.register

    async def delayed(owner):
        if owner == 42:
            entered.set()
            await release.wait()
        return await original(owner)

    users.register = delayed

    async def updates(*args, **kwargs):
        yield command_update("/start")
        answer = command_update("Python")
        yield answer.model_copy(
            update={"update_id": 2, "message": answer.message.model_copy(update={"entities": []})}
        )
        yield command_update("/start", user_id=99, chat_id=99).model_copy(update={"update_id": 3})
        await entered.wait()
        for _ in range(100):
            if await users.get(99):
                break
            await asyncio.sleep(0.01)
        assert await users.get(99) is not None and await users.get(42) is None
        release.set()

    dispatcher._listen_updates = updates
    await dispatcher._polling(bot)
    user = await users.get(42)
    assert "python" in {s.lower() for s in user.profile.skills}
    assert user.form_state == "setup:experience"
    await bot.session.close()
