import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bot.rate_limits import TelegramLimits
from jobs.serving_compute import compute


async def test_committed_updates_survive_restart_preserve_order_and_expire_private_data(
    db_sessions,
):
    from datetime import UTC, datetime, timedelta

    from test_bot import command_update

    from bot.update_inbox import UpdateInbox
    from storage.models import TelegramUpdate

    inbox = UpdateInbox(db_sessions, 19)
    event = command_update("/start")
    assert await inbox.receive(42, event) == "accepted"
    assert await inbox.receive(42, event) == "duplicate"
    for key in range(2, 9):
        assert await inbox.receive(42, event.model_copy(update={"update_id": key})) == "accepted"
    assert await inbox.receive(42, event.model_copy(update={"update_id": 9})) == "overflow"
    other = command_update("/start", user_id=99, chat_id=99).model_copy(update={"update_id": 10})
    assert await inbox.receive(99, other) == "accepted"

    restarted = UpdateInbox(db_sessions, 19)
    first = await restarted.claim()
    second = await restarted.claim()
    async with db_sessions() as session:
        claimed = await session.get(TelegramUpdate, (19, 1))
        assert claimed.payload is None and claimed.owner_id == 42
    assert (first[0], second[0]) == (1, 10) and await restarted.claim() is None
    assert await restarted.change(1, first[3], ready_at=datetime.now(UTC) - timedelta(seconds=1))
    interrupted = await restarted.claim()
    assert interrupted[-1] and interrupted[0] == 1
    bot = SimpleNamespace(send_message=AsyncMock())
    dispatcher = SimpleNamespace(feed_update=AsyncMock())
    await restarted.handle(interrupted, dispatcher, bot, {})
    dispatcher.feed_update.assert_not_called()
    bot.send_message.assert_awaited_once()
    assert await inbox.receive(42, event) == "duplicate"
    next_job = await restarted.claim()
    assert next_job[0] == 2
    # A permanently unreachable chat must not strand the rest of its mailbox.
    from aiogram.exceptions import TelegramForbiddenError
    from aiogram.methods import SendMessage

    bot.send_message.side_effect = TelegramForbiddenError(
        method=SendMessage(chat_id=42, text="test"), message="forbidden"
    )
    await restarted.change(2, next_job[3], ready_at=datetime.now(UTC) - timedelta(seconds=1))
    await restarted.handle(await restarted.claim(), dispatcher, bot, {})
    assert (await restarted.claim())[0] == 3
    async with db_sessions() as session:
        done = await session.get(TelegramUpdate, (19, 1))
        assert done.owner_id is None and done.payload is None


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
