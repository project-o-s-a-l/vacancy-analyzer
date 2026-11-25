"""Long-poll adapter: bounded owner mailboxes, not one task per incoming update."""

import asyncio

from aiogram import Dispatcher
from aiogram.dispatcher.dispatcher import DEFAULT_BACKOFF_CONFIG
from aiogram.types import Message

from bot.update_inbox import UpdateInbox
from storage.database import DATABASE_ERRORS


def owner_for(update):
    event = update.message or update.callback_query
    message = event if isinstance(event, Message) else getattr(event, "message", None)
    user = getattr(event, "from_user", None)
    if (
        not isinstance(message, Message)
        or message.chat.type != "private"
        or user is None
        or user.is_bot
        or message.chat.id != user.id
    ):
        return None
    return user.id


class OrderedDispatcher(Dispatcher):
    def __init__(self, sessions, **kwargs):
        super().__init__(**kwargs)
        self.sessions = sessions

    async def _polling(
        self,
        bot,
        polling_timeout=30,
        handle_as_tasks=True,
        backoff_config=DEFAULT_BACKOFF_CONFIG,
        allowed_updates=None,
        tasks_concurrency_limit=None,
        **kwargs,
    ):
        inbox = UpdateInbox(self.sessions, bot.id)
        notices = asyncio.Queue(maxsize=16)
        notified = set()

        async def send_notices():
            while True:
                owner = await notices.get()
                try:
                    await inbox.reject(bot, owner)
                finally:
                    notified.discard(owner)
                    notices.task_done()

        async with asyncio.TaskGroup() as group:
            workers = [group.create_task(inbox.run(self, bot, kwargs)) for _ in range(4)]
            workers.append(group.create_task(send_notices()))
            try:
                async for item in self._listen_updates(
                    bot,
                    polling_timeout=polling_timeout,
                    backoff_config=backoff_config,
                    allowed_updates=allowed_updates,
                ):
                    owner = owner_for(item)
                    if owner is None:
                        continue
                    # The next getUpdates offset acknowledges only committed receipts.
                    while True:
                        try:
                            accepted = await inbox.receive(owner, item)
                        except DATABASE_ERRORS:
                            accepted = "busy"
                        if accepted == "overflow" and owner not in notified and not notices.full():
                            notified.add(owner)
                            notices.put_nowait(owner)
                        if accepted != "busy":
                            break
                        await asyncio.sleep(1)
                await inbox.drain()
            finally:
                for worker in workers:
                    worker.cancel()
