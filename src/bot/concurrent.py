"""Long-poll adapter: bounded owner mailboxes, not one task per incoming update."""

import asyncio

from aiogram import Dispatcher
from aiogram.dispatcher.dispatcher import DEFAULT_BACKOFF_CONFIG
from aiogram.types import Message

from jobs.owner_queue import OwnerQueue, QueueFull


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
        queue = OwnerQueue()
        # Rejections have their own bounded lane; a flood cannot allocate unbounded tasks
        # or stop accepting commands from other owners while Telegram is slow.
        notices = asyncio.Queue(maxsize=16)
        notified = set()

        def notice(owner, message):
            if not notices.full() and owner not in notified:
                notices.put_nowait((owner, message))
                notified.add(owner)

        async def notify():
            while True:
                owner, message = await notices.get()
                try:
                    await bot.send_message(owner, message)
                except Exception:
                    pass
                finally:
                    notified.discard(owner)

        notifier = asyncio.create_task(notify())
        try:
            async for update in self._listen_updates(
                bot,
                polling_timeout=polling_timeout,
                backoff_config=backoff_config,
                allowed_updates=allowed_updates,
            ):
                owner = owner_for(update)
                if owner is None:
                    continue
                try:
                    future = queue.submit(
                        owner,
                        lambda update=update: self._process_update(
                            bot=bot, update=update, **kwargs
                        ),
                    )
                    future.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
                    if queue.count > queue.limit or len(queue.owners[owner]) > 1:
                        notice(owner, "Команда принята. Дождитесь завершения предыдущих команд.")
                except QueueFull:
                    notice(
                        owner, "Сейчас слишком много команд. Дождитесь ответа и повторите позже."
                    )
        finally:
            await queue.close()
            notifier.cancel()
            await asyncio.gather(notifier, return_exceptions=True)
