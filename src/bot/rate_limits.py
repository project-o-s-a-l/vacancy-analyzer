"""Per-bot network concurrency plus global and per-chat pacing, without retries."""

import asyncio
import logging
import time
from collections import OrderedDict

from aiogram.client.session.middlewares.base import BaseRequestMiddleware

from observability import emit_event


class TelegramLimits(BaseRequestMiddleware):
    def __init__(self, *, concurrency=4, rate=20, chat_interval=1.05):
        self.network = asyncio.Semaphore(concurrency)
        self.interval, self.chat_interval = 1 / rate, chat_interval
        self.next_global = 0
        self.chats = OrderedDict()

    async def __call__(self, make_request, bot, method):
        chat = getattr(method, "chat_id", None)
        # Long polling and administrative requests have separate HTTP connections.
        if chat is None:
            return await make_request(bot, method)
        queued = time.monotonic()
        while True:
            due = max(self.next_global, self.chats.get(chat, 0))
            await asyncio.sleep(max(0, due - time.monotonic()))
            async with self.network:
                started = time.monotonic()
                # Recheck after acquiring a socket slot. A long HTTP request must not
                # accumulate expired permits and create a burst when it completes.
                if started < max(self.next_global, self.chats.get(chat, 0)):
                    continue
                self.next_global = started + self.interval
                self.chats[chat] = started + self.chat_interval
                self.chats.move_to_end(chat)
                while len(self.chats) > 4096:
                    self.chats.popitem(last=False)
                try:
                    return await make_request(bot, method)
                finally:
                    emit_event(
                        logging.getLogger(__name__),
                        "delivery.http_finished",
                        wait_ms=(started - queued) * 1000,
                        duration_ms=(time.monotonic() - started) * 1000,
                    )
