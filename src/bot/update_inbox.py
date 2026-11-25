"""Receive, claim and finish private updates in separate short transactions."""

import asyncio
import logging
import time
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError
from aiogram.methods import TelegramMethod
from aiogram.types import Update
from sqlalchemy import and_, delete, func, or_, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import aliased

from bot.common import COMMAND_FAILED
from observability import emit_event
from storage.database import DATABASE_ERRORS
from storage.models import TelegramUpdate as Entry

LEASE = timedelta(minutes=2)
RETENTION = timedelta(days=2)
logger = logging.getLogger(__name__)


def private_payload(item):
    # Callback messages may contain an entire historical card or resume. Routing only
    # needs its Telegram identity and keyboard. A new message needs its command/document.
    user = {"id": True, "is_bot": True, "first_name": True, "username": True}
    message = {
        "message_id": True,
        "date": True,
        "chat": {"id": True, "type": True},
        "from_user": user,
        "reply_markup": True,
    }
    fields = {"update_id": True}
    if item.message:
        fields["message"] = {
            **message,
            "text": True,
            "entities": True,
            "caption": True,
            "caption_entities": True,
            "document": {
                "file_id": True,
                "file_unique_id": True,
                "file_name": True,
                "file_size": True,
                "mime_type": True,
            },
        }
    else:
        fields["callback_query"] = {
            "id": True,
            "data": True,
            "chat_instance": True,
            "from_user": user,
            "message": message,
        }
    return item.model_dump(mode="json", by_alias=True, exclude_none=True, include=fields)


class UpdateInbox:
    def __init__(self, sessions, bot_id):
        self.sessions = sessions
        self.bot_id = bot_id
        self.wake = asyncio.Event()

    async def receive(self, owner, update_):
        now = datetime.now(UTC)
        async with self.sessions.begin() as session:
            dialect = session.get_bind().dialect.name
            if dialect == "postgresql":
                # Keep receipt deduplication and capacity checks atomic across poller restarts.
                await session.execute(
                    text("SELECT pg_advisory_xact_lock(hashtextextended(:scope, 0))"),
                    {"scope": f"telegram-inbox:{self.bot_id}"},
                )
            insert = pg_insert if dialect == "postgresql" else sqlite_insert
            await session.execute(
                delete(Entry).where(
                    Entry.bot_id == self.bot_id,
                    Entry.ready_at.is_(None),
                    Entry.received_at < now - RETENTION,
                )
            )
            if await session.get(Entry, (self.bot_id, update_.update_id)) is not None:
                return "duplicate"
            pending, per_owner = (
                await session.execute(
                    select(func.count(), func.count().filter(Entry.owner_id == owner))
                    .select_from(Entry)
                    .where(Entry.bot_id == self.bot_id, Entry.ready_at.is_not(None))
                )
            ).one()
            full = per_owner >= 8
            if pending >= 128 and not full:
                return "busy"
            await session.execute(
                insert(Entry)
                .values(
                    bot_id=self.bot_id,
                    update_id=update_.update_id,
                    owner_id=None if full else owner,
                    received_at=now,
                    ready_at=None if full else now,
                    attempts=0,
                    payload=None if full else private_payload(update_),
                )
                .on_conflict_do_nothing(index_elements=["bot_id", "update_id"])
            )
        self.wake.set()
        return "overflow" if full else "accepted"

    async def drain(self):
        try:
            async with asyncio.timeout(20):
                while True:
                    async with self.sessions() as session:
                        if not await session.scalar(
                            select(
                                select(Entry.update_id)
                                .where(Entry.bot_id == self.bot_id, Entry.ready_at.is_not(None))
                                .exists()
                            )
                        ):
                            return
                    await asyncio.sleep(0.05)
        except DATABASE_ERRORS:
            return

    @staticmethod
    async def reject(bot, owner):
        try:
            await bot.send_message(
                owner, "Сейчас слишком много команд. Дождитесь ответа и повторите позже."
            )
        except Exception as exc:
            emit_event(
                logger,
                "serving.command_finished",
                level=logging.WARNING,
                error=exc,
                result="failed",
            )

    async def claim(self):
        now = datetime.now(UTC)
        earlier = aliased(Entry)
        waiting = (
            select(earlier.update_id)
            .where(
                earlier.bot_id == Entry.bot_id,
                earlier.owner_id == Entry.owner_id,
                earlier.ready_at.is_not(None),
                or_(
                    earlier.received_at < Entry.received_at,
                    and_(
                        earlier.received_at == Entry.received_at,
                        earlier.update_id < Entry.update_id,
                    ),
                ),
            )
            .exists()
        )
        async with self.sessions.begin() as session:
            row = await session.scalar(
                select(Entry)
                .where(Entry.bot_id == self.bot_id, Entry.ready_at <= now, ~waiting)
                .order_by(Entry.received_at, Entry.update_id)
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            if row is None:
                return None
            interrupted = row.claim is not None
            token = uuid4()
            payload, attempts = row.payload, row.attempts + 1
            # SQLite ignores FOR UPDATE. A conditional write also fences two concurrent readers.
            claimed = await session.execute(
                update(Entry)
                .where(
                    Entry.bot_id == self.bot_id,
                    Entry.update_id == row.update_id,
                    Entry.ready_at <= now,
                    Entry.attempts == row.attempts,
                )
                .values(claim=token, ready_at=now + LEASE, payload=None, attempts=attempts)
                .execution_options(synchronize_session=False)
            )
            if not claimed.rowcount:
                return None
            return row.update_id, row.owner_id, payload, token, attempts, interrupted

    async def change(self, key, token, **values):
        async with self.sessions.begin() as session:
            result = await session.execute(
                update(Entry)
                .where(Entry.bot_id == self.bot_id, Entry.update_id == key, Entry.claim == token)
                .values(**values)
            )
            return result.rowcount == 1

    async def run(self, dispatcher, bot, kwargs):
        while True:
            self.wake.clear()
            try:
                job = await self.claim()
                if job:
                    await self.handle(job, dispatcher, bot, kwargs)
                    continue
            except DATABASE_ERRORS as exc:
                emit_event(
                    logger,
                    "serving.inbox_unavailable",
                    level=logging.WARNING,
                    error=exc,
                    result="deferred",
                )
            with suppress(TimeoutError):
                await asyncio.wait_for(self.wake.wait(), timeout=1)

    async def handle(self, job, dispatcher, bot, kwargs):
        key, owner, payload, token, attempts, interrupted = job

        async def renew():
            while True:
                await asyncio.sleep(LEASE.total_seconds() / 3)
                if not await self.change(key, token, ready_at=datetime.now(UTC) + LEASE):
                    return

        async def dispatch():
            if interrupted:
                # A previous handler may have committed an edit or sent a card.
                # Replaying the command could apply it to the next form field.
                await bot.send_message(
                    owner,
                    "Обработка команды прервалась. Проверьте результат "
                    "в чате и при необходимости повторите команду.",
                )
                return "uncertain"
            event = Update.model_validate(payload, context={"bot": bot})
            response = await dispatcher.feed_update(bot, event, **kwargs)
            if isinstance(response, TelegramMethod):
                await bot(response)
            return "failed" if response is COMMAND_FAILED else "success"

        started = time.monotonic()
        result = "uncertain"
        error = None
        worker = asyncio.create_task(dispatch())
        heartbeat = asyncio.create_task(renew())
        try:
            done, _ = await asyncio.wait((worker, heartbeat), return_when=asyncio.FIRST_COMPLETED)
            if heartbeat in done:
                # The owner erased their data, another worker owns the claim or storage is down.
                await heartbeat
                return
            try:
                outcome = await worker
            except (TelegramForbiddenError, TelegramBadRequest) as exc:
                outcome, error = "failed", exc
            except Exception as exc:
                error = exc
                if not interrupted or attempts < 4:
                    return
                outcome = "failed"
            if await self.change(key, token, ready_at=None, claim=None, owner_id=None):
                result = outcome
                self.wake.set()
        except asyncio.CancelledError:
            result = "cancelled"
            raise
        finally:
            worker.cancel()
            heartbeat.cancel()
            await asyncio.gather(worker, heartbeat, return_exceptions=True)
            emit_event(
                logger,
                "serving.command_finished",
                result=result,
                error=error,
                duration_ms=(time.monotonic() - started) * 1000,
            )
