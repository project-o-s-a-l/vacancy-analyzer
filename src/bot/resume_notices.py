"""Deliver completed imports from the bot even when analysis runs in a separate worker."""

import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime, timedelta

from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError, TelegramRetryAfter
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from sqlalchemy import select, update

from storage.models import BackgroundJob, ResumeModel

logger = logging.getLogger(__name__)
LEASE = timedelta(minutes=2)


async def deliver_next_notice(bot, sessions):
    now = datetime.now(UTC)
    lease_until = now + LEASE
    async with sessions.begin() as session:
        job = (
            await session.execute(
                select(ResumeModel.user_id, BackgroundJob.id, BackgroundJob.state)
                .join(BackgroundJob, ResumeModel.processing_job_id == BackgroundJob.id)
                .where(
                    ResumeModel.notice_due_at <= now,
                    ResumeModel.user_id == BackgroundJob.user_id,
                    BackgroundJob.kind == "resume_analysis",
                    BackgroundJob.state.in_(("succeeded", "failed", "superseded")),
                )
                .order_by(ResumeModel.notice_due_at, ResumeModel.user_id)
                .limit(1)
            )
        ).first()
        if job is None:
            return False
        current = update(ResumeModel).where(
            ResumeModel.user_id == job.user_id, ResumeModel.processing_job_id == job.id
        )
        claimed = await session.execute(
            current.where(ResumeModel.notice_due_at <= now).values(notice_due_at=lease_until)
        )
        if not claimed.rowcount:
            return False

    due = None
    if job.state == "succeeded":
        text = "Резюме обработано. Проверьте распознанные сведения и примените их к профилю."
        button = InlineKeyboardButton(
            text="Посмотреть результат", callback_data="resume:ready:" + job.id.hex
        )
    else:
        text = (
            "Профиль изменился во время обработки. Пришлите резюме ещё раз."
            if job.state == "superseded"
            else "Не удалось обработать резюме. Попробуйте загрузить его ещё раз."
        )
        button = InlineKeyboardButton(text="Загрузить резюме", callback_data="resume:upload")
    try:
        await bot.send_message(
            job.user_id,
            text,
            parse_mode=None,
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[[button]]),
        )
    except TelegramRetryAfter as exc:
        due = datetime.now(UTC) + timedelta(seconds=max(1, exc.retry_after))
    except (TelegramForbiddenError, TelegramBadRequest) as exc:
        logger.info("Resume notice unavailable: %s", type(exc).__name__)
    # If delivery was interrupted or its outcome is unknown, the lease permits a retry.
    async with sessions.begin() as session:
        await session.execute(
            current.where(ResumeModel.notice_due_at == lease_until).values(notice_due_at=due)
        )
    return True


@asynccontextmanager
async def background_resume_notices(bot, sessions):
    async def run():
        while True:
            try:
                if await deliver_next_notice(bot, sessions):
                    continue
            except Exception as exc:
                logger.warning("Resume notice deferred: %s", type(exc).__name__)
            await asyncio.sleep(3)

    task = asyncio.create_task(run())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
