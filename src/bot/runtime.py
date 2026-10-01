import asyncio
import logging
import time

from aiogram import Bot
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.types import BotCommand

from bot.handlers import build_dispatcher
from domain.serving_ranker import RankerPolicy
from jobs.analysis_worker import background_analysis
from jobs.collection_runtime import background_collection
from jobs.settings import Settings
from jobs.users import UserService
from observability import emit_event, operation
from storage.database import create_engine, session_factory

logger = logging.getLogger(__name__)


def create_bot(settings: Settings) -> Bot:
    return Bot(token=settings.require_telegram_token(), session=AiohttpSession(timeout=30))


async def run_bot(settings: Settings) -> None:
    with operation("bot"):
        started = time.monotonic()
        emit_event(logger, "component.started", component_kind="bot")
        try:
            await _run_bot(settings)
        except asyncio.CancelledError:
            emit_event(
                logger,
                "component.finished",
                component_kind="bot",
                result="cancelled",
                duration_ms=(time.monotonic() - started) * 1000,
            )
            raise
        except Exception as exc:
            emit_event(
                logger,
                "component.failed",
                component_kind="bot",
                result="failed",
                level=logging.ERROR,
                error=exc,
                duration_ms=(time.monotonic() - started) * 1000,
            )
            raise
        else:
            emit_event(
                logger,
                "component.finished",
                component_kind="bot",
                result="success",
                duration_ms=(time.monotonic() - started) * 1000,
            )


async def _run_bot(settings: Settings) -> None:
    bot = create_bot(settings)
    engine = None
    try:
        engine = create_engine(settings.database_url)
        service = UserService(
            session_factory(engine),
            top=settings.telegram_top,
            min_score=settings.telegram_min_score,
            queued_resumes=True,
            ranker_policy=RankerPolicy.from_settings(settings),
            saved_selections_enabled=getattr(settings, "saved_selections_enabled", True),
            resume_model=settings.ollama_model if settings.resume_llm_enabled else "rules",
        )
        # Check schema/connectivity before entering long polling.
        await service.check_ready()
        webhook = await bot.get_webhook_info()
        if webhook.url:
            raise ValueError("Для бота уже установлен webhook. Отключите его перед long polling.")
        await bot.set_my_commands(
            [
                BotCommand(command="resume", description="Загрузить резюме"),
                BotCommand(command="latest", description="Новые рекомендации"),
                BotCommand(command="saved", description="Сохранённые вакансии"),
                BotCommand(command="applied", description="Ваши отклики"),
                BotCommand(command="profile", description="Текущий профиль"),
                BotCommand(command="skillgap", description="Что можно дополнить в профиле"),
            ]
        )
        async with background_collection(settings, engine), background_analysis(settings, engine):
            emit_event(logger, "component.ready", component_kind="bot", result="ready")
            await build_dispatcher(service).start_polling(
                bot,
                allowed_updates=["message", "callback_query"],
                handle_as_tasks=False,
                close_bot_session=False,
            )
    finally:
        try:
            await bot.session.close()
        finally:
            if engine is not None:
                await engine.dispose()
