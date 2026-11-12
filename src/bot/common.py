import logging
from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.enums import ChatType
from aiogram.exceptions import TelegramAPIError
from aiogram.types import CallbackQuery, Message

from bot.cards import render_profile
from bot.keyboards import (
    form_keyboard,
    profile_keyboard,
)
from bot.profile_form import PROMPTS
from jobs.users import UserService
from observability import operation
from storage.database import DATABASE_ERRORS
from storage.repositories.user_repository import RegisteredUser

logger = logging.getLogger(__name__)


class PrivateChatsMiddleware(BaseMiddleware):
    async def __call__(
        self,
        handler: Callable[..., Awaitable[Any]],
        event: Message | CallbackQuery,
        data: dict[str, Any],
    ) -> Any:
        message = event if isinstance(event, Message) else event.message
        user = event.from_user
        if (
            not isinstance(message, Message)
            or message.chat.type != ChatType.PRIVATE
            or user is None
            or user.is_bot
            or message.chat.id != user.id
        ):
            return None
        # Identity comes exclusively from Telegram, never command text or callback data.
        data["user_id"] = user.id
        with operation("message"):
            return await self._handle(handler, event, data, message)

    async def _handle(self, handler, event, data, message):
        try:
            return await handler(event, data)
        except DATABASE_ERRORS:
            try:
                await message.answer(
                    "Сервис временно недоступен. Попробуйте позже.", parse_mode=None
                )
            except TelegramAPIError as exc:
                logger.warning("Telegram reply failed: %s", type(exc).__name__)
        except TelegramAPIError as exc:
            logger.warning("Telegram command failed: %s", type(exc).__name__)
        return None


async def send_form(message: Message, user: RegisteredUser) -> None:
    if user.form_state:
        field = user.form_state.split(":", 1)[1]
        await message.answer(
            PROMPTS[field] + "\n/cancel — закончить редактирование.",
            reply_markup=form_keyboard(field, user.revision),
            parse_mode=None,
        )
    else:
        await message.answer(
            "Профиль сохранён. Рекомендации доступны по /latest.\n\n"
            + render_profile(user.profile),
            reply_markup=profile_keyboard(),
            parse_mode=None,
        )


async def registered(service: UserService, message: Message, user_id: int) -> RegisteredUser | None:
    user = await service.get(user_id)
    if user is None:
        await message.answer("Начните с /start, чтобы создать свой профиль.", parse_mode=None)
    return user
