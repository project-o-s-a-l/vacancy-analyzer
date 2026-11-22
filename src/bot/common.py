import logging
from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.enums import ChatType
from aiogram.exceptions import TelegramAPIError
from aiogram.types import CallbackQuery, Message

from bot.cards import render_profile
from bot.help import HELP
from bot.keyboards import form_keyboard, main_keyboard
from bot.profile_form import PROMPTS, salary_prompt
from jobs.profiles import SETUP_FIELDS
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


async def send_form(message: Message, user: RegisteredUser, *, replace: bool = False) -> None:
    send = message.edit_text if replace else message.answer
    if user.form_state:
        mode, field, *selection = user.form_state.split(":")
        text = PROMPTS[field]
        if field == "salary":
            prefs = user.profile.preferences
            text = salary_prompt(prefs, selection[0] if selection else prefs.currency)
        if mode == "setup" and field in SETUP_FIELDS:
            step = SETUP_FIELDS.index(field) + 1
            text = f"Шаг {step} из {len(SETUP_FIELDS)}\n{text}"
            if field == "skills":
                text = f"{HELP}\n\n{text}\n\nМожно также прикрепить резюме."
        await send(
            text,
            reply_markup=form_keyboard(field, user.revision),
            parse_mode=None,
        )
    else:
        await send(
            render_profile(user.profile, title="Профиль сохранён"),
            reply_markup=main_keyboard(),
            parse_mode=None,
        )


async def registered(service: UserService, message: Message, user_id: int) -> RegisteredUser | None:
    user = await service.get(user_id)
    if user is None:
        await message.answer("Начните с /start, чтобы создать свой профиль.", parse_mode=None)
    return user
