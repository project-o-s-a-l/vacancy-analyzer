from aiogram import F, Router
from aiogram.dispatcher.event.bases import SkipHandler
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, Message

from bot.cards import render_profile
from bot.common import registered, send_form
from bot.help import HELP
from bot.keyboards import (
    profile_keyboard,
)
from jobs.users import UserService
from nlp.resume import looks_like_resume


def register_profiles(router: Router, service: UserService) -> None:
    @router.message(CommandStart())
    async def start(message: Message, user_id: int) -> None:
        user = await service.register(user_id)
        await message.answer(HELP, parse_mode=None)
        if user.form_state:
            await send_form(message, user)
        else:
            await message.answer(
                render_profile(user.profile), reply_markup=profile_keyboard(), parse_mode=None
            )

    @router.message(Command("profile"))
    async def profile(message: Message, user_id: int) -> None:
        user = await registered(service, message, user_id)
        if user:
            await message.answer(
                render_profile(user.profile), reply_markup=profile_keyboard(), parse_mode=None
            )

    @router.message(Command("cancel"))
    async def cancel(message: Message, user_id: int) -> None:
        if await registered(service, message, user_id):
            await service.resumes.cancel(user_id)
            user = await service.cancel(user_id)
            await message.answer(
                "Редактирование завершено. Уже введённые данные сохранены.", parse_mode=None
            )
            await message.answer(
                render_profile(user.profile), reply_markup=profile_keyboard(), parse_mode=None
            )

    @router.callback_query(F.data.startswith("profile:"))
    async def edit(query: CallbackQuery, user_id: int) -> None:
        try:
            user = await service.begin_edit(user_id, query.data.split(":", 1)[1])
        except ValueError as exc:
            await query.answer(str(exc), show_alert=True)
            return
        await query.answer()
        await send_form(query.message, user)

    @router.callback_query(F.data.startswith("answer:"))
    async def answer_button(query: CallbackQuery, user_id: int) -> None:
        parts = query.data.split(":")
        if len(parts) != 3:
            await query.answer("Некорректная кнопка", show_alert=True)
            return
        _, field, value = parts
        if field not in {"remote", "constraints"} and value != "skip":
            await query.answer("Некорректная кнопка", show_alert=True)
            return
        try:
            user = await service.accept_answer(
                user_id,
                "пропустить" if value == "skip" else value,
                expected_field=field,
            )
        except ValueError as exc:
            await query.answer(str(exc)[:200], show_alert=True)
            return
        await query.answer()
        await send_form(query.message, user)

    @router.message(F.text & ~F.text.startswith("/"))
    async def form_answer(message: Message, user_id: int) -> None:
        user = await registered(service, message, user_id)
        if user is None:
            return
        if (user.form_state in {None, "setup:skills", "edit:skills"}) and looks_like_resume(
            message.text
        ):
            raise SkipHandler
        if not user.form_state:
            await message.answer(
                "Откройте /profile и выберите поле или запросите /latest.", parse_mode=None
            )
            return
        try:
            user = await service.accept_answer(user_id, message.text)
        except ValueError as exc:
            await message.answer(str(exc)[:500], parse_mode=None)
            return
        await send_form(message, user)
