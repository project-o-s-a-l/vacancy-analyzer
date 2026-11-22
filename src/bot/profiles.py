from aiogram import F, Router
from aiogram.dispatcher.event.bases import SkipHandler
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, Message

from bot.cards import render_profile
from bot.common import registered, send_form
from bot.help import HELP
from bot.keyboards import main_keyboard, profile_keyboard
from jobs.profiles import SALARY_CURRENCIES
from jobs.users import UserService
from nlp.resume import looks_like_resume


def register_profiles(router: Router, service: UserService) -> None:
    async def show_profile(message: Message, user, *, replace: bool = False) -> None:
        draft = await service.editor.get(user.user_id)
        text = render_profile(user.profile)
        editing = bool(draft and draft.dirty)
        if editing:
            text += "\n\nУ вас есть несохранённые изменения."
        send = message.edit_text if replace else message.answer
        await send(text, reply_markup=profile_keyboard(editing=editing), parse_mode=None)

    @router.message(CommandStart())
    async def start(message: Message, user_id: int) -> None:
        user = await service.register(user_id)
        if user.form_state and not await service.editor.get(user_id):
            await send_form(message, user)
        else:
            await message.answer(HELP, reply_markup=main_keyboard(), parse_mode=None)

    @router.message(Command("menu", "help"))
    async def menu(message: Message, user_id: int) -> None:
        if await registered(service, message, user_id):
            await message.answer(HELP, reply_markup=main_keyboard(), parse_mode=None)

    @router.callback_query(F.data.in_({"menu:home", "menu:profile", "menu:setup"}))
    async def navigate(query: CallbackQuery, user_id: int) -> None:
        user = await service.get(user_id)
        if user is None:
            await query.answer("Начните с /start.", show_alert=True)
            return
        await query.answer()
        if query.data == "menu:setup" and user.form_state:
            await send_form(query.message, user, replace=True)
        elif query.data == "menu:profile":
            await show_profile(query.message, user, replace=True)
        else:
            await query.message.edit_text(HELP, reply_markup=main_keyboard(), parse_mode=None)

    @router.message(Command("profile"))
    async def profile(message: Message, user_id: int) -> None:
        user = await registered(service, message, user_id)
        if user:
            await show_profile(message, user)

    @router.message(Command("cancel"))
    async def cancel(message: Message, user_id: int) -> None:
        if user := await registered(service, message, user_id):
            cancelled_edit = await service.editor.cancel(user_id)
            if not cancelled_edit:
                await service.resumes.cancel(user_id)
                user = await service.cancel(user_id)
            title = "Правки отменены" if cancelled_edit else "Редактирование завершено"
            await message.answer(
                render_profile(user.profile, title=title),
                reply_markup=profile_keyboard(),
                parse_mode=None,
            )

    @router.callback_query(F.data.startswith("profile:"))
    async def edit(query: CallbackQuery, user_id: int) -> None:
        await query.answer("Кнопка старой анкеты. Откройте /profile.", show_alert=True)

    @router.callback_query(F.data.startswith("answer:"))
    async def answer_button(query: CallbackQuery, user_id: int) -> None:
        parts = query.data.split(":")
        if len(parts) != 4 or not parts[1].isdecimal():
            await query.answer("Некорректная кнопка", show_alert=True)
            return
        _, revision, field, value = parts
        currency_choice = field == "salary" and value in SALARY_CURRENCIES
        if field not in {"remote", "constraints"} and value != "skip" and not currency_choice:
            await query.answer("Некорректная кнопка", show_alert=True)
            return
        try:
            user = await service.accept_answer(
                user_id,
                "пропустить" if value == "skip" else value,
                expected_field=field,
                expected_revision=int(revision),
            )
        except ValueError as exc:
            await query.answer(str(exc)[:200], show_alert=True)
            return
        await query.answer()
        await send_form(query.message, user, replace=True)

    @router.message(F.text & ~F.text.startswith("/"))
    async def form_answer(message: Message, user_id: int) -> None:
        user = await registered(service, message, user_id)
        if user is None:
            return
        if looks_like_resume(message.text):
            raise SkipHandler
        if not user.form_state:
            await message.answer(HELP, reply_markup=main_keyboard(), parse_mode=None)
            return
        try:
            user = await service.accept_answer(user_id, message.text)
        except ValueError as exc:
            await message.answer(str(exc)[:500], parse_mode=None)
            return
        await send_form(message, user)
