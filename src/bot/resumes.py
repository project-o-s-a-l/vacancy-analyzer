import asyncio
import re

from aiogram import Bot, F, Router
from aiogram.dispatcher.event.bases import SkipHandler
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from bot.cards import split_message
from bot.common import registered, send_form
from bot.keyboards import main_keyboard
from bot.resume_cards import RESUME_PROMPT, render_resume, render_resume_details, resume_keyboard
from jobs.resumes import UnsavedProfileEdit
from jobs.users import UserService
from nlp.resume import looks_like_resume
from processing.resume import (
    MAX_RESUME_BYTES,
    ResumeBuffer,
    ResumeError,
    read_resume,
    resume_format,
)

EDIT_ACTIONS = {"c": "confirm", "e": "review", "m": "merge", "r": "replace_all"}


async def show_unsaved_edits(send, draft, action, conflict):
    target = "профиле" if conflict.target == "profile" else "предпросмотре резюме"
    code = next(key for key, value in EDIT_ACTIONS.items() if value == action)
    suffix = f"{draft.token}:{conflict.key}"
    buttons = []
    if draft.reviewed_profile is None or conflict.target == "resume":
        buttons.append(
            InlineKeyboardButton(
                text="Сохранить правки и сверить резюме", callback_data=f"resume:k:{suffix}"
            )
        )
    proceed = (
        "Отменить правки и применить резюме"
        if action == "confirm"
        else "Отменить правки и продолжить"
    )
    buttons.append(InlineKeyboardButton(text=proceed, callback_data=f"resume:{code}:{suffix}"))
    buttons.append(
        InlineKeyboardButton(text="Вернуться к моим правкам", callback_data="details:home")
    )
    await send(
        f"В {target} есть несохранённые изменения. Как с ними поступить?",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[[button] for button in buttons]),
        parse_mode=None,
    )


def register_resumes(router: Router, service: UserService) -> None:
    resume_slots = asyncio.Semaphore(2)

    async def show_resume(message: Message, user_id: int, draft, *, replace=False) -> None:
        if draft is None:
            await show_status(message, user_id, replace=replace)
            return
        user = await service.get(user_id)
        chunks = split_message(render_resume(draft, user.profile))
        send = message.edit_text if replace and len(chunks) == 1 else message.answer
        for index, chunk in enumerate(chunks):
            await send(
                chunk,
                parse_mode=None,
                reply_markup=resume_keyboard(draft, user.profile)
                if index == len(chunks) - 1
                else None,
            )

    async def show_status(message: Message, user_id: int, *, replace=False):
        send = message.edit_text if replace else message.answer
        status = await service.resumes.status(user_id)
        state = status["state"] if status else None
        if state not in {"pending", "running", "succeeded"}:
            user = await service.get(user_id)
            if user is None:
                await send("Начните с /start, чтобы создать профиль.", parse_mode=None)
                return
            back = (
                InlineKeyboardButton(text="← К анкете", callback_data="menu:setup")
                if user.form_state
                else InlineKeyboardButton(text="← В профиль", callback_data="menu:profile")
            )
            prefix = {
                "failed": "Не удалось обработать резюме. Попробуйте загрузить его ещё раз.",
                "superseded": "Профиль изменился. Загрузите актуальное резюме.",
                "cancelled": "Обработка отменена.",
            }.get(state)
            await send(
                f"{prefix}\n\n{RESUME_PROMPT}" if prefix else RESUME_PROMPT,
                parse_mode=None,
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[[back]]),
            )
            return
        labels = {
            "pending": "Резюме получено. Напишу сюда, когда оно будет готово.",
            "running": "Обрабатываю резюме. Напишу сюда, когда результат будет готов.",
            "succeeded": "Резюме обработано. Можно посмотреть результат.",
        }
        if status["state"] == "pending" and status["error_code"]:
            labels["pending"] = (
                "Резюме сохранено. Обработка продолжится автоматически. "
                "Когда результат будет готов, напишу сюда."
            )
        if state in {"pending", "running"}:
            buttons = [
                InlineKeyboardButton(
                    text="Отменить загрузку", callback_data="resume:jobcancel:" + status["id"]
                )
            ]
        else:
            buttons = [
                InlineKeyboardButton(
                    text="Посмотреть результат", callback_data="resume:ready:" + status["id"]
                )
            ]
        keyboard = InlineKeyboardMarkup(inline_keyboard=[buttons])
        if not (replace and message.text == labels[state] and message.reply_markup == keyboard):
            await send(labels[state], parse_mode=None, reply_markup=keyboard)

    @router.message(Command("resume"))
    async def resume(message: Message, user_id: int) -> None:
        if not await registered(service, message, user_id):
            return
        parts = (message.text or "").split(maxsplit=1)
        if len(parts) == 1:
            draft = await service.resumes.pending(user_id)
            if draft:
                await show_resume(message, user_id, draft)
            else:
                await show_status(message, user_id)
            return
        try:
            draft = await service.resumes.submit(user_id, parts[1])
        except ValueError as exc:
            await message.answer(str(exc), parse_mode=None)
            return
        await show_resume(message, user_id, draft)

    @router.message(F.text & ~F.text.startswith("/"))
    async def resume_text(message: Message, user_id: int) -> None:
        if not looks_like_resume(message.text):
            raise SkipHandler
        if not await registered(service, message, user_id):
            return
        try:
            draft = await service.resumes.submit(user_id, message.text)
        except ValueError as exc:
            await message.answer(str(exc)[:500], parse_mode=None)
            return
        await show_resume(message, user_id, draft)

    @router.message(Command("resume_delete"))
    async def delete_resume(message: Message, user_id: int) -> None:
        if await registered(service, message, user_id):
            await service.resumes.delete(user_id)
            await message.answer(
                "Тексты импортов и черновик удалены. Подтверждённые сведения и их цитаты "
                "сохранены в профиле и истории; "
                "их можно изменить через /profile.",
                parse_mode=None,
            )

    @router.message(Command("resume_details", "resume_edit"))
    async def review_resume(message: Message, user_id: int) -> None:
        if not await registered(service, message, user_id):
            return
        draft = await service.resumes.pending(user_id)
        if draft is None:
            await message.answer("Сначала загрузите резюме через /resume.", parse_mode=None)
            return
        parts = (message.text or "").split(maxsplit=2)
        if parts[0].split("@")[0] == "/resume_details":
            for chunk in split_message(render_resume_details(draft)):
                await message.answer(chunk, parse_mode=None)
            return
        if len(parts) != 3:
            await message.answer(
                "Пример: /resume_edit skills Python, SQL. "
                "Поля: skills, positions, experience, locations, salary, remote, mode, remove.",
                parse_mode=None,
            )
            return
        try:
            draft = await service.resumes.revise(user_id, draft.token, parts[1], parts[2])
        except UnsavedProfileEdit as exc:
            await show_unsaved_edits(message.answer, draft, "review", exc)
            return
        except ValueError as exc:
            await message.answer(str(exc), parse_mode=None)
            return
        await show_resume(message, user_id, draft)

    @router.message(F.document)
    async def resume_file(message: Message, bot: Bot, user_id: int) -> None:
        if not await registered(service, message, user_id):
            return
        document = message.document
        try:
            resume_format(document.file_name)
            if document.file_size is None or document.file_size > MAX_RESUME_BYTES:
                raise ResumeError(
                    "Резюме должно быть не больше 2 МБ; размер файла должен быть известен."
                )
            async with resume_slots:
                with ResumeBuffer() as buffer:
                    await bot.download(document.file_id, destination=buffer)
                    text, kind = await asyncio.to_thread(
                        read_resume, buffer.getvalue(), document.file_name
                    )
                draft = await service.resumes.submit(user_id, text, format=kind)
        except TelegramAPIError:
            await message.answer(
                "Не удалось скачать резюме из Telegram. Отправьте файл ещё раз.", parse_mode=None
            )
            return
        except ValueError as exc:
            await message.answer(str(exc), parse_mode=None)
            return
        await show_resume(message, user_id, draft)

    @router.callback_query(F.data.startswith("resume:"))
    async def resume_action(query: CallbackQuery, user_id: int) -> None:
        keep = re.fullmatch(r"resume:k:([0-9a-f]{32}):([0-9a-f]{16})", query.data)
        if keep:
            try:
                draft = await service.editor.save_before_resume(user_id, keep[1], keep[2])
            except ValueError as exc:
                await query.answer(str(exc)[:200], show_alert=True)
            else:
                await query.answer("Правки сохранены")
                await show_resume(query.message, user_id, draft, replace=True)
            return
        ready = re.fullmatch(r"resume:ready:([0-9a-f]{32})", query.data)
        if ready:
            draft = await service.resumes.pending(user_id, job_id=ready[1])
            if draft is None:
                await query.answer(
                    "Эта загрузка уже применена, отменена или заменена.", show_alert=True
                )
                return
            await query.answer()
            await show_resume(query.message, user_id, draft, replace=True)
            return
        if query.data == "resume:status":
            await query.answer("Проверяю результат")
            draft = await service.resumes.pending(user_id)
            await show_resume(query.message, user_id, draft, replace=True)
            return
        pending_cancel = re.fullmatch(r"resume:jobcancel:([0-9a-f]{32})", query.data)
        if pending_cancel:
            try:
                await service.resumes.cancel(user_id, processing_id=pending_cancel[1])
            except ValueError as exc:
                await query.answer(str(exc)[:200], show_alert=True)
            else:
                await query.answer("Обработка отменена")
                await show_status(query.message, user_id, replace=True)
            return
        if query.data == "resume:upload":
            if await service.get(user_id) is None:
                await query.answer("Сначала зарегистрируйтесь командой /start.", show_alert=True)
            else:
                await query.answer()
                await show_resume(
                    query.message, user_id, await service.resumes.pending(user_id), replace=True
                )
            return
        request = re.fullmatch(
            r"resume:(confirm|cancel|review|merge|replace_all):([0-9a-f]{32})", query.data
        )
        choice = re.fullmatch(r"resume:([cemr]):([0-9a-f]{32}):([0-9a-f]{16})", query.data)
        if not (request or choice):
            await query.answer("Некорректная кнопка", show_alert=True)
            return
        action, token, discard = (
            (EDIT_ACTIONS[choice[1]], choice[2], choice[3]) if choice else (*request.groups(), None)
        )
        try:
            if action == "review":
                from bot.profile_editor import show_editor

                edit = await service.editor.open(user_id, resume_token=token, discard=discard)
                await query.answer()
                await show_editor(query.message, edit, replace=True)
            elif action in {"merge", "replace_all"}:
                draft = await service.resumes.revise(
                    user_id, token, "mode", action, discard=discard
                )
                await query.answer()
                await show_resume(query.message, user_id, draft, replace=True)
            elif action == "confirm":
                user = await service.resumes.confirm(user_id, token, discard=discard)
                await query.answer("Резюме применено")
                await send_form(query.message, user, replace=True)
            else:
                await service.resumes.cancel(user_id, token)
                await query.answer("Загрузка отменена")
                user = await service.get(user_id)
                if user.form_state:
                    await send_form(query.message, user, replace=True)
                else:
                    await query.message.edit_text(
                        "Загрузка отменена.", reply_markup=main_keyboard(), parse_mode=None
                    )
        except UnsavedProfileEdit as exc:
            draft = await service.resumes.pending(user_id)
            if draft is None or draft.token != token:
                await query.answer(
                    "Эта загрузка уже изменилась. Откройте текущее резюме.", show_alert=True
                )
                return
            await query.answer()
            await show_unsaved_edits(query.message.edit_text, draft, action, exc)
        except ValueError as exc:
            await query.answer(str(exc)[:200], show_alert=True)
