import asyncio
import re

from aiogram import Bot, F, Router
from aiogram.dispatcher.event.bases import SkipHandler
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from bot.cards import split_message
from bot.common import registered, send_form
from bot.resume_cards import RESUME_PROMPT, render_resume, resume_keyboard
from jobs.users import UserService
from nlp.resume import looks_like_resume
from processing.resume import (
    MAX_RESUME_BYTES,
    ResumeBuffer,
    ResumeError,
    read_resume,
    resume_format,
)


def register_resumes(router: Router, service: UserService) -> None:
    resume_slots = asyncio.Semaphore(2)

    async def show_resume(message: Message, user_id: int, draft) -> None:
        if draft is None:
            await show_status(message, user_id)
            return
        user = await service.get(user_id)
        chunks = split_message(render_resume(draft, user.profile))
        for index, chunk in enumerate(chunks):
            await message.answer(
                chunk,
                parse_mode=None,
                reply_markup=resume_keyboard(draft) if index == len(chunks) - 1 else None,
            )

    async def show_status(message: Message, user_id: int):
        status = await service.resumes.status(user_id)
        if not status:
            await message.answer(RESUME_PROMPT, parse_mode=None)
            return
        labels = {
            "pending": "Резюме сохранено и ожидает обработки.",
            "running": "Резюме обрабатывается. Остальные команды доступны.",
            "succeeded": "Обработка завершена. Откройте /resume для проверки результата.",
            "failed": "Не удалось обработать резюме. Загрузите его ещё раз или заполните /profile.",
            "superseded": "Профиль или обработчик изменился. Загрузите резюме ещё раз.",
            "cancelled": "Обработка отменена. Можно загрузить новое резюме.",
        }
        if status["state"] == "pending" and status["error_code"]:
            labels["pending"] = (
                "Обработка временно недоступна; повтор запланирован. "
                "Резюме сохранено, остальные команды доступны."
            )
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(text="Проверить результат", callback_data="resume:status"),
                    InlineKeyboardButton(
                        text="Отменить", callback_data="resume:jobcancel:" + status["id"]
                    ),
                ]
            ]
        )
        await message.answer(labels[status["state"]], parse_mode=None, reply_markup=keyboard)

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
                "Текст резюме и черновик удалены. Поля профиля сохранены; "
                "их можно изменить через /profile.",
                parse_mode=None,
            )

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
        if query.data == "resume:status":
            await query.answer()
            draft = await service.resumes.pending(user_id)
            await show_resume(query.message, user_id, draft)
            return
        pending_cancel = re.fullmatch(r"resume:jobcancel:([0-9a-f]{32})", query.data)
        if pending_cancel:
            try:
                await service.resumes.cancel(user_id, processing_id=pending_cancel[1])
            except ValueError as exc:
                await query.answer(str(exc)[:200], show_alert=True)
            else:
                await query.answer("Обработка отменена, профиль сохранён.")
            return
        if query.data == "resume:upload":
            if await service.get(user_id) is None:
                await query.answer("Сначала зарегистрируйтесь командой /start.", show_alert=True)
            else:
                await query.answer()
                await query.message.answer(RESUME_PROMPT, parse_mode=None)
            return
        match = re.fullmatch(r"resume:(confirm|cancel):([0-9a-f]{32})", query.data)
        if not match:
            await query.answer("Некорректная кнопка", show_alert=True)
            return
        try:
            if match[1] == "confirm":
                user = await service.resumes.confirm(user_id, match[2])
                await query.answer("Профиль обновлён. Проверьте распознанные сведения.")
                await send_form(query.message, user)
            else:
                await service.resumes.cancel(user_id, match[2])
                await query.answer("Загрузка отменена, профиль сохранён.")
        except ValueError as exc:
            await query.answer(str(exc)[:200], show_alert=True)
