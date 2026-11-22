import asyncio
import logging
import re
from contextlib import suppress
from uuid import UUID

from aiogram import Bot, F, Router
from aiogram.enums import ChatAction
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    LinkPreviewOptions,
    Message,
    ReplyParameters,
)

from bot.cards import format_card, render_match_details, split_message
from bot.common import registered
from bot.delivery import TelegramSender
from bot.keyboards import (
    history_reset_keyboard,
    library_keyboard,
    main_keyboard,
    page_keyboard,
    profile_keyboard,
    reason_keyboard,
    vacancy_keyboard,
)
from domain.exchange_rates import use_rates
from jobs.exchange_rates import rates_for
from jobs.serving_compute import compute
from jobs.users import UserService
from scoring import score_vacancy
from scoring.skill_gap import render_skill_gap

logger = logging.getLogger(__name__)


def register_recommendations(router: Router, service: UserService) -> None:
    @router.message(Command("latest"))
    async def latest(message: Message, bot: Bot, user_id: int) -> None:
        try:
            notifications = await service.notifications(user_id, wait_for_rating=True)
        except ValueError as exc:
            await message.answer(str(exc), reply_markup=profile_keyboard(), parse_mode=None)
            return
        with suppress(TelegramAPIError):
            await bot.send_chat_action(message.chat.id, ChatAction.TYPING)
        report = await notifications.deliver(TelegramSender(bot))
        if report.sent:
            return
        if report.uncertain:
            text = "Доставка карточки не подтверждена. Проверьте чат. Повтор пока заблокирован."
        elif report.failed:
            text = "Telegram отклонил отправку карточки. Попробуйте /latest позже."
        else:
            text = report.empty_message or (
                "Новых рекомендаций с заданной оценкой пока нет. Уже показанные не повторяю."
            )
        if report.pending_message_id is not None:
            try:
                await message.answer(
                    text,
                    reply_parameters=ReplyParameters(
                        message_id=report.pending_message_id, allow_sending_without_reply=False
                    ),
                    parse_mode=None,
                )
                return
            except TelegramBadRequest:
                text = (
                    "Откройте /history и оцените последнюю полученную вакансию. "
                    "Затем нажмите /latest, чтобы получить следующую."
                )
        await message.answer(text, reply_markup=main_keyboard(), parse_mode=None)

    @router.callback_query(F.data == "menu:latest")
    async def find_vacancies(query: CallbackQuery, bot: Bot, user_id: int) -> None:
        await query.answer()
        await latest(query.message, bot, user_id)

    @router.message(Command("library"))
    async def library(message: Message, user_id: int) -> None:
        if await registered(service, message, user_id):
            await message.answer("Мои вакансии", reply_markup=library_keyboard(), parse_mode=None)

    @router.callback_query(F.data == "menu:library")
    async def open_library(query: CallbackQuery, user_id: int) -> None:
        await query.answer()
        await query.message.edit_text(
            "Мои вакансии", reply_markup=library_keyboard(), parse_mode=None
        )

    async def send_page(message: Message, user_id: int, kind: str, page: int = 0) -> None:
        try:
            profile, items = await service.feedback.page(user_id, kind, page)
        except ValueError as exc:
            await message.answer(str(exc), parse_mode=None)
            return
        title = {
            "saved": "Сохранённые вакансии",
            "applied": "Ваши отклики",
            "history": "Ранее полученные вакансии",
        }[kind]
        if not items:
            empty = {
                "saved": "Здесь появятся вакансии, в которых вы нажмёте «Сохранить».",
                "applied": "Чтобы отметить отклик, откройте «Ещё действия» в карточке вакансии.",
                "history": "Здесь появятся вакансии после первого подбора.",
            }
            await message.answer(
                f"{title}\n\n{empty[kind]}" if page == 0 else "На этой странице вакансий нет.",
                reply_markup=library_keyboard(),
                parse_mode=None,
            )
            return
        rates = await rates_for(profile)
        for index, (vacancy, state) in enumerate(items[:5]):
            if index:
                await asyncio.sleep(1.05)
            with use_rates(rates):
                match = await compute(score_vacancy, vacancy, profile)
            sent = await message.answer(
                **format_card(vacancy, match).as_kwargs(),
                reply_markup=vacancy_keyboard(vacancy, state),
                link_preview_options=LinkPreviewOptions(is_disabled=True),
            )
            await service.feedback.confirm_display(
                user_id, vacancy, profile, match, sent.message_id
            )
        navigation = page_keyboard(kind, page, len(items) > 5)
        await asyncio.sleep(1.05)
        await message.answer(
            f"{title} · страница {page + 1}", reply_markup=navigation, parse_mode=None
        )

    @router.message(Command("saved", "applied", "history"))
    async def marked(message: Message, user_id: int) -> None:
        kind = (message.text or "").split()[0].split("@")[0][1:]
        await send_page(message, user_id, kind)

    @router.callback_query(F.data.startswith("fbpage:"))
    async def feedback_page(query: CallbackQuery, user_id: int) -> None:
        match = re.fullmatch(r"fbpage:(saved|applied|history):(\d{1,6})", query.data)
        if not match or int(match[2]) > 100_000:
            await query.answer("Некорректная страница", show_alert=True)
            return
        await query.answer()
        await send_page(query.message, user_id, match[1], int(match[2]))

    async def request_history_reset(message: Message, user_id: int) -> None:
        try:
            request = await service.feedback.request_history_reset(user_id)
        except ValueError as exc:
            await message.answer(str(exc), parse_mode=None)
            return
        if request is None:
            await message.answer(
                "История вакансий пока пуста. Подбор доступен по /latest.", parse_mode=None
            )
            return
        await message.answer(
            f"Сбросить историю полученных вакансий ({request.count})?\n\n"
            "Они снова смогут попасть в подборку /latest, если подходят вашему профилю.\n"
            "Сохранённые вакансии, оценки и отметки откликов останутся. "
            "Вакансии с оценкой «Не подходит» и уже отмеченные отклики повторно не предлагаю.",
            reply_markup=history_reset_keyboard(request.token),
            parse_mode=None,
        )

    @router.message(Command("history_clear"))
    async def clear_history(message: Message, user_id: int) -> None:
        await request_history_reset(message, user_id)

    @router.callback_query(F.data == "history:reset")
    async def ask_history_reset(query: CallbackQuery, user_id: int) -> None:
        await query.answer()
        await request_history_reset(query.message, user_id)

    @router.callback_query(F.data.startswith("history:"))
    async def finish_history_reset(query: CallbackQuery, user_id: int) -> None:
        match = re.fullmatch(r"history:(confirm|cancel):([0-9a-f]{32})", query.data)
        if not match:
            await query.answer("Некорректное действие", show_alert=True)
            return
        try:
            if match[1] == "cancel":
                await service.feedback.cancel_history_reset(user_id, match[2])
                text = "Сброс истории отменён."
            else:
                count = await service.feedback.confirm_history_reset(user_id, match[2])
                text = (
                    f"История сброшена: {count}.\n"
                    "Вакансии снова доступны для подбора по /latest. "
                    "Сохранённые вакансии, оценки и отклики сохранены."
                )
        except ValueError as exc:
            await query.answer(str(exc), show_alert=True)
            return
        await query.answer("Отменено" if match[1] == "cancel" else "История сброшена")
        await query.message.edit_text(text, reply_markup=None, parse_mode=None)

    @router.callback_query(F.data.startswith("card:"))
    async def card_action(query: CallbackQuery, user_id: int) -> None:
        action = re.fullmatch(r"card:(more|back|match):([0-9a-f]{32})", query.data)
        if not action:
            await query.answer("Некорректная кнопка", show_alert=True)
            return
        vacancy_id = UUID(hex=action[2])
        try:
            if action[1] == "match":
                match = await service.feedback.shown_match(
                    user_id, vacancy_id, query.message.message_id
                )
                await query.answer()
                documents = any(
                    doc.importance in {"required", "preferred"} for doc in match.qualifications
                )
                keyboard = (
                    InlineKeyboardMarkup(
                        inline_keyboard=[
                            [
                                InlineKeyboardButton(
                                    text="Указать документы в профиле",
                                    callback_data="details:qualifications",
                                )
                            ]
                        ]
                    )
                    if documents
                    else None
                )
                await query.message.reply(
                    render_match_details(match), parse_mode=None, reply_markup=keyboard
                )
            else:
                vacancy, state = await service.feedback.card(user_id, vacancy_id)
                await query.answer()
                await query.message.edit_reply_markup(
                    reply_markup=vacancy_keyboard(vacancy, state, expanded=action[1] == "more")
                )
        except ValueError as exc:
            await query.answer(str(exc)[:200], show_alert=True)

    async def update_feedback_keyboard(
        query: CallbackQuery, user_id: int, state, *, expanded=False
    ) -> None:
        # State is already committed; a Telegram edit failure must not undo the user's action.
        vacancy, current = await service.feedback.card(user_id, state.vacancy_id)
        await query.message.edit_reply_markup(
            reply_markup=vacancy_keyboard(vacancy, current, expanded=expanded)
        )

    @router.callback_query(F.data.startswith("fb:"))
    async def feedback_action(query: CallbackQuery, bot: Bot, user_id: int) -> None:
        match = re.fullmatch(
            r"fb:(like|dislike|save|unsave|apply|unapply):([0-9a-f]{32})", query.data
        )
        if not match:
            await query.answer("Некорректное действие", show_alert=True)
            return
        try:
            state = await service.feedback.record(
                user_id,
                UUID(hex=match[2]),
                match[1],
                event_key=query.id,
                exposure_message_id=query.message.message_id,
            )
        except ValueError as exc:
            await query.answer(str(exc), show_alert=True)
            return
        try:
            if match[1] == "dislike" and state.interest == "dislike":
                await query.answer("Отмечено. Можно выбрать причину.")
                await query.message.edit_reply_markup(reply_markup=reason_keyboard(state))
            else:
                labels = {
                    "like": "Отмечено как интересное",
                    "save": "Вакансия сохранена",
                    "unsave": "Убрано из сохранённых",
                    "apply": "Отклик отмечен",
                    "unapply": "Отметка отклика убрана",
                }
                await query.answer(labels.get(match[1], "Оценка сохранена"))
                await update_feedback_keyboard(
                    query, user_id, state, expanded=match[1] in {"apply", "unapply"}
                )
        except TelegramAPIError as exc:
            # The rating is committed even if Telegram cannot update the old keyboard.
            logger.warning("Telegram feedback controls failed: %s", type(exc).__name__)
        if state.advance_feed:
            await latest(query.message, bot, user_id)

    @router.callback_query(F.data.startswith("fbr:"))
    async def feedback_reason(query: CallbackQuery, user_id: int) -> None:
        match = re.fullmatch(r"fbr:([0-9a-f]{32}):(\d{1,10}):([a-z]+)", query.data)
        if not match:
            await query.answer("Некорректная причина", show_alert=True)
            return
        try:
            state = await service.feedback.record(
                user_id,
                UUID(hex=match[1]),
                "reason",
                reason=None if match[3] == "skip" else match[3],
                expected_version=int(match[2]),
                exposure_message_id=query.message.message_id,
                event_key=query.id,
            )
        except ValueError as exc:
            await query.answer(str(exc), show_alert=True)
            return
        await query.answer("Оценка сохранена" if match[3] == "skip" else "Причина сохранена")
        await update_feedback_keyboard(query, user_id, state)

    @router.message(Command("skillgap"))
    async def analysis(message: Message, user_id: int) -> None:
        try:
            report = await service.skill_gap(user_id)
        except ValueError as exc:
            await message.answer(str(exc), parse_mode=None)
            return
        text = render_skill_gap(report)
        # Imported source labels and profile directions may be long. Respect Telegram's limit.
        for index, chunk in enumerate(split_message(text)):
            if index:
                await asyncio.sleep(1.05)
            await message.answer(chunk, parse_mode=None)

    @router.message(Command("status"))
    async def status(message: Message, user_id: int) -> None:
        if not await registered(service, message, user_id):
            return
        notifications = await service.notifications(user_id, require_profile=False)
        summary = await notifications.summary()
        counts = summary["deliveries"]
        text = "\n".join(
            [
                f"Вакансий за 30 дней: {summary['recent']}",
                f"Отправлено вам карточек: {counts.get('sent', 0)}",
                f"Ваших ошибок отправки: {counts.get('failed', 0)}",
                "Ваших отправок без подтверждения/в процессе: "
                + str(counts.get("sending", 0) + counts.get("uncertain", 0)),
                f"Минимальная оценка: {service.min_score:g}%. Вакансии выдаются по одной.",
                f"Сохранено: {summary['feedback']['saved']}; "
                f"откликов: {summary['feedback']['applied']}.",
            ]
        )
        await message.answer(text, parse_mode=None)
