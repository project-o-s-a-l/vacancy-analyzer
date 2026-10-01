import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from aiogram import Bot
from aiogram.methods import AnswerCallbackQuery, EditMessageText, SendMessage
from sqlalchemy import func, select
from test_bot import TEST_TOKEN, RecordingSession, command_update
from test_feedback import seed, update

from bot.handlers import build_dispatcher
from bot.keyboards import history_reset_keyboard, page_keyboard
from domain.delivery import DeliveryFailure
from jobs.feedback import FeedbackService
from jobs.notify import NotificationService
from jobs.users import UserService
from storage.models import FeedbackEventModel, NotificationModel, TelegramUserModel, VacancyModel
from storage.repositories import VacancyRepository
from storage.repositories.notification_repository import NotificationRepository


async def test_reset_requires_confirmation_and_releases_only_own_history(
    sessions, vacancy, profile
):
    await seed(sessions, vacancy, profile, users=(42, 99))
    feedback = FeedbackService(sessions)
    draft = await feedback.request_history_reset(42)
    assert draft.count == 1
    assert len((await feedback.page(42, "history"))[1]) == 1
    service = NotificationService(sessions, profile, chat_id=42, min_score=0)
    assert await service.pending() == []
    # Confirmation remains valid after process/service recreation.
    assert await FeedbackService(sessions).confirm_history_reset(42, draft.token) == 1
    assert (await feedback.page(42, "history"))[1] == []
    assert len((await feedback.page(99, "history"))[1]) == 1
    assert [item.id for item, _ in await service.pending()] == [vacancy.id]
    async with sessions() as session:
        assert await session.scalar(select(func.count()).select_from(NotificationModel)) == 2
        assert await session.scalar(select(func.count()).select_from(VacancyModel)) == 1
    with pytest.raises(ValueError, match="уже выполнен"):
        await feedback.confirm_history_reset(42, draft.token)


@pytest.mark.parametrize("action", ["save", "apply", "dislike", "like"])
async def test_reset_preserves_feedback_and_suppression(sessions, vacancy, profile, action):
    await seed(sessions, vacancy, profile)
    feedback = FeedbackService(sessions)
    original = await feedback.record(42, vacancy.id, action, event_key="before")
    draft = await feedback.request_history_reset(42)
    await feedback.confirm_history_reset(42, draft.token)
    assert (await feedback.card(42, vacancy.id))[1] == original
    async with sessions() as session:
        assert await session.scalar(select(func.count()).select_from(FeedbackEventModel)) == 1
    pending = await NotificationService(sessions, profile, chat_id=42, min_score=0).pending()
    assert bool(pending) == (action not in {"apply", "dislike"})
    if action in {"save", "apply"}:
        assert len((await feedback.page(42, "saved" if action == "save" else "applied"))[1]) == 1


async def test_redelivery_blocks_duplicates_and_reappears_in_history(sessions, vacancy, profile):
    await seed(sessions, vacancy, profile)
    feedback = FeedbackService(sessions)
    draft = await feedback.request_history_reset(42)
    await feedback.confirm_history_reset(42, draft.token)
    transport = SimpleNamespace(send_recommendation=AsyncMock(return_value=55))
    service = NotificationService(sessions, profile, chat_id=42, min_score=0)
    assert (await service.deliver(transport)).sent == 1
    assert (await service.deliver(transport)).sent == 0
    assert len((await feedback.page(42, "history"))[1]) == 1
    async with sessions() as session:
        row = await session.get(NotificationModel, (42, vacancy.id))
        assert row.attempts == 2 and row.message_id == 55 and row.history_reset_at is None
    with pytest.raises(ValueError):
        await feedback.confirm_history_reset(42, draft.token)
    assert await service.pending() == []


@pytest.mark.parametrize("uncertain", [False, True])
async def test_redelivery_failure_keeps_old_card_access_and_saves(
    sessions, vacancy, profile, uncertain
):
    await seed(sessions, vacancy, profile)
    feedback = FeedbackService(sessions)
    draft = await feedback.request_history_reset(42)
    await feedback.confirm_history_reset(42, draft.token)
    transport = SimpleNamespace(
        send_recommendation=AsyncMock(side_effect=DeliveryFailure(uncertain=uncertain, kind="test"))
    )
    service = NotificationService(sessions, profile, chat_id=42, min_score=0)
    report = await service.deliver(transport)
    assert (report.uncertain, report.failed) == ((1, 0) if uncertain else (0, 1))
    state = await feedback.record(42, vacancy.id, "save", event_key="after-failure")
    assert state.saved and len((await feedback.page(42, "saved"))[1]) == 1
    assert bool(await service.pending()) == (not uncertain)
    assert (await feedback.page(42, "history"))[1] == []


async def test_uncertain_and_in_flight_deliveries_are_not_released(sessions, vacancy, profile):
    await seed(sessions, vacancy, profile)
    unknown = vacancy.model_copy(
        update={
            "id": uuid4(),
            "source_id": "uncertain",
            "title": "Другая вакансия с неясной доставкой",
        }
    )
    in_flight = vacancy.model_copy(
        update={"id": uuid4(), "source_id": "sending", "title": "Другая вакансия в отправке"}
    )
    async with sessions.begin() as session:
        repository = NotificationRepository(session)
        for item in (unknown, in_flight):
            await VacancyRepository(session).save(item)
            token = await repository.claim(42, item.id)
            if item is unknown:
                await repository.finish(42, item.id, token, "uncertain")
    feedback = FeedbackService(sessions)
    draft = await feedback.request_history_reset(42)
    assert draft.count == 1
    await feedback.confirm_history_reset(42, draft.token)
    async with sessions.begin() as session:
        repository = NotificationRepository(session)
        blocked = await repository.blocked_ids(42)
        assert blocked == {unknown.id, in_flight.id}
        assert await repository.claim(42, unknown.id) is None
        assert await repository.claim(42, in_flight.id) is None


async def test_later_deliveries_survive_an_older_confirmation(sessions, vacancy, profile):
    await seed(sessions, vacancy, profile)
    feedback = FeedbackService(sessions)
    draft = await feedback.request_history_reset(42)
    newer = vacancy.model_copy(
        update={"id": uuid4(), "source_id": "newer", "title": "Другая новая вакансия"}
    )
    await seed(sessions, newer, profile)
    assert await feedback.confirm_history_reset(42, draft.token) == 1
    assert [item.id for item, _ in (await feedback.page(42, "history"))[1]] == [newer.id]


async def test_stale_foreign_and_cancelled_confirmations_do_nothing(sessions, vacancy, profile):
    await seed(sessions, vacancy, profile, users=(42, 99))
    feedback = FeedbackService(sessions)
    first = await feedback.request_history_reset(42)
    second = await feedback.request_history_reset(42)
    for user_id, token in ((42, first.token), (99, second.token), (42, "bad")):
        with pytest.raises(ValueError):
            await feedback.confirm_history_reset(user_id, token)
    await feedback.cancel_history_reset(42, second.token)
    with pytest.raises(ValueError):
        await feedback.confirm_history_reset(42, second.token)
    assert len((await feedback.page(42, "history"))[1]) == 1
    assert len((await feedback.page(99, "history"))[1]) == 1


async def test_global_cancel_invalidates_pending_reset(sessions, vacancy, profile):
    await seed(sessions, vacancy, profile)
    feedback = FeedbackService(sessions)
    draft = await feedback.request_history_reset(42)
    await UserService(sessions).cancel(42)
    with pytest.raises(ValueError):
        await feedback.confirm_history_reset(42, draft.token)
    assert len((await feedback.page(42, "history"))[1]) == 1


async def test_empty_and_unregistered_history_cannot_be_reset(sessions):
    feedback = FeedbackService(sessions)
    for action in (
        feedback.request_history_reset(42),
        feedback.confirm_history_reset(42, "0" * 32),
        feedback.cancel_history_reset(42, "0" * 32),
    ):
        with pytest.raises(ValueError, match="/start"):
            await action
    await UserService(sessions).register(42)
    assert await feedback.request_history_reset(42) is None


def test_history_controls_have_bounded_callbacks_and_do_not_expand_other_lists():
    keyboard = page_keyboard("history", 0, False)
    assert keyboard.inline_keyboard[0][0].callback_data == "history:reset"
    assert page_keyboard("saved", 0, False) is None
    for row in history_reset_keyboard("0" * 32).inline_keyboard:
        for button in row:
            assert len(button.callback_data.encode()) <= 64


async def test_dispatcher_confirmation_and_cancel_flow(sessions, vacancy, profile, monkeypatch):
    monkeypatch.setattr("bot.recommendations.asyncio.sleep", AsyncMock())
    await seed(sessions, vacancy, profile)
    transport = RecordingSession()
    bot = Bot(TEST_TOKEN, session=transport)
    dispatcher = build_dispatcher(UserService(sessions))
    await dispatcher.feed_update(bot, command_update("/history"))
    assert any(
        button.callback_data == "history:reset"
        for message in transport.requests
        if isinstance(message, SendMessage) and message.reply_markup
        for row in message.reply_markup.inline_keyboard
        for button in row
    )
    await dispatcher.feed_update(bot, update("history:reset"))
    prompt = next(
        message
        for message in reversed(transport.requests)
        if isinstance(message, SendMessage) and "Сбросить историю" in message.text
    )
    confirm = prompt.reply_markup.inline_keyboard[0][0].callback_data
    cancel = prompt.reply_markup.inline_keyboard[1][0].callback_data
    assert len((await FeedbackService(sessions).page(42, "history"))[1]) == 1
    await dispatcher.feed_update(bot, update(cancel, "cancel"))
    await dispatcher.feed_update(bot, update(confirm, "stale"))
    assert len((await FeedbackService(sessions).page(42, "history"))[1]) == 1
    await dispatcher.feed_update(bot, command_update("/history_clear"))
    prompt = next(
        message
        for message in reversed(transport.requests)
        if isinstance(message, SendMessage) and "Сбросить историю" in message.text
    )
    await dispatcher.feed_update(
        bot, update(prompt.reply_markup.inline_keyboard[0][0].callback_data)
    )
    assert (await FeedbackService(sessions).page(42, "history"))[1] == []
    assert any(
        isinstance(message, EditMessageText) and "История сброшена" in message.text
        for message in transport.requests
    )
    assert any(
        isinstance(message, AnswerCallbackQuery) and message.show_alert
        for message in transport.requests
    )


@pytest.mark.postgres
async def test_competing_redeliveries_after_reset_send_only_once(
    postgres_sessions, vacancy, profile
):
    await seed(postgres_sessions, vacancy, profile)
    feedback = FeedbackService(postgres_sessions)
    draft = await feedback.request_history_reset(42)
    results = await asyncio.gather(
        *(feedback.confirm_history_reset(42, draft.token) for _ in range(4)), return_exceptions=True
    )
    assert sum(result == 1 for result in results) == 1
    assert sum(isinstance(result, ValueError) for result in results) == 3
    sender = SimpleNamespace(send_recommendation=AsyncMock(return_value=88))
    service = NotificationService(postgres_sessions, profile, chat_id=42, min_score=0)
    reports = await asyncio.gather(*(service.deliver(sender) for _ in range(6)))
    assert sum(report.sent for report in reports) == 1
    sender.send_recommendation.assert_awaited_once()


@pytest.mark.postgres
async def test_reset_serializes_with_feedback_and_keeps_other_user(
    postgres_sessions, vacancy, profile
):
    await seed(postgres_sessions, vacancy, profile, users=(42, 99))
    feedback = FeedbackService(postgres_sessions)
    draft = await feedback.request_history_reset(42)
    await asyncio.gather(
        feedback.confirm_history_reset(42, draft.token),
        feedback.record(42, vacancy.id, "save", event_key="concurrent-save"),
    )
    assert (await feedback.card(42, vacancy.id))[1].saved
    assert (await feedback.page(42, "history"))[1] == []
    assert len((await feedback.page(99, "history"))[1]) == 1
    async with postgres_sessions() as session:
        row = await session.get(TelegramUserModel, 42)
        assert row.history_reset_token is None
