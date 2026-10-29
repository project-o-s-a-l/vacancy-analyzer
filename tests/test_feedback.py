import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from aiogram import Bot
from aiogram.exceptions import TelegramNetworkError
from aiogram.methods import AnswerCallbackQuery, EditMessageReplyMarkup, SendMessage
from sqlalchemy import func, select
from test_bot import TEST_TOKEN, RecordingSession, command_update
from test_users import callback_update

from bot.delivery import TelegramSender
from bot.handlers import build_dispatcher
from bot.keyboards import reason_keyboard, vacancy_keyboard
from domain.feedback import FeedbackState
from jobs.feedback import FeedbackService
from jobs.notify import NotificationService
from jobs.users import UserService
from storage.models import FeedbackEventModel, NotificationModel
from storage.payloads import payload_hash
from storage.repositories import VacancyRepository
from storage.repositories.notification_repository import NotificationRepository
from storage.repositories.user_repository import RegisteredUser, UserRepository


async def seed(sessions, vacancy, profile, *, users=(42,), context=None):
    vacancy.published_at = datetime.now(UTC)
    async with sessions.begin() as session:
        await VacancyRepository(session).save(vacancy)
        for identifier in users:
            repo = UserRepository(session)
            user = await repo.register(identifier)
            await repo.save(RegisteredUser(identifier, profile, None, user.revision))
            notifications = NotificationRepository(session)
            token = await notifications.claim(identifier, vacancy.id, exposure_context=context)
            message_id = (
                10
                + await session.scalar(
                    select(func.count())
                    .select_from(NotificationModel)
                    .where(NotificationModel.chat_id == identifier)
                )
                - 1
            )
            await notifications.finish(identifier, vacancy.id, token, "sent", message_id=message_id)


@pytest.fixture
async def feedback(sessions, vacancy, profile):
    await seed(sessions, vacancy, profile)
    return FeedbackService(sessions)


async def event_count(sessions):
    async with sessions() as session:
        return await session.scalar(select(func.count()).select_from(FeedbackEventModel))


async def test_states_are_independent_and_survive_service_restart(feedback, sessions, vacancy):
    await feedback.record(42, vacancy.id, "like", event_key="1")
    await feedback.record(42, vacancy.id, "save", event_key="2")
    await feedback.record(42, vacancy.id, "apply", event_key="3")
    _, state = await FeedbackService(sessions).card(42, vacancy.id)
    assert state.interest == "like" and state.saved and state.applied and state.version == 3
    await feedback.record(42, vacancy.id, "unsave", event_key="4")
    state = await feedback.record(42, vacancy.id, "unapply", event_key="5")
    assert state.interest == "like" and not state.saved and not state.applied
    assert await event_count(sessions) == 5


async def test_duplicate_callback_is_idempotent(feedback, sessions, vacancy):
    first = await feedback.record(42, vacancy.id, "save", event_key="same")
    repeated = await feedback.record(42, vacancy.id, "save", event_key="same")
    assert first == repeated and repeated.version == 1
    assert await event_count(sessions) == 1
    with pytest.raises(ValueError, match="другие данные"):
        await feedback.record(42, vacancy.id, "apply", event_key="same")
    assert (await feedback.card(42, vacancy.id))[1] == first


async def test_reason_is_versioned_and_cleared_when_interest_changes(feedback, sessions, vacancy):
    disliked = await feedback.record(42, vacancy.id, "dislike", event_key="1")
    state = await feedback.record(
        42, vacancy.id, "reason", event_key="2", reason="salary", expected_version=disliked.version
    )
    assert state.interest == "dislike" and state.reason == "salary"
    liked = await feedback.record(42, vacancy.id, "like", event_key="3")
    assert liked.reason is None and liked.interest == "like"
    with pytest.raises(ValueError, match="уже изменилась"):
        await feedback.record(
            42,
            vacancy.id,
            "reason",
            event_key="stale",
            reason="company",
            expected_version=disliked.version,
        )
    assert await event_count(sessions) == 3
    assert (await feedback.card(42, vacancy.id))[1] == liked


async def test_reason_may_be_skipped_and_duplicate_reason_does_not_change_version(
    feedback, vacancy
):
    state = await feedback.record(42, vacancy.id, "dislike", event_key="1")
    skipped = await feedback.record(
        42, vacancy.id, "reason", event_key="2", expected_version=state.version
    )
    assert skipped.reason is None and skipped.interest == "dislike"
    assert (
        await feedback.record(
            42, vacancy.id, "reason", event_key="2", expected_version=state.version
        )
        == skipped
    )


@pytest.mark.parametrize(
    "action,reason,key",
    [
        ("oops", None, "x"),
        ("like", "salary", "x"),
        ("reason", "oops", "x"),
        ("like", None, ""),
        ("like", None, "x" * 129),
    ],
)
async def test_invalid_actions_do_not_write_events(
    feedback, sessions, vacancy, action, reason, key
):
    with pytest.raises(ValueError):
        await feedback.record(42, vacancy.id, action, reason=reason, event_key=key)
    assert await event_count(sessions) == 0


async def test_feedback_requires_registered_user_and_own_exposure(
    feedback, sessions, vacancy, profile
):
    with pytest.raises(ValueError, match="/start"):
        await feedback.record(99, vacancy.id, "like", event_key="x")
    async with sessions.begin() as session:
        await UserRepository(session).register(99)
    with pytest.raises(ValueError, match="не была показана"):
        await feedback.record(99, vacancy.id, "like", event_key="x")
    with pytest.raises(ValueError, match="не была показана"):
        await feedback.record(42, uuid4(), "like", event_key="x")
    assert await event_count(sessions) == 0


async def test_users_keep_separate_flags_and_event_keys(sessions, vacancy, profile):
    await seed(sessions, vacancy, profile, users=(42, 99))
    service = FeedbackService(sessions)
    await service.record(42, vacancy.id, "save", event_key="same")
    await service.record(99, vacancy.id, "apply", event_key="same")
    a, b = (await service.card(42, vacancy.id))[1], (await service.card(99, vacancy.id))[1]
    assert a.saved and not a.applied and b.applied and not b.saved
    assert len((await service.page(42, "saved"))[1]) == 1
    assert (await service.page(99, "saved"))[1] == []
    assert await event_count(sessions) == 2


async def test_event_retains_exposure_context_after_profile_and_vacancy_changes(
    sessions, vacancy, profile
):
    context = {
        "profile": profile.model_dump(mode="json"),
        "match": {"score": 72.5},
        "rank": 2,
        "payload_hash": payload_hash(vacancy.raw_payload),
    }
    await seed(sessions, vacancy, profile, context=context)
    async with sessions.begin() as session:
        user = await UserRepository(session).get(42)
        user.profile.experience_years = 0
        await UserRepository(session).save(user)
        vacancy.title = "Changed title"
        vacancy.raw_payload = {"updated": True}
        await VacancyRepository(session).save(vacancy)
    await FeedbackService(sessions).record(42, vacancy.id, "like", event_key="x")
    async with sessions() as session:
        event = await session.get(FeedbackEventModel, (42, "x"))
        assert event.exposure_context == context
        assert event.exposure_context["profile"]["experience_years"] != 0
        assert event.action_context["profile"]["experience_years"] == 0
        assert event.action_context["payload_hash"] == payload_hash({"updated": True})


async def test_notification_captures_original_score_profile_rank_and_payload_hash(
    sessions, vacancy, profile
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    vacancy.published_at = datetime.now(UTC)
    async with sessions.begin() as session:
        await VacancyRepository(session).save(vacancy)
    service = NotificationService(sessions, profile, chat_id=42, min_score=0)
    sender = SimpleNamespace(send_message=AsyncMock(return_value=SimpleNamespace(message_id=11)))
    assert (await service.deliver(TelegramSender(sender))).sent == 1
    async with sessions() as session:
        exposure = (await session.get(NotificationModel, (42, vacancy.id))).exposure_context
        assert exposure["profile"] == profile.model_dump(mode="json")
        assert exposure["match"]["score"] > 0 and exposure["rank"] == 1
        assert exposure["payload_hash"] == payload_hash(vacancy.raw_payload)


@pytest.mark.parametrize("action", ["dislike", "apply"])
async def test_rejected_and_applied_vacancies_stay_suppressed(
    feedback, sessions, vacancy, profile, action
):
    await feedback.record(42, vacancy.id, action, event_key="x")
    async with sessions.begin() as session:
        notification = await session.get(NotificationModel, (42, vacancy.id))
        notification.status = "failed"  # Delivery alone no longer blocks this ID.
    assert await NotificationService(sessions, profile, chat_id=42, min_score=0).pending() == []


async def test_pagination_is_private_and_preserves_archived_bookmarks(sessions, vacancy, profile):
    identifiers = []
    for index in range(7):
        item = vacancy.model_copy(
            update={
                "id": uuid4(),
                "source_id": str(index),
                "archived": True,
                "title": f"Разная вакансия {index}",
            }
        )
        await seed(sessions, item, profile)
        await FeedbackService(sessions).record(42, item.id, "save", event_key=str(index))
        identifiers.append(item.id)
    service = FeedbackService(sessions)
    first = (await service.page(42, "saved", 0))[1]
    second = (await service.page(42, "saved", 1))[1]
    assert len(first) == 6 and len(second) == 2
    assert len({v.id for v, _ in first[:5]} | {v.id for v, _ in second}) == 7
    assert all(v.archived for v, _ in first)
    assert len((await service.page(42, "history"))[1]) == 6


def update(data, event_key="callback", user_id=42):
    result = callback_update(data, user_id=user_id)
    return result.model_copy(
        update={
            "callback_query": result.callback_query.model_copy(
                update={
                    "id": event_key,
                    "message": result.callback_query.message.model_copy(update={"message_id": 10}),
                }
            )
        }
    )


async def test_real_dispatcher_records_feedback_and_updates_buttons(feedback, sessions, vacancy):
    service = UserService(sessions)
    session = RecordingSession()
    bot = Bot(TEST_TOKEN, session=session)
    dispatcher = build_dispatcher(service)
    await dispatcher.feed_update(bot, update(f"fb:save:{vacancy.id.hex}"))
    assert (await feedback.card(42, vacancy.id))[1].saved
    edits = [request for request in session.requests if isinstance(request, EditMessageReplyMarkup)]
    assert edits and any(
        button.callback_data == f"fb:unsave:{vacancy.id.hex}"
        for row in edits[-1].reply_markup.inline_keyboard
        for button in row
    )
    await dispatcher.feed_update(bot, update(f"fb:dislike:{vacancy.id.hex}", "dislike"))
    state = (await feedback.card(42, vacancy.id))[1]
    await dispatcher.feed_update(
        bot, update(f"fbr:{vacancy.id.hex}:{state.version}:skills", "reason")
    )
    assert (await feedback.card(42, vacancy.id))[1].reason == "skills"


@pytest.mark.parametrize("command", ["/saved", "/applied", "/history"])
async def test_saved_applied_and_history_commands_show_own_cards(
    feedback, sessions, vacancy, command
):
    await feedback.record(42, vacancy.id, "save", event_key="save")
    await feedback.record(42, vacancy.id, "apply", event_key="apply")
    session = RecordingSession()
    bot = Bot(TEST_TOKEN, session=session)
    await build_dispatcher(UserService(sessions)).feed_update(bot, command_update(command))
    cards = [
        request
        for request in session.requests
        if isinstance(request, SendMessage) and request.reply_markup
    ]
    assert cards and vacancy.title in cards[0].text
    assert cards[0].reply_markup.inline_keyboard[0][0].url == str(vacancy.url)


async def test_history_pagination_and_archived_label(sessions, vacancy, profile, monkeypatch):
    from unittest.mock import AsyncMock

    monkeypatch.setattr("bot.recommendations.asyncio.sleep", AsyncMock())
    for index in range(7):
        item = vacancy.model_copy(
            update={
                "id": uuid4(),
                "source_id": str(index),
                "archived": True,
                "title": f"Разная вакансия {index}",
            }
        )
        await seed(sessions, item, profile)
    session = RecordingSession()
    bot = Bot(TEST_TOKEN, session=session)
    await build_dispatcher(UserService(sessions)).feed_update(bot, update("fbpage:history:1"))
    messages = [request for request in session.requests if isinstance(request, SendMessage)]
    assert any("страница 2" in request.text for request in messages)
    cards = [request for request in messages if request.text.startswith("Вакансия в архиве")]
    assert len(cards) == 2
    assert any(
        button.callback_data == "fbpage:history:0"
        for request in messages
        if request.reply_markup
        for row in request.reply_markup.inline_keyboard
        for button in row
    )


@pytest.mark.parametrize(
    "data",
    [
        "fb:delete:bad",
        "fb:like:" + "z" * 32,
        "fbr:" + "0" * 32 + ":1:unknown",
        "fbpage:history:999999",
    ],
)
async def test_malformed_callbacks_do_not_write(feedback, sessions, data):
    session = RecordingSession()
    bot = Bot(TEST_TOKEN, session=session)
    await build_dispatcher(UserService(sessions)).feed_update(bot, update(data))
    assert await event_count(sessions) == 0
    assert any(
        isinstance(request, AnswerCallbackQuery) and request.show_alert
        for request in session.requests
    )


async def test_callback_does_not_expose_another_users_vacancy(feedback, sessions, vacancy):
    async with sessions.begin() as session:
        await UserRepository(session).register(99)
    session = RecordingSession()
    bot = Bot(TEST_TOKEN, session=session)
    await build_dispatcher(UserService(sessions)).feed_update(
        bot, update(f"fb:like:{vacancy.id.hex}", user_id=99)
    )
    assert await event_count(sessions) == 0
    assert not any(isinstance(request, EditMessageReplyMarkup) for request in session.requests)


async def test_telegram_edit_failure_does_not_undo_committed_action(feedback, sessions, vacancy):
    class FailingSession(RecordingSession):
        async def make_request(self, bot, method, timeout=None):  # noqa: ASYNC109 — aiogram API
            if isinstance(method, EditMessageReplyMarkup):
                raise TelegramNetworkError(method=method, message="test")
            return await super().make_request(bot, method, timeout)

    bot = Bot(TEST_TOKEN, session=FailingSession())
    dispatcher = build_dispatcher(UserService(sessions))
    event = update(f"fb:save:{vacancy.id.hex}")
    await dispatcher.feed_update(bot, event)
    await dispatcher.feed_update(bot, event)
    assert (await feedback.card(42, vacancy.id))[1].saved
    assert await event_count(sessions) == 1


def test_callbacks_fit_telegram_limit_and_contain_no_user_ids(vacancy):
    for keyboard in [
        vacancy_keyboard(vacancy),
        reason_keyboard(FeedbackState(vacancy.id, version=2_147_483_647)),
    ]:
        for row in keyboard.inline_keyboard:
            for button in row:
                if button.callback_data:
                    assert len(button.callback_data.encode()) <= 64


@pytest.mark.postgres
async def test_concurrent_duplicate_callbacks_are_one_event(postgres_sessions, vacancy, profile):
    await seed(postgres_sessions, vacancy, profile)
    service = FeedbackService(postgres_sessions)
    states = await asyncio.gather(
        *(service.record(42, vacancy.id, "save", event_key="same") for _ in range(8))
    )
    assert all(state.saved and state.version == 1 for state in states)
    assert await event_count(postgres_sessions) == 1


@pytest.mark.postgres
async def test_concurrent_independent_actions_do_not_overwrite_each_other(
    postgres_sessions, vacancy, profile
):
    await seed(postgres_sessions, vacancy, profile)
    service = FeedbackService(postgres_sessions)
    await asyncio.gather(
        service.record(42, vacancy.id, "save", event_key="save"),
        service.record(42, vacancy.id, "apply", event_key="apply"),
    )
    state = (await service.card(42, vacancy.id))[1]
    assert state.saved and state.applied and state.version == 2
    assert await event_count(postgres_sessions) == 2
