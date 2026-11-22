import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import EditMessageReplyMarkup, SendMessage
from test_bot import TEST_TOKEN, RecordingSession, command_update
from test_users import callback_update, store_profile

from bot import runtime
from bot.handlers import build_dispatcher
from domain import Vacancy
from jobs import cli
from jobs.settings import Settings
from jobs.users import UserService
from storage.models import NotificationModel
from storage.repositories import VacancyRepository
from storage.repositories.notification_repository import REVIEW_SEND_TIMEOUT, NotificationRepository


async def seed_feed(sessions, sample_data, profile):
    await store_profile(sessions, 42, profile.model_copy(update={"experience_years": 6}))
    async with sessions.begin() as session:
        for payload in sample_data["vacancies"]:
            await VacancyRepository(session).save(
                Vacancy.model_validate(payload | {"published_at": datetime.now(UTC).isoformat()})
            )
    return UserService(sessions, min_score=0)


def buttons(method):
    return {b.text: b for row in method.reply_markup.inline_keyboard for b in row}


def cards(transport):
    return [
        (message_id, method)
        for message_id, method in enumerate(transport.requests, 1)
        if isinstance(method, SendMessage)
        and method.reply_markup
        and any(b.url for b in buttons(method).values())
    ]


def card_action(card, label, event_id):
    message_id, method = card
    return callback_update(
        buttons(method)[label].callback_data, event_key=event_id, message_id=message_id
    )


async def test_dispatcher_advances_only_after_rating_the_current_card(
    sessions, sample_data, profile
):
    users = await seed_feed(sessions, sample_data, profile)
    dispatcher = build_dispatcher(users)
    transport = RecordingSession()
    async with Bot(TEST_TOKEN, session=transport) as bot:

        async def click(card, label, event_id):
            await dispatcher.feed_update(bot, card_action(card, label, event_id))

        await dispatcher.feed_update(bot, command_update("/latest"))
        [first] = cards(transport)
        dispatcher = build_dispatcher(UserService(sessions, min_score=0))
        await dispatcher.feed_update(bot, callback_update("menu:latest"))
        assert cards(transport) == [first]
        assert transport.requests[-1].reply_parameters.message_id == first[0]

        await click(first, "Сохранить", "save")
        assert cards(transport) == [first]
        await dispatcher.feed_update(bot, command_update("/saved"))
        _, library_copy = cards(transport)
        await click(first, "👍 Интересно", "first-like")
        _, _, second = cards(transport)
        await click(library_copy, "👍 Интересно", "rate-saved-copy")
        await click(first, "👍 Интересно", "first-like")  # Telegram replay
        await click(first, "👎 Не подходит", "change-old-rating")
        assert len(cards(transport)) == 3

        await click(second, "👎 Не подходит", "second-dislike")
        _, _, _, third = cards(transport)
        assert len({card[1].text for card in (first, second, third)}) == 3
        reason_menu = next(
            r for r in reversed(transport.requests) if isinstance(r, EditMessageReplyMarkup)
        )
        await click((second[0], reason_menu), "Зарплата", "optional-reason")
        assert len(cards(transport)) == 4

        await click(third, "👍 Интересно", "last-like")
        assert "Новых рекомендаций пока нет" in transport.requests[-1].text
        assert len(cards(transport)) == 4  # three recommendations and a saved copy
        counts = await (await users.notifications(42)).delivery_counts()
        assert counts == {"sent": 3}


async def test_deleted_card_can_be_rated_in_history_to_resume_latest(
    sessions, sample_data, profile
):
    class DeletedCardSession(RecordingSession):
        async def make_request(self, bot, method, timeout=None):  # noqa: ASYNC109
            if isinstance(method, SendMessage) and method.reply_parameters:
                assert method.reply_parameters.allow_sending_without_reply is False
                raise TelegramBadRequest(method=method, message="message to be replied not found")
            return await super().make_request(bot, method, timeout)

    users = await seed_feed(sessions, sample_data, profile)
    dispatcher = build_dispatcher(users)
    transport = DeletedCardSession()
    async with Bot(TEST_TOKEN, session=transport) as bot:
        await dispatcher.feed_update(bot, command_update("/latest"))
        await dispatcher.feed_update(bot, command_update("/latest"))
        assert "/history" in transport.requests[-1].text
        await dispatcher.feed_update(bot, command_update("/history"))
        original, copy = cards(transport)
        await dispatcher.feed_update(bot, card_action(copy, "👍 Интересно", "from-history"))
        assert cards(transport) == [original, copy]
        await dispatcher.feed_update(bot, command_update("/latest"))
        _, _, second = cards(transport)
        assert original[1].text != second[1].text


async def test_operator_notify_uses_the_feed_and_a_failed_keyboard_edit_allows_the_next_card(
    sessions, sample_data, profile, monkeypatch
):
    class UneditableCardSession(RecordingSession):
        async def make_request(self, bot, method, timeout=None):  # noqa: ASYNC109
            if isinstance(method, EditMessageReplyMarkup):
                raise TelegramBadRequest(method=method, message="message is not modified")
            return await super().make_request(bot, method, timeout)

    users = await seed_feed(sessions, sample_data, profile)
    transport = UneditableCardSession()
    monkeypatch.setattr(cli, "Settings", lambda: Settings(_env_file=None, telegram_min_score=0))
    monkeypatch.setattr(cli, "create_engine", lambda url: SimpleNamespace(dispose=AsyncMock()))
    monkeypatch.setattr(cli, "session_factory", lambda engine: sessions)
    monkeypatch.setattr(runtime, "create_bot", lambda settings: Bot(TEST_TOKEN, session=transport))
    args = cli.parser().parse_args(["notify", "--user-id", "42"])
    await cli.run_telegram(args)
    [first] = cards(transport)
    await cli.run_telegram(args)
    assert cards(transport) == [first]
    async with Bot(TEST_TOKEN, session=transport) as bot:
        await build_dispatcher(users).feed_update(bot, card_action(first, "👍 Интересно", "rate"))
    assert len(cards(transport)) == 2


@pytest.mark.postgres
async def test_overlapping_selections_and_ratings_use_one_review_slot(
    postgres_sessions, sample_data, profile, monkeypatch
):
    users = await seed_feed(postgres_sessions, sample_data, profile)
    candidates = await (await users.notifications(42)).pending()
    barrier = asyncio.Barrier(2)

    async def deliver(candidate):
        service = await users.notifications(42, wait_for_rating=True)

        async def simultaneous_selection():
            await barrier.wait()
            return [candidate], None

        monkeypatch.setattr(service, "_selection", simultaneous_selection)
        return await service.deliver(sender)

    sender = SimpleNamespace(send_recommendation=AsyncMock(return_value=17))
    async with asyncio.timeout(20):
        reports = await asyncio.gather(*(deliver(v) for v in candidates[:2]))
    assert sorted(r.sent for r in reports) == [0, 1]
    sender.send_recommendation.assert_awaited_once()
    vacancy_id = sender.send_recommendation.await_args.args[1].id
    states = await asyncio.gather(
        *(
            users.feedback.record(42, vacancy_id, action, event_key=action, exposure_message_id=17)
            for action in ("like", "dislike")
        )
    )
    assert sum(s.advance_feed for s in states) == 1


@pytest.mark.parametrize("outcome", ["failed", "uncertain", "interrupted", "history_reset"])
async def test_review_slot_recovers_without_losing_delivery_history(sessions, vacancy, outcome):
    async with sessions.begin() as session:
        await VacancyRepository(session).save(vacancy)
        repo = NotificationRepository(session)
        token = await repo.claim(42, vacancy.id, awaiting_rating=True)
        assert (await repo.pending_review(42)).status == "sending"
        if outcome == "interrupted":
            row = await session.get(NotificationModel, (42, vacancy.id))
            row.attempted_at = datetime.now(UTC) - REVIEW_SEND_TIMEOUT - timedelta(seconds=1)
            await session.flush()
        elif outcome == "history_reset":
            await repo.finish(42, vacancy.id, token, "sent", message_id=8)
            assert (await repo.pending_review(42)).message_id == 8
            await repo.reset_history(42, before=datetime.now(UTC))
        else:
            await repo.finish(42, vacancy.id, token, outcome)
        assert await repo.pending_review(42) is None
        assert (vacancy.id in await repo.blocked_ids(42)) == (
            outcome in {"uncertain", "interrupted"}
        )
