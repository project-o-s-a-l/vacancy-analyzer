import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from aiogram.exceptions import (
    TelegramBadRequest,
    TelegramForbiddenError,
    TelegramNetworkError,
    TelegramRetryAfter,
    TelegramServerError,
)
from aiogram.methods import SendMessage
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from bot.delivery import TelegramSender
from domain import CandidateProfile, Vacancy
from jobs.notify import NotificationService
from storage.models import NotificationModel
from storage.repositories import VacancyRepository
from storage.repositories.notification_repository import NotificationRepository


@pytest.fixture
async def seeded_sessions(sessions, sample_data):
    async with sessions.begin() as session:
        for index, payload in enumerate(sample_data["vacancies"]):
            payload["published_at"] = (datetime.now(UTC) - timedelta(days=index)).isoformat()
            await VacancyRepository(session).save(Vacancy.model_validate(payload))
    return sessions


@pytest.fixture
def sender():
    return SimpleNamespace(send_message=AsyncMock(return_value=SimpleNamespace(message_id=123)))


async def test_delivery_service_accepts_non_telegram_sender(seeded_sessions, profile):
    transport = SimpleNamespace(send_recommendation=AsyncMock(return_value=77))
    service = NotificationService(seeded_sessions, profile, chat_id=42, top=1, min_score=0)
    report = await service.deliver(transport)
    assert report.sent == 1
    transport.send_recommendation.assert_awaited_once()
    async with seeded_sessions() as session:
        row = await session.scalar(select(NotificationModel).where(NotificationModel.chat_id == 42))
        assert row.message_id == 77 and row.status == "sent"


async def test_claim_is_durable_and_unique_per_chat(session, vacancy):
    async with session.begin():
        await VacancyRepository(session).save(vacancy)
        repository = NotificationRepository(session)
        token = await repository.claim(42, vacancy.id)
        assert token is not None
        assert await repository.claim(42, vacancy.id) is None
        assert await repository.claim(43, vacancy.id) is not None


async def test_failed_claim_can_retry_with_new_token(session, vacancy):
    async with session.begin():
        await VacancyRepository(session).save(vacancy)
        repository = NotificationRepository(session)
        first = await repository.claim(42, vacancy.id)
        await repository.finish(42, vacancy.id, first, "failed", error_kind="Rejected")
        second = await repository.claim(42, vacancy.id)
        assert second is not None and second != first
        with pytest.raises(ValueError):
            await repository.finish(42, vacancy.id, first, "sent", message_id=1)
        await repository.finish(42, vacancy.id, second, "sent", message_id=2)
        assert await repository.claim(42, vacancy.id) is None
    row = await session.get(NotificationModel, (42, vacancy.id))
    assert row.attempts == 2
    assert row.message_id == 2


@pytest.mark.parametrize("state", ["sent", "uncertain", "sending"])
async def test_sent_or_ambiguous_attempt_never_auto_retried(session, vacancy, state):
    async with session.begin():
        await VacancyRepository(session).save(vacancy)
        repository = NotificationRepository(session)
        token = await repository.claim(42, vacancy.id)
        if state != "sending":
            await repository.finish(
                42, vacancy.id, token, state, message_id=1 if state == "sent" else None
            )
        assert await repository.claim(42, vacancy.id) is None


async def test_repeated_runs_progress_past_previously_sent_top(seeded_sessions, profile, sender):
    profile = profile.model_copy(update={"experience_years": 6})
    service = NotificationService(seeded_sessions, profile, chat_id=42, top=2, min_score=0)
    assert (await service.deliver(TelegramSender(sender))).sent == 2
    assert (await service.deliver(TelegramSender(sender))).sent == 1
    assert (await service.deliver(TelegramSender(sender))).sent == 0
    assert sender.send_message.await_count == 3
    for call in sender.send_message.await_args_list:
        assert call.args[0] == 42
        assert call.kwargs["parse_mode"] is None
        assert call.kwargs["reply_markup"].inline_keyboard[0][0].url.startswith("https://")
    assert await service.delivery_counts() == {"sent": 3}


async def test_preview_does_not_claim_or_send(seeded_sessions, profile):
    service = NotificationService(seeded_sessions, profile, chat_id=42, min_score=95)
    first = await service.pending()
    second = await service.pending()
    assert len(first) == len(second) == 1
    assert first[0][0].source_id == "demo-1"
    assert await service.delivery_counts() == {}


async def test_empty_catalog_explains_missing_active_vacancies(sessions, profile, sender):
    service = NotificationService(sessions, profile, chat_id=42)
    report = await service.deliver(TelegramSender(sender))
    assert report.sent == 0
    assert "нет активных вакансий" in report.empty_message
    sender.send_message.assert_not_awaited()
    assert await service.delivery_counts() == {}


async def test_below_threshold_explains_best_score_without_claiming(sessions, vacancy, sender):
    vacancy.published_at = datetime.now(UTC)
    async with sessions.begin() as session:
        await VacancyRepository(session).save(vacancy)
    profile = CandidateProfile.model_validate({"skills": {"java": {}}, "experience_years": 3})
    service = NotificationService(sessions, profile, chat_id=42, min_score=50)
    report = await service.deliver(TelegramSender(sender))
    assert report.sent == 0 and "лучшая оценка" in report.empty_message
    assert "порог — 50%" in report.empty_message
    sender.send_message.assert_not_awaited()
    assert await service.delivery_counts() == {}


async def test_already_shown_is_distinguished_from_below_threshold(
    sessions, vacancy, profile, sender
):
    vacancy.published_at = datetime.now(UTC)
    async with sessions.begin() as session:
        await VacancyRepository(session).save(vacancy)
    service = NotificationService(sessions, profile, chat_id=42, min_score=0)
    assert (await service.deliver(TelegramSender(sender))).sent == 1
    second = await service.deliver(TelegramSender(sender))
    assert second.sent == 0 and "уже отправлены" in second.empty_message
    sender.send_message.assert_awaited_once()


@pytest.mark.parametrize("error_class", [TelegramNetworkError, TelegramServerError])
async def test_network_and_server_errors_leave_ambiguous_attempt_blocked(
    seeded_sessions,
    profile,
    sender,
    error_class,
):
    service = NotificationService(seeded_sessions, profile, chat_id=42, top=1, min_score=0)
    pending = await service.pending()
    sender.send_message.side_effect = error_class(
        method=SendMessage(chat_id=42, text="x"), message="sensitive diagnostic"
    )
    report = await service.deliver(TelegramSender(sender))
    assert report.sent == 0 and report.uncertain == 1
    assert pending[0][0].id not in {vacancy.id for vacancy, _ in await service.pending()}
    async with seeded_sessions() as session:
        row = await session.get(NotificationModel, (42, pending[0][0].id))
        assert row.error_kind == error_class.__name__
        assert "sensitive diagnostic" not in row.error_kind


@pytest.mark.parametrize("error_class", [TelegramBadRequest, TelegramForbiddenError])
async def test_explicit_rejection_remains_retryable(seeded_sessions, profile, sender, error_class):
    service = NotificationService(seeded_sessions, profile, chat_id=42, top=1, min_score=0)
    sender.send_message.side_effect = error_class(
        method=SendMessage(chat_id=42, text="x"), message="no"
    )
    assert (await service.deliver(TelegramSender(sender))).failed == 1
    sender.send_message.side_effect = None
    assert (await service.deliver(TelegramSender(sender))).sent == 1


async def test_rate_limit_does_not_automatically_resend(seeded_sessions, profile, sender):
    service = NotificationService(seeded_sessions, profile, chat_id=42, min_score=0)
    sender.send_message.side_effect = TelegramRetryAfter(
        method=SendMessage(chat_id=42, text="x"), message="slow down", retry_after=10
    )
    assert (await service.deliver(TelegramSender(sender))).failed == 1
    assert sender.send_message.await_count == 1


async def test_cancelled_send_is_not_retried(seeded_sessions, profile, sender):
    service = NotificationService(seeded_sessions, profile, chat_id=42, top=1, min_score=0)
    pending = await service.pending()
    sender.send_message.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await service.deliver(TelegramSender(sender))
    assert (await service.delivery_counts())["uncertain"] == 1
    assert pending[0][0].id not in {vacancy.id for vacancy, _ in await service.pending()}


async def test_send_success_followed_by_database_failure_cannot_duplicate(
    seeded_sessions,
    profile,
    sender,
    monkeypatch,
):
    service = NotificationService(seeded_sessions, profile, chat_id=42, top=1, min_score=0)
    pending = await service.pending()
    monkeypatch.setattr(service, "_finish", AsyncMock(side_effect=SQLAlchemyError()))
    with pytest.raises(SQLAlchemyError):
        await service.deliver(TelegramSender(sender))
    assert (await service.delivery_counts())["sending"] == 1
    assert pending[0][0].id not in {vacancy.id for vacancy, _ in await service.pending()}


async def test_changed_vacancy_does_not_resend(seeded_sessions, profile, sender):
    service = NotificationService(seeded_sessions, profile, chat_id=42, top=1, min_score=0)
    first = (await service.pending())[0][0]
    await service.deliver(TelegramSender(sender))
    updated = first.model_dump()
    updated.update(id=uuid4(), title="Updated Python Developer", raw_payload={"changed": True})
    async with seeded_sessions.begin() as session:
        assert await VacancyRepository(session).save(Vacancy(**updated)) == first.id
    assert first.id not in {vacancy.id for vacancy, _ in await service.pending()}


async def test_summary_for_empty_and_nonempty_database(sessions, profile, vacancy):
    service = NotificationService(sessions, profile, chat_id=42)
    assert (await service.summary())["total"] == 0
    async with sessions.begin() as session:
        await VacancyRepository(session).save(vacancy)
    result = await service.summary()
    assert result["total"] == 1
    assert result["sources"] == {"hh": 1}
    assert result["last_seen"].tzinfo == UTC


async def test_claim_rollback_allows_future_send(session, vacancy):
    async with session.begin():
        await VacancyRepository(session).save(vacancy)
    with pytest.raises(RuntimeError):
        async with session.begin():
            await NotificationRepository(session).claim(42, vacancy.id)
            raise RuntimeError()
    async with session.begin():
        assert await NotificationRepository(session).claim(42, vacancy.id) is not None
    assert len((await session.scalars(select(NotificationModel))).all()) == 1


async def test_cli_dry_run_needs_no_token_and_does_not_record_delivery(
    seeded_sessions,
    profile,
    monkeypatch,
    capsys,
):
    from bot import runtime
    from jobs import cli
    from jobs.settings import Settings
    from storage.repositories.user_repository import RegisteredUser, UserRepository

    async with seeded_sessions.begin() as session:
        repository = UserRepository(session)
        await repository.register(42)
        await repository.save(RegisteredUser(42, profile, None))
    monkeypatch.setattr(cli, "Settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(cli, "load_profile", lambda path: profile)
    monkeypatch.setattr(cli, "create_engine", lambda url: SimpleNamespace(dispose=AsyncMock()))
    monkeypatch.setattr(cli, "session_factory", lambda engine: seeded_sessions)
    monkeypatch.setattr(
        runtime, "create_bot", lambda settings: pytest.fail("Dry run created a bot")
    )
    args = cli.parser().parse_args(["notify", "--user-id", "42", "--dry-run"])
    await cli.run_telegram(args)
    assert "Python Backend Developer" in capsys.readouterr().out
    async with seeded_sessions() as session:
        assert (await session.scalars(select(NotificationModel))).all() == []
