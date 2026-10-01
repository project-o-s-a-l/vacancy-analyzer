"""facts, concurrency and privacy; no external transport or model."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from sqlalchemy import event, select
from test_duplicate_vacancies import seed

from domain.delivery import DeliveryFailure
from jobs.feedback import FeedbackService
from jobs.notify import NotificationService
from jobs.resumes import ResumeService
from jobs.users import UserService
from observability import logging_session
from observability.settings import LoggingOptions
from storage.models import NotificationModel
from storage.repositories import VacancyRepository
from storage.repositories.notification_repository import NotificationRepository
from storage.repositories.user_repository import RegisteredUser, UserRepository

SECRET = "PRIVATE_RESUME+79991234567@example.com"


def logs(capsys):
    captured = capsys.readouterr()
    assert not captured.out and SECRET not in captured.err
    return [json.loads(line) for line in captured.err.splitlines()]


def facts(rows, name):
    return [row for row in rows if row["event"] == name]


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def log_sessions(request):
    return request.getfixturevalue(
        "postgres_sessions" if request.param == "postgres" else "sessions"
    )


async def test_six_copies_reset_one_repeat_and_separate_users(
    log_sessions, vacancy, profile, capsys
):
    cards = await seed(log_sessions, vacancy, profile)
    service = NotificationService(log_sessions, profile, chat_id=42, min_score=0)
    sender = SimpleNamespace(send_recommendation=AsyncMock(return_value=17))
    with logging_session(console_format="json"):
        assert (await service.deliver(sender)).sent == 1
        rows = logs(capsys)
        selection = facts(rows, "selection.finished")[0]
        assert selection["catalog_rows"] == selection["scanned_count"] == 6
        assert selection["copy_excluded"] == 5
        assert selection["retained_groups"] == selection["scored_count"] == 1
        assert selection["selected_count"] == 1 and selection["top_k"] == 64
        copies = facts(rows, "selection.copy_excluded")
        assert len(copies) == 5 and len({r["duplicate_group"] for r in copies}) == 1
        assert {r["vacancy_id"] for r in copies} | {r["related_vacancy_id"] for r in copies} == {
            str(c.id) for c in cards
        }
        delivery = [
            r
            for r in rows
            if r["event"]
            in {
                "delivery.claimed",
                "delivery.attempt_started",
                "delivery.transport_finished",
                "delivery.persisted",
            }
        ]
        assert [r["event"] for r in delivery] == [
            "delivery.claimed",
            "delivery.attempt_started",
            "delivery.transport_finished",
            "delivery.persisted",
        ]
        assert len({r["operation_id"] for r in delivery}) == 1
        assert all(r["parent_operation_id"] == selection["operation_id"] for r in delivery)
        assert (await service.deliver(sender)).sent == 0
        blocked = facts(logs(capsys), "selection.finished")[0]
        assert blocked["history_feedback_excluded"] == 6 and blocked["scanned_count"] == 0
        assert blocked["code"] == "history_feedback_blocked"
        assert (
            len(await NotificationService(log_sessions, profile, chat_id=99, min_score=0).pending())
            == 1
        )
        other = facts(logs(capsys), "selection.finished")[0]
        assert other["history_feedback_excluded"] == 0
        feedback = FeedbackService(log_sessions)
        request = await feedback.request_history_reset(42)
        assert await feedback.confirm_history_reset(42, request.token) == 1
        assert (await service.deliver(sender)).sent == 1
        assert (await service.deliver(sender)).sent == 0
        rows = logs(capsys)
        assert facts(rows, "history.cleared")[0]["count"] == 1
        assert len(facts(rows, "delivery.persisted")) == 1
    assert sender.send_recommendation.await_count == 2


@pytest.mark.parametrize("status", ["failed", "uncertain", "sent"])
async def test_transport_and_persistence_outcomes(log_sessions, vacancy, profile, capsys, status):
    await seed(log_sessions, vacancy, profile)
    failure = (
        None if status == "sent" else DeliveryFailure(uncertain=status == "uncertain", kind=SECRET)
    )
    sender = SimpleNamespace(send_recommendation=AsyncMock(return_value=1, side_effect=failure))
    service = NotificationService(log_sessions, profile, chat_id=42, min_score=0)
    with logging_session(console_format="json", level="DEBUG"):
        report = await service.deliver(sender)
        rows = logs(capsys)
        assert getattr(report, status if status != "sent" else "sent") == 1
        transport = facts(rows, "delivery.transport_finished")[0]
        persisted = facts(rows, "delivery.persisted")[0]
        assert transport["result"] == persisted["result"] == status
        if status != "sent":
            assert transport["code"] == (
                "transport_uncertain" if status == "uncertain" else "transport_rejected"
            )
        assert transport["operation_id"] == persisted["operation_id"]
        assert bool(await service.pending()) == (status == "failed")
        logs(capsys)


async def test_failed_commit_after_transport_success_is_never_saved_success(
    log_sessions, vacancy, profile, capsys
):
    await seed(log_sessions, vacancy, profile)
    service = NotificationService(log_sessions, profile, chat_id=42, min_score=0)
    sender = SimpleNamespace(send_recommendation=AsyncMock(return_value=22))
    original = service._finish

    async def fail_commit(*args, **kwargs):
        from sqlalchemy.orm import Session

        def fail(session):
            raise RuntimeError(SECRET)

        event.listen(Session, "before_commit", fail)
        try:
            return await original(*args, **kwargs)
        finally:
            event.remove(Session, "before_commit", fail)

    service._finish = fail_commit
    with logging_session(console_format="json", level="DEBUG"):
        with pytest.raises(RuntimeError):
            await service.deliver(sender)
        rows = logs(capsys)
        assert facts(rows, "delivery.transport_finished")[0]["result"] == "sent"
        assert not facts(rows, "delivery.persisted")
        assert facts(rows, "delivery.persistence_failed")[0]["result"] == "uncertain"
        assert await service.pending() == []
        logs(capsys)
    assert await service.delivery_counts() == {"sending": 1}


async def test_claim_and_finish_facts_require_commit(log_sessions, vacancy, profile, capsys):
    cards = await seed(log_sessions, vacancy, profile)
    with logging_session(console_format="json"):
        async with log_sessions() as session:
            with pytest.raises(RuntimeError):
                async with session.begin():
                    await NotificationRepository(session).claim(42, cards[0].id)
                    assert not logs(capsys)
                    raise RuntimeError(SECRET)
            assert not logs(capsys)
        async with log_sessions.begin() as session:
            token = await NotificationRepository(session).claim(42, cards[0].id)
        assert len(facts(logs(capsys), "delivery.claimed")) == 1
        async with log_sessions() as session:
            with pytest.raises(RuntimeError):
                async with session.begin():
                    await NotificationRepository(session).finish(
                        42, cards[0].id, token, "sent", message_id=2
                    )
                    raise RuntimeError(SECRET)
            assert not facts(logs(capsys), "delivery.persisted")


async def test_cancelled_transport_remains_uncertain(log_sessions, vacancy, profile, capsys):
    await seed(log_sessions, vacancy, profile)
    service = NotificationService(log_sessions, profile, chat_id=42, min_score=0)
    sender = SimpleNamespace(send_recommendation=AsyncMock(side_effect=asyncio.CancelledError()))
    with logging_session(console_format="json"):
        with pytest.raises(asyncio.CancelledError):
            await service.deliver(sender)
        rows = logs(capsys)
        assert facts(rows, "delivery.transport_finished")[0]["result"] == "uncertain"
        assert facts(rows, "delivery.persisted")[0]["result"] == "uncertain"
        feedback = FeedbackService(log_sessions)
        assert await feedback.request_history_reset(42) is None
        assert await service.pending() == []
        logs(capsys)
    assert await service.delivery_counts() == {"uncertain": 1}


async def test_profile_change_before_claim_is_a_skip(log_sessions, vacancy, profile, capsys):
    await seed(log_sessions, vacancy, profile)
    service = NotificationService(log_sessions, profile, chat_id=42, min_score=0)
    selection = service._selection

    async def change_after_selection():
        result = await selection()
        changed = profile.model_copy(deep=True)
        changed.experience_years = 100
        async with log_sessions.begin() as session:
            await UserRepository(session).save(RegisteredUser(42, changed, None))
        return result

    service._selection = change_after_selection
    sender = SimpleNamespace(send_recommendation=AsyncMock())
    with logging_session(console_format="json"):
        assert (await service.deliver(sender)).sent == 0
        rows = logs(capsys)
    assert facts(rows, "delivery.skipped")[0]["code"] == "profile_changed"
    assert not facts(rows, "delivery.claimed") and not facts(rows, "delivery.attempt_started")
    sender.send_recommendation.assert_not_awaited()


async def test_copy_details_are_bounded_but_total_is_exact(sessions, vacancy, profile, capsys):
    await seed(sessions, vacancy, profile, copies=24)
    with logging_session(console_format="json"):
        selected = await NotificationService(sessions, profile, chat_id=42, min_score=0).pending()
        rows = logs(capsys)
    assert len(selected) == 1
    assert len(facts(rows, "selection.copy_excluded")) == 16
    assert facts(rows, "selection.finished")[0]["copy_excluded"] == 23


async def test_warning_skips_aggregate_and_preserves_order(
    sessions, vacancy, profile, capsys, monkeypatch
):
    from storage.repositories.candidate_repository import CandidateRepository

    await seed(sessions, vacancy, profile)
    service = NotificationService(sessions, profile, chat_id=42, min_score=0)
    with logging_session(console_format="json"):
        reference = await service.pending()
        logs(capsys)
    monkeypatch.setattr(
        CandidateRepository, "diagnostic_counts", AsyncMock(side_effect=AssertionError())
    )
    with logging_session(console_format="json", level="WARNING"):
        quiet = await service.pending()
        assert logs(capsys) == []
    assert [(v.id, m.score) for v, m in quiet] == [(v.id, m.score) for v, m in reference]


async def test_selection_counter_stages(log_sessions, vacancy, profile, capsys):
    # Two active copies, one inactive, one SQL-required exclusion.
    profile = profile.model_copy(deep=True)
    profile.preferences.remote = "required"
    current = vacancy.model_copy(update={"remote": True})
    cards = await seed(log_sessions, current, profile, copies=2)
    async with log_sessions.begin() as session:
        repo = VacancyRepository(session)
        await repo.save(
            cards[0].model_copy(update={"id": uuid4(), "source_id": "3001", "archived": True})
        )
        await repo.save(
            cards[0].model_copy(update={"id": uuid4(), "source_id": "3002", "remote": False})
        )
    service = NotificationService(log_sessions, profile, chat_id=42, min_score=0)
    with logging_session(console_format="json"):
        assert len(await service.pending()) == 1
        summary = facts(logs(capsys), "selection.finished")[0]
    assert summary["catalog_rows"] == 4
    assert summary["state_excluded"] == summary["required_sql_excluded"] == 1
    assert summary["available_rows"] == summary["scanned_count"] == 2
    assert summary["retained_groups"] == summary["selected_count"] == 1


async def test_empty_reason_is_about_selection_not_entire_catalog(
    log_sessions, vacancy, profile, capsys
):
    profile = profile.model_copy(deep=True)
    profile.preferences.remote = "required"
    await seed(log_sessions, vacancy.model_copy(update={"remote": False}), profile)
    with logging_session(console_format="json"):
        assert await NotificationService(log_sessions, profile, chat_id=42).pending() == []
        summary = facts(logs(capsys), "selection.finished")[0]
    assert summary["catalog_rows"] == summary["required_sql_excluded"] == 6
    assert summary["code"] == "no_active_candidates"


async def test_below_threshold_is_distinct(log_sessions, vacancy, profile, capsys):
    profile = profile.model_copy(deep=True)
    profile.skills = {}
    profile.preferences.positions = ["Другое направление"]
    await seed(log_sessions, vacancy, profile)
    with logging_session(console_format="json"):
        assert (
            await NotificationService(log_sessions, profile, chat_id=42, min_score=100).pending()
            == []
        )
        summary = facts(logs(capsys), "selection.finished")[0]
    assert summary["below_threshold_count"] == 1 and summary["code"] == "below_threshold"


@pytest.mark.parametrize("action", ["dislike", "apply"])
async def test_feedback_still_blocks_copies_after_clear(
    log_sessions, vacancy, profile, capsys, action
):
    cards = await seed(log_sessions, vacancy, profile)
    sender = SimpleNamespace(send_recommendation=AsyncMock(return_value=1))
    service = NotificationService(log_sessions, profile, chat_id=42, min_score=0)
    assert (await service.deliver(sender)).sent == 1
    feedback = FeedbackService(log_sessions)
    async with log_sessions() as session:
        sent_id = await session.scalar(select(NotificationModel.vacancy_id))
    with logging_session(console_format="json"):
        await feedback.record(42, sent_id, action, event_key=SECRET)
        await feedback.record(42, sent_id, action, event_key=SECRET)
        request = await feedback.request_history_reset(42)
        await feedback.confirm_history_reset(42, request.token)
        assert await service.pending() == []
        with pytest.raises(ValueError):
            await feedback.record(99, cards[0].id, "save", event_key=SECRET)
        rows = logs(capsys)
    assert len(facts(rows, "feedback.changed")) == len(facts(rows, "feedback.reused")) == 1
    assert facts(rows, "feedback.rejected")[0]["code"] == "feedback_not_allowed"
    assert facts(rows, "selection.finished")[0]["history_feedback_excluded"] == 6


async def test_resume_and_history_no_tokens_text_or_values(log_sessions, capsys, tmp_path):
    await UserService(log_sessions).register(42)
    resumes = ResumeService(log_sessions)
    options = LoggingOptions(_env_file=None, directory=tmp_path / "logs")
    with logging_session(console_format="json", level="DEBUG", file_options=options):
        draft = await resumes.prepare(
            42, "Желаемая должность: Python Developer\nНавыки: python\n" + SECRET
        )
        await resumes.confirm(42, draft.token)
        with pytest.raises(ValueError):
            await resumes.confirm(42, draft.token)
        new = await resumes.prepare(
            42, "Желаемая должность: Java Developer\nНавыки: java\n" + SECRET
        )
        await resumes.cancel(42, new.token)
        feedback = FeedbackService(log_sessions)
        assert await feedback.request_history_reset(42) is None
        with pytest.raises(ValueError):
            await feedback.confirm_history_reset(42, SECRET)
        rows = logs(capsys)
    assert len(facts(rows, "resume.confirmed")) == len(facts(rows, "resume.cancelled")) == 1
    assert len(facts(rows, "resume.rejected")) == 1
    assert facts(rows, "history.rejected")[0]["code"] == "stale_history_request"
    file_rows = [
        json.loads(line)
        for path in options.directory.glob("*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert file_rows == rows
    serialized = json.dumps(file_rows, ensure_ascii=False)
    assert all(
        value not in serialized
        for value in (SECRET, draft.token, new.token, "Python Developer", "Java Developer")
    )


async def test_profile_stale_resume_is_not_applied(log_sessions, capsys):
    users = UserService(log_sessions)
    await users.register(42)
    resumes = ResumeService(log_sessions)
    draft = await resumes.prepare(42, "Желаемая должность: Python Developer\nНавыки: python")
    user = await users.get(42)
    changed = user.profile.model_copy(deep=True)
    changed.experience_years = 100
    async with log_sessions.begin() as session:
        await UserRepository(session).save(RegisteredUser(42, changed, None))
    with logging_session(console_format="json"):
        with pytest.raises(ValueError):
            await resumes.confirm(42, draft.token)
        rows = logs(capsys)
    assert facts(rows, "resume.rejected")[0]["code"] == "stale_resume"
    assert not facts(rows, "resume.confirmed")
    assert (await users.get(42)).profile.experience_years == 100


async def test_history_cancel_is_committed(log_sessions, vacancy, profile, capsys):
    await seed(log_sessions, vacancy, profile)
    await NotificationService(log_sessions, profile, chat_id=42, min_score=0).deliver(
        SimpleNamespace(send_recommendation=AsyncMock(return_value=1))
    )
    feedback = FeedbackService(log_sessions)
    request = await feedback.request_history_reset(42)
    with logging_session(console_format="json"):
        await feedback.cancel_history_reset(42, request.token)
        with pytest.raises(ValueError):
            await feedback.confirm_history_reset(42, request.token)
        rows = logs(capsys)
    assert facts(rows, "history.cancelled")[0]["result"] == "cancelled"
    assert not facts(rows, "history.cleared")
    assert len((await feedback.page(42, "history"))[1]) == 1
    request = await feedback.request_history_reset(42)
    with logging_session(console_format="json"):
        await UserService(log_sessions).cancel(42)
        rows = logs(capsys)
    assert facts(rows, "history.cancelled")[0]["code"] == "user_cancelled"
    with pytest.raises(ValueError):
        await feedback.confirm_history_reset(42, request.token)


@pytest.mark.postgres
async def test_competing_group_claim_events(postgres_sessions, vacancy, profile, capsys):
    cards = await seed(postgres_sessions, vacancy, profile)

    async def claim(card):
        async with postgres_sessions.begin() as session:
            return await NotificationRepository(session).claim(42, card.id)

    with logging_session(console_format="json"):
        results = await asyncio.gather(*(claim(card) for card in cards))
        rows = logs(capsys)
    assert sum(token is not None for token in results) == 1
    assert len(facts(rows, "delivery.claimed")) == 1
    assert len(facts(rows, "delivery.skipped")) == 5
    assert all(r["code"] == "delivery_blocked" for r in facts(rows, "delivery.skipped"))
    assert len({r["duplicate_group"] for r in rows}) == 1


@pytest.mark.postgres
async def test_competing_delivery_of_different_copy_ids(
    postgres_sessions, vacancy, profile, capsys
):
    from scoring import score_vacancy

    cards = await seed(postgres_sessions, vacancy, profile)
    sender = SimpleNamespace(send_recommendation=AsyncMock(return_value=19))

    async def deliver(card):
        service = NotificationService(postgres_sessions, profile, chat_id=42, min_score=0)
        service._selection = AsyncMock(return_value=([(card, score_vacancy(card, profile))], None))
        return await service.deliver(sender)

    with logging_session(console_format="json"):
        reports = await asyncio.gather(*(deliver(card) for card in cards))
        rows = logs(capsys)
    assert sum(report.sent for report in reports) == 1
    sender.send_recommendation.assert_awaited_once()
    assert len(facts(rows, "delivery.claimed")) == len(facts(rows, "delivery.persisted")) == 1
    assert len(facts(rows, "delivery.skipped")) == 5
    assert len({r["operation_id"] for r in facts(rows, "delivery.skipped")}) == 5
