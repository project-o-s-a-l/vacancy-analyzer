import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import delete, select, update
from test_candidates import candidate, job
from test_saved_selections import dumped, rank, seed

from jobs.candidate_cache import identity
from jobs.feedback import FeedbackService
from jobs.rank import rank_recent
from jobs.users import UserService
from storage.models import (
    ExposureModel,
    FeedbackEventModel,
    SavedSelectionModel,
    TelegramUserModel,
    VacancyModel,
)
from storage.repositories import VacancyRepository
from storage.repositories.notification_repository import NotificationRepository
from storage.repositories.user_repository import UserRepository


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def cache_sessions(request):
    return request.getfixturevalue(
        "postgres_sessions" if request.param == "postgres" else "sessions"
    )


async def cached(sessions, *, owner=42, profile=None, **options):
    stats = {}
    async with sessions.begin() as session:
        result = await rank_recent(
            VacancyRepository(session),
            profile or candidate(),
            cache_owner=owner,
            diagnostics=stats,
            **options,
        )
    return result, stats


async def test_hit_skips_shortlist_but_matches_full_and_invalidates_on_sql_write(
    cache_sessions, vacancy, monkeypatch
):
    from storage.repositories.candidate_repository import CandidateRepository

    sessions = cache_sessions
    await seed(sessions, vacancy)
    now = datetime.now(UTC)
    first, stats = await cached(sessions, now=now)
    assert stats["candidate_cache_hit"] == 0
    original = CandidateRepository.shortlist
    spy = AsyncMock(side_effect=AssertionError("A valid candidate hit must not run SQL ranking"))
    monkeypatch.setattr(CandidateRepository, "shortlist", spy)
    second, stats = await cached(sessions, now=now)
    assert stats["candidate_cache_hit"] == 1 and dumped(second) == dumped(first)
    spy.assert_not_called()
    monkeypatch.setattr(CandidateRepository, "shortlist", original)
    async with sessions.begin() as s:
        await s.execute(
            update(VacancyModel).where(VacancyModel.id == first[0][0].id).values(archived=True)
        )
    changed, stats = await cached(sessions, now=now)
    full, _ = await rank(sessions, owner=None, now=now)
    assert stats["candidate_cache_hit"] == 0 and dumped(changed) == dumped(full)


async def test_future_publication_expiry_and_day_boundary_refill(cache_sessions, vacancy):
    sessions = cache_sessions
    now = datetime(2026, 11, 9, 23, 59, 50, tzinfo=UTC)
    cards = [
        job(vacancy, title=f"Java Developer {i}", published_at=now - timedelta(days=1))
        for i in range(4)
    ]
    cards[0].valid_until = now + timedelta(seconds=3)
    cards[1].published_at = now + timedelta(seconds=5)
    async with sessions.begin() as s:
        for v in cards:
            await VacancyRepository(s).save(v)
    for at in (
        now,
        now + timedelta(seconds=2),
        now + timedelta(seconds=4),
        now + timedelta(seconds=6),
        now + timedelta(seconds=12),
    ):
        actual, _ = await cached(sessions, now=at, limit=2)
        full, _ = await rank(sessions, owner=None, now=at, limit=2)
        assert dumped(actual) == dumped(full)


async def test_revision_rollback_and_other_owner_are_isolated(cache_sessions, vacancy):
    sessions = cache_sessions
    await seed(sessions, vacancy)
    await cached(sessions)
    async with sessions() as s:
        before = await identity(s, 42)
    async with sessions() as s:
        await s.execute(update(VacancyModel).values(archived=True))
        assert (await identity(s, 42))[0] != before[0]
        await s.rollback()
    _, stats = await cached(sessions)
    assert stats["candidate_cache_hit"] == 1
    _, stats = await cached(sessions, owner=99)
    assert stats["candidate_cache_hit"] == 0


@pytest.mark.postgres
async def test_catalogue_revision_does_not_invert_entity_lock_order(postgres_sessions, vacancy):
    sessions = postgres_sessions
    cards = await seed(sessions, vacancy, n=2)
    first_locked, second_locked, allow_first = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def first():
        async with sessions.begin() as s:
            await s.execute(
                update(VacancyModel)
                .where(VacancyModel.id == cards[0].id)
                .values(check_error="test")
            )
            first_locked.set()
            await second_locked.wait()
            allow_first.set()
            await s.execute(
                update(VacancyModel)
                .where(VacancyModel.id == cards[1].id)
                .values(check_error="test")
            )

    async def second():
        await first_locked.wait()
        async with sessions.begin() as s:
            await s.execute(
                update(VacancyModel)
                .where(VacancyModel.id == cards[1].id)
                .values(check_error="other")
            )
            second_locked.set()
            await allow_first.wait()

    await asyncio.wait_for(asyncio.gather(first(), second()), 5)


async def test_document_expiry_invalidates_pair_and_result_at_midnight(cache_sessions, vacancy):
    from domain.qualifications import Qualification

    sessions = cache_sessions
    now = datetime(2026, 11, 9, 23, 59, 55, tzinfo=UTC)
    profile = candidate(qualifications=[Qualification(name="медкнижка", valid_until=now.date())])
    card = job(
        vacancy,
        description="Требования: медкнижка обязательна.",
        published_at=now - timedelta(days=1),
    )
    async with sessions.begin() as s:
        await VacancyRepository(s).save(card)
    first, _ = await cached(sessions, profile=profile, now=now)
    assert len(first) == 1
    same, stats = await cached(sessions, profile=profile, now=now + timedelta(seconds=1))
    assert len(same) == 1 and stats["candidate_cache_hit"] == 1
    expired, stats = await cached(sessions, profile=profile, now=now + timedelta(seconds=10))
    assert expired == [] and stats["candidate_cache_hit"] == 0


async def test_repeated_empty_history_result_does_not_repeat_expensive_diagnostic_probe(
    cache_sessions, vacancy, monkeypatch
):
    from jobs.notify import NotificationService
    from storage.repositories.candidate_repository import CandidateRepository

    sessions = cache_sessions
    await seed(sessions, vacancy, n=1)
    service = NotificationService(sessions, candidate(), chat_id=42, min_score=0)
    assert (
        await service.deliver(SimpleNamespace(send_recommendation=AsyncMock(return_value=7)))
    ).sent == 1
    assert await service.pending() == []
    spy = AsyncMock(side_effect=AssertionError("Repeated history diagnostic SQL ranking"))
    monkeypatch.setattr(CandidateRepository, "shortlist", spy)
    assert await service.pending() == []
    spy.assert_not_called()


@pytest.mark.postgres
@pytest.mark.parametrize("mutation", ["edit", "delete", "delete_resume", "aba"])
async def test_owner_mutation_during_ranking_cannot_publish_or_recreate_cache(
    postgres_sessions, vacancy, monkeypatch, mutation
):
    sessions = postgres_sessions
    await seed(sessions, vacancy)
    users = UserService(sessions, min_score=0)
    await users.register(42)
    async with sessions.begin() as s:
        repo = UserRepository(s)
        user = await repo.get(42)
        await repo.save(replace(user, profile=candidate(), form_state=None))
    if mutation == "delete_resume":
        await users.resumes.prepare(42, "Java Python SQL developer, опыт 3 года. " * 10)
    notifications = await users.notifications(42)
    from jobs import saved_selections

    original = saved_selections.score_selection
    entered, proceed = asyncio.Event(), asyncio.Event()

    async def blocked(*args, **kwargs):
        entered.set()
        await proceed.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(saved_selections, "score_selection", blocked)
    sender = SimpleNamespace(send_recommendation=AsyncMock(return_value=7))
    task = asyncio.create_task(notifications.deliver(sender))
    await entered.wait()
    if mutation == "delete_resume":
        await users.resumes.delete(42)
    else:
        async with sessions.begin() as s:
            if mutation in {"delete", "aba"}:
                await s.execute(delete(TelegramUserModel).where(TelegramUserModel.user_id == 42))
            else:
                repo = UserRepository(s)
                user = await repo.get(42, lock=True)
                await repo.save(replace(user, profile=candidate(skills={"python": {}})))
        if mutation == "aba":
            await users.register(42)
            async with sessions.begin() as s:
                repo = UserRepository(s)
                user = await repo.get(42)
                await repo.save(replace(user, profile=candidate(), form_state=None))
    proceed.set()
    report = await task
    assert report.sent == 0 and "Профиль изменился" in report.empty_message
    sender.send_recommendation.assert_not_called()
    async with sessions() as s:
        assert await s.get(SavedSelectionModel, 42) is None


async def test_old_card_resolves_original_confirmed_snapshot_after_redelivery(
    cache_sessions, vacancy
):
    sessions = cache_sessions
    await UserService(sessions).register(42)
    async with sessions.begin() as s:
        await VacancyRepository(s).save(vacancy)
        repo = NotificationRepository(s)
        token = await repo.claim(42, vacancy.id, exposure_context={"score": 10})
        await repo.finish(42, vacancy.id, token, "sent", message_id=100)
        await repo.reset_history(42, before=datetime.now(UTC))
        token = await repo.claim(42, vacancy.id, exposure_context={"score": 90})
        await repo.finish(42, vacancy.id, token, "sent", message_id=200)
    feedback = FeedbackService(sessions)
    await feedback.record(42, vacancy.id, "like", event_key="old", exposure_message_id=100)
    await feedback.record(42, vacancy.id, "save", event_key="new", exposure_message_id=200)
    with pytest.raises(ValueError, match="не подтверждён"):
        await feedback.record(42, vacancy.id, "like", event_key="unknown", exposure_message_id=300)
    async with sessions() as s:
        assert (await s.get(FeedbackEventModel, (42, "old"))).exposure_context == {"score": 10}
        assert (await s.get(FeedbackEventModel, (42, "new"))).exposure_context == {"score": 90}
        assert len(list(await s.scalars(select(ExposureModel)))) == 2


@pytest.mark.postgres
async def test_slow_delivery_owner_does_not_hold_ranking_or_sql_slot(postgres_sessions, vacancy):
    from jobs.notify import NotificationService

    sessions = postgres_sessions
    await seed(sessions, vacancy, n=1)
    entered, release = asyncio.Event(), asyncio.Event()

    async def slow(*args):
        entered.set()
        await release.wait()
        return 11

    slow_task = asyncio.create_task(
        NotificationService(sessions, candidate(), chat_id=42, min_score=0).deliver(
            SimpleNamespace(send_recommendation=slow)
        )
    )
    await entered.wait()
    fast = NotificationService(sessions, candidate(), chat_id=99, min_score=0)
    report = await asyncio.wait_for(
        fast.deliver(SimpleNamespace(send_recommendation=AsyncMock(return_value=12))), 2
    )
    assert report.sent == 1 and not slow_task.done()
    release.set()
    assert (await slow_task).sent == 1
