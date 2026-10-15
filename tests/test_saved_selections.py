import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import delete, func, select, update
from test_candidates import candidate, job

from domain.serving_ranker import RankerPolicy
from jobs.feedback import FeedbackService
from jobs.notify import NotificationService
from jobs.rank import rank_recent
from jobs.users import UserService
from storage.models import NotificationModel, SavedSelectionModel, TelegramUserModel, VacancyModel
from storage.repositories import VacancyRepository
from storage.repositories.selection_repository import MAX_POOL


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def selection_sessions(request):
    return request.getfixturevalue(
        "postgres_sessions" if request.param == "postgres" else "sessions"
    )


async def seed(sessions, vacancy, n=8):
    now = datetime.now(UTC)
    cards = [
        job(vacancy, title=f"Java Developer {i}", published_at=now - timedelta(days=i + 1))
        for i in range(n)
    ]
    async with sessions.begin() as session:
        for v in cards:
            await VacancyRepository(session).save(v)
    return cards


async def rank(sessions, profile=None, owner=42, **kwargs):
    stats = {}
    async with sessions.begin() as session:
        result = await rank_recent(
            VacancyRepository(session),
            profile or candidate(),
            cache_owner=owner,
            cache_stats=stats,
            **kwargs,
        )
    return result, stats


def dumped(ranking):
    # A compatibility read can rebuild the same facts at a later observation time.
    return [
        (
            v.model_dump(mode="json", exclude={"professional_facts": {"extracted_at"}}),
            m.model_dump(mode="json"),
        )
        for v, m in ranking
    ]


async def test_repeat_matches_full_calculation_and_persists_across_sessions(
    selection_sessions, vacancy
):
    sessions = selection_sessions
    await seed(sessions, vacancy)
    now = datetime.now(UTC)
    first, stats = await rank(sessions, now=now)
    assert stats == {"selected": 8, "reused": 0, "recomputed": 8}
    later = now + timedelta(minutes=4)
    second, stats = await rank(sessions, now=later)
    full, _ = await rank(sessions, owner=None, now=later)
    assert stats == {"selected": 8, "reused": 8, "recomputed": 0}
    assert dumped(second) == dumped(full)
    assert first[0][1].components["freshness"] != second[0][1].components["freshness"]
    async with sessions() as session:
        row = await session.get(SavedSelectionModel, 42)
        assert len(row.entries) == 8 and len(row.request_revision) == 64
        assert "profile" not in row.context and "raw_payload" not in str(row.entries)
        assert await session.scalar(select(func.count()).select_from(NotificationModel)) == 0


async def test_only_changed_card_recalculated_and_new_cards_visible(selection_sessions, vacancy):
    sessions = selection_sessions
    cards = await seed(sessions, vacancy)
    await rank(sessions)
    async with sessions.begin() as session:
        await session.execute(
            update(VacancyModel)
            .where(VacancyModel.id == cards[3].id)
            .values(description="Java Spring SQL: новые требования")
        )
        extra = job(vacancy, title="Java Developer new", published_at=datetime.now(UTC))
        await VacancyRepository(session).save(extra)
    now = datetime.now(UTC)
    actual, stats = await rank(sessions, now=now)
    full, _ = await rank(sessions, owner=None, now=now)
    assert stats == {"selected": 9, "reused": 7, "recomputed": 2}
    assert extra.id in {v.id for v, _ in actual}
    assert dumped(actual) == dumped(full)


@pytest.mark.parametrize("change", ["archived", "expired", "deleted", "future"])
async def test_cached_candidates_recheck_activity(selection_sessions, vacancy, change):
    sessions = selection_sessions
    cards = await seed(sessions, vacancy)
    await rank(sessions)
    now = datetime.now(UTC)
    async with sessions.begin() as session:
        stmt = update(VacancyModel).where(VacancyModel.id == cards[0].id)
        if change == "archived":
            stmt = stmt.values(archived=True)
        elif change == "expired":
            stmt = stmt.values(valid_until=now - timedelta(seconds=1))
        elif change == "future":
            stmt = stmt.values(published_at=now + timedelta(days=1))
        else:
            stmt = delete(VacancyModel).where(VacancyModel.id == cards[0].id)
        await session.execute(stmt)
    actual, stats = await rank(sessions)
    assert cards[0].id not in {v.id for v, _ in actual}
    assert stats["recomputed"] == 0


@pytest.mark.parametrize("reason", ["profile", "owner", "ttl", "date", "contract", "parameters"])
async def test_invalidation_and_owner_isolation(selection_sessions, vacancy, monkeypatch, reason):
    sessions = selection_sessions
    await seed(sessions, vacancy)
    now = datetime(2026, 10, 27, 23, 59, tzinfo=UTC)
    # Fix card dates for UTC-day-boundary checks independent of the host clock.
    async with sessions.begin() as session:
        await session.execute(update(VacancyModel).values(published_at=now - timedelta(days=2)))
    await rank(sessions, now=now)
    kwargs = {"now": now}
    if reason == "profile":
        kwargs["profile"] = candidate(skills={"java": {}, "spring": {}, "sql": {}})
    elif reason == "owner":
        kwargs["owner"] = 99
    elif reason == "ttl":
        kwargs["now"] = now + timedelta(minutes=16)
    elif reason == "date":
        kwargs["now"] = now + timedelta(minutes=2)
    elif reason == "parameters":
        kwargs["limit"] = 7
    else:
        from jobs import saved_selections

        monkeypatch.setattr(saved_selections, "selection_contract", lambda: "a" * 64)
    _, stats = await rank(sessions, **kwargs)
    assert stats["reused"] == 0 and stats["recomputed"] > 0


async def test_pool_larger_than_page_and_history_reset_preserves_feedback(
    selection_sessions, vacancy
):
    sessions = selection_sessions
    await seed(sessions, vacancy, n=16)
    profile = candidate()
    users = UserService(sessions)
    await users.register(42)
    async with sessions.begin() as session:
        await session.execute(
            update(TelegramUserModel)
            .where(TelegramUserModel.user_id == 42)
            .values(profile=profile.model_dump(mode="json"), form_state=None)
        )
    service = NotificationService(sessions, profile, chat_id=42, top=1, min_score=0)
    sender = SimpleNamespace(send_recommendation=AsyncMock(return_value=101))
    first = (await service.pending())[0][0]
    assert (await service.deliver(sender)).sent == 1
    next_card = (await service.pending())[0][0]
    assert next_card.id != first.id
    async with sessions() as session:
        row = await session.get(SavedSelectionModel, 42)
        assert len(row.entries) == 16
    feedback = FeedbackService(sessions)
    draft = await feedback.request_history_reset(42)
    await feedback.confirm_history_reset(42, draft.token)
    assert (await service.pending())[0][0].id == first.id
    await feedback.record(42, first.id, "dislike", event_key="decline")
    assert all(v.id != first.id for v, _ in await service.pending())


async def test_corrupt_entry_recalculated_and_owner_pool_bounded(selection_sessions, vacancy):
    sessions = selection_sessions
    await seed(sessions, vacancy)
    await rank(sessions)
    async with sessions.begin() as session:
        row = await session.get(SavedSelectionModel, 42)
        entries = list(row.entries)
        entries[0] = entries[0] | {"match": {"score": "broken"}}
        row.entries = entries
    _, stats = await rank(sessions)
    assert stats["recomputed"] == 1
    for offset in range(3):
        cards = [job(vacancy, title=f"Java Developer unique{offset}-{i}") for i in range(64)]
        async with sessions.begin() as session:
            for v in cards:
                await VacancyRepository(session).save(v)
        excluded = set()
        async with sessions() as session:
            all_ids = set(await session.scalars(select(VacancyModel.id)))
            excluded = all_ids - {v.id for v in cards}
        await rank(sessions, excluded_ids=excluded)
    async with sessions() as session:
        row = await session.get(SavedSelectionModel, 42)
        assert len(row.entries) == MAX_POOL
        assert await session.scalar(select(func.count()).select_from(SavedSelectionModel)) == 1


async def test_current_ranker_always_applied_to_reused_baselines(sessions, vacancy, monkeypatch):
    from jobs import rank as module

    await seed(sessions, vacancy)
    calls = []

    def apply(items, profile, policy, params):
        assert all(match.ranking is None for _, match in items)
        calls.append(policy.model_path)
        return items

    monkeypatch.setattr(module, "apply_ranker", apply)
    await rank(sessions, ranker_policy=RankerPolicy(mode="shadow", model_path="first.json"))
    _, stats = await rank(
        sessions, ranker_policy=RankerPolicy(mode="model", model_path="second.json")
    )
    assert stats["reused"] == 8 and len(calls) == 2 and calls[0] != calls[1]


async def test_changed_or_missing_model_cannot_reuse_previous_model_decision(
    selection_sessions, vacancy, tmp_path
):
    from test_serving_ranker import artifact, save

    sessions = selection_sessions
    await seed(sessions, vacancy)
    now = datetime.now(UTC)
    path = save(tmp_path, artifact())
    active = RankerPolicy(mode="model", model_path=path)
    first, _ = await rank(sessions, now=now, ranker_policy=active)
    assert all(m.ranking.applied for _, m in first)
    # Remove the same file after warming the persistent baseline pool.
    path.unlink()
    fallback, stats = await rank(sessions, now=now, ranker_policy=active)
    full, _ = await rank(sessions, now=now, owner=None, ranker_policy=active)
    assert stats["reused"] == 8 and dumped(fallback) == dumped(full)
    assert all(m.ranking.fallback_reason == "model_unavailable" for _, m in fallback)
    rules, stats = await rank(sessions, now=now)
    assert stats["reused"] == 8 and all(m.ranking is None for _, m in rules)


async def test_disabled_cache_preserves_delivery_path_without_creating_pool(sessions, vacancy):
    await seed(sessions, vacancy)
    service = NotificationService(
        sessions, candidate(), chat_id=42, min_score=0, saved_selections_enabled=False
    )
    first, second = await service.pending(), await service.pending()
    assert [v.id for v, _ in first] == [v.id for v, _ in second]
    async with sessions() as session:
        assert await session.scalar(select(func.count()).select_from(SavedSelectionModel)) == 0


@pytest.mark.parametrize("change", ["closed", "profile", "conditions", "feedback"])
async def test_change_between_selection_and_send_is_not_delivered(
    selection_sessions, vacancy, monkeypatch, change
):
    sessions = selection_sessions
    cards = await seed(sessions, vacancy, n=1)
    profile = candidate()
    await UserService(sessions).register(42)
    async with sessions.begin() as session:
        await session.execute(
            update(TelegramUserModel)
            .where(TelegramUserModel.user_id == 42)
            .values(profile=profile.model_dump(mode="json"), form_state=None)
        )
    service = NotificationService(sessions, profile, chat_id=42, min_score=0)
    original = service._selection

    async def changing_selection():
        result = await original()
        async with sessions.begin() as session:
            if change == "profile":
                await session.execute(
                    update(TelegramUserModel)
                    .where(TelegramUserModel.user_id == 42)
                    .values(profile=candidate(experience_years=0).model_dump(mode="json"))
                )
            elif change == "feedback":
                from storage.models import FeedbackModel

                session.add(
                    FeedbackModel(
                        user_id=42,
                        vacancy_id=cards[0].id,
                        interest="dislike",
                        saved=False,
                        applied=False,
                        version=1,
                        updated_at=datetime.now(UTC),
                    )
                )
            else:
                await session.execute(
                    update(VacancyModel)
                    .where(VacancyModel.id == cards[0].id)
                    .values(**({"archived": True} if change == "closed" else {"experience": "6+"}))
                )
        return result

    monkeypatch.setattr(service, "_selection", changing_selection)
    sender = SimpleNamespace(send_recommendation=AsyncMock(return_value=77))
    assert (await service.deliver(sender)).sent == 0
    sender.send_recommendation.assert_not_awaited()


@pytest.mark.postgres
async def test_competing_cached_requests_preserve_claim_uniqueness(postgres_sessions, vacancy):
    sessions = postgres_sessions
    await seed(sessions, vacancy, n=1)
    service = NotificationService(sessions, candidate(), chat_id=42, min_score=0)
    sender = SimpleNamespace(send_recommendation=AsyncMock(return_value=77))
    reports = await asyncio.gather(*(service.deliver(sender) for _ in range(4)))
    assert sum(report.sent for report in reports) == 1
    sender.send_recommendation.assert_awaited_once()
    async with sessions() as session:
        assert await session.scalar(select(func.count()).select_from(SavedSelectionModel)) == 1
