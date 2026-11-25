from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
import requests
from sqlalchemy import func, select

from collectors.base import SearchQuery
from collectors.errors import CollectionStopped, VacancyParseError
from domain import CandidateProfile
from domain.vacancy_analysis import analysis_input_hash, is_fresh
from jobs.catalog_checks import check_catalog
from jobs.collect import collect_vacancies
from jobs.collection_scheduler import collection_loop, load_profile_queries, profile_queries
from jobs.rank import rank_recent
from jobs.search_plan import plan_profile
from storage.models import SearchPlanModel, VacancyModel, VacancySnapshot
from storage.repositories import VacancyRepository
from storage.repositories.user_repository import RegisteredUser, UserRepository


def candidate(title):
    return CandidateProfile(skills={"предметное умение": {}}, preferences={"positions": [title]})


@pytest.mark.parametrize(
    "title,expected",
    [
        (
            "Преподаватель разработки на Unity",
            ["Преподаватель разработки на Unity", "Преподаватель Unity"],
        ),
        (
            "Преподаватель программирования на Python",
            ["Преподаватель программирования на Python", "Преподаватель Python"],
        ),
        ("Бухгалтер", ["Бухгалтер"]),
        ("Водитель категории CE", ["Водитель категории CE"]),
        ("Медицинская сестра", ["Медицинская сестра"]),
        ("Менеджер по продажам", ["Менеджер по продажам"]),
        ("JavaScript Developer", ["JavaScript Developer"]),
    ],
)
def test_expansion_keeps_profession_and_subject(title, expected):
    plan = plan_profile(candidate(title))
    assert [item.text for item in plan] == expected
    assert all(item.original == title and item.origin == "position" for item in plan)
    template = SearchQuery(text="unused", search_field="name")
    queries = profile_queries([candidate(title), candidate(title)], template)
    assert [q.text for q in queries] == expected
    assert all(q.search_field == "all" and q.period is None for q in queries)


async def test_plan_persists_owner_and_original_and_removes_stale_mapping(db_sessions):
    sessions = db_sessions
    async with sessions.begin() as session:
        repo = UserRepository(session)
        for owner in (1, 2):
            await repo.register(owner)
            await repo.save(
                RegisteredUser(owner, candidate("Преподаватель разработки на Unity"), None)
            )
    queries = await load_profile_queries(sessions, SearchQuery(text="unused"))
    assert len(queries) == 2  # Shared by both owners, no duplicate network work.
    async with sessions() as session:
        rows = list(
            await session.scalars(select(SearchPlanModel).order_by(SearchPlanModel.user_id))
        )
        assert len(rows) == 2
        assert rows[0].queries[1] == {
            "original": "Преподаватель разработки на Unity",
            "text": "Преподаватель Unity",
            "reason": "short_teacher_title",
            "origin": "position",
            "search_field": "all",
        }
        updated = rows[0].updated_at
    await load_profile_queries(sessions, SearchQuery(text="unused"))
    async with sessions.begin() as session:
        assert (await session.get(SearchPlanModel, 1)).updated_at == updated
        await UserRepository(session).save(
            RegisteredUser(
                1, candidate("Бухгалтер"), None, (await UserRepository(session).get(1)).revision
            )
        )
        await UserRepository(session).save(
            RegisteredUser(
                2,
                candidate("Unity"),
                "setup:skills",
                (await UserRepository(session).get(2)).revision,
            )
        )
    queries = await load_profile_queries(sessions, SearchQuery(text="unused"))
    assert [q.text for q in queries] == ["Бухгалтер"]
    async with sessions() as session:
        assert (await session.get(SearchPlanModel, 1)).queries[0]["text"] == "Бухгалтер"
        assert await session.get(SearchPlanModel, 2) is None


async def save(sessions, vacancy):
    async with sessions.begin() as session:
        return await VacancyRepository(session).save(vacancy)


async def metadata(sessions, key):
    async with sessions() as session:
        return await session.get(VacancyModel, key)


async def test_identical_and_changed_source_keep_identity_dates_and_revisions(db_sessions, vacancy):
    sessions = db_sessions
    key = await save(sessions, vacancy)
    observed = vacancy.collected_at + timedelta(days=2)
    refreshed = vacancy.model_copy(
        update={
            "id": uuid4(),
            "collected_at": observed,
            "published_at": vacancy.published_at + timedelta(days=2),
        }
    )
    assert await save(sessions, refreshed) == key
    row = await metadata(sessions, key)
    assert row.revision_number == 1
    assert row.published_at.replace(tzinfo=UTC) == vacancy.published_at
    assert row.collected_at.replace(tzinfo=UTC) == vacancy.collected_at
    assert row.last_seen_at.replace(tzinfo=UTC) == observed
    assert row.last_checked_at.replace(tzinfo=UTC) == observed
    changed = refreshed.model_copy(
        update={"description": "Иные обязанности", "raw_payload": {"new": "source"}}
    )
    assert await save(sessions, changed) == key
    row = await metadata(sessions, key)
    assert row.revision_number == 2 and row.content_revision == analysis_input_hash(changed)
    # Raw-only changes retain their source snapshot, but do not change semantic revision.
    await save(sessions, changed.model_copy(update={"raw_payload": {"new": "tracking change"}}))
    assert (await metadata(sessions, key)).revision_number == 2
    async with sessions() as session:
        assert await session.scalar(select(func.count()).select_from(VacancySnapshot)) == 3


async def test_same_card_from_multiple_queries_and_empty_direction(db_sessions, vacancy):
    now = datetime.now(UTC)
    vacancy = vacancy.model_copy(
        update={"published_at": now - timedelta(days=200), "valid_until": None}
    )
    source = SimpleNamespace(search=AsyncMock(side_effect=[[], [vacancy], [vacancy]]))
    await collection_loop(
        source,
        db_sessions,
        SearchQuery(text="unused"),
        False,
        3600,
        query_texts=["Учитель", "Бухгалтер", "Водитель"],
    )
    assert source.search.await_count == 3
    async with db_sessions() as session:
        assert await session.scalar(select(func.count()).select_from(VacancyModel)) == 1
        assert await session.scalar(select(func.count()).select_from(VacancySnapshot)) == 1


@pytest.mark.parametrize("status", [404, 410])
async def test_direct_missing_confirms_closed_and_search_can_reopen(db_sessions, vacancy, status):
    sessions = db_sessions
    key = await save(sessions, vacancy)
    now = vacancy.collected_at + timedelta(days=3)
    response = requests.Response()
    response.status_code = status
    source = SimpleNamespace(
        get_vacancy=AsyncMock(side_effect=requests.HTTPError(response=response))
    )
    assert await check_catalog(source, sessions, now=now) == 1
    row = await metadata(sessions, key)
    assert row.availability == "closed" and row.archived and row.closed_at
    async with sessions() as session:
        assert not is_fresh(await VacancyRepository(session).get_by_id(key), now=now)
    await save(sessions, vacancy.model_copy(update={"collected_at": now + timedelta(hours=1)}))
    row = await metadata(sessions, key)
    assert row.availability == "active" and not row.archived and row.closed_at is None


@pytest.mark.parametrize(
    "error,code,stop",
    [
        (requests.Timeout(), "network_error", False),
        (VacancyParseError("changed markup"), "parse_error", False),
        (CollectionStopped("captcha"), "access_blocked", True),
        (httpx.ReadTimeout("timeout"), "network_error", False),
    ],
)
async def test_uncertainty_never_closes_or_changes_publication(
    db_sessions, vacancy, error, code, stop
):
    sessions = db_sessions
    key = await save(sessions, vacancy)
    now = vacancy.collected_at + timedelta(days=3)
    source = SimpleNamespace(get_vacancy=AsyncMock(side_effect=error))
    if stop:
        with pytest.raises(CollectionStopped):
            await check_catalog(source, sessions, now=now)
    else:
        await check_catalog(source, sessions, now=now)
    row = await metadata(sessions, key)
    assert row.availability == "unknown" and not row.archived and row.closed_at is None
    assert row.check_error == code
    assert row.published_at.replace(tzinfo=UTC) == vacancy.published_at
    # Retry is postponed by the regular check interval; there is no hot retry loop.
    await check_catalog(source, sessions, now=now + timedelta(minutes=1))
    assert source.get_vacancy.await_count == 1


@pytest.mark.parametrize("status", [403, 429])
async def test_recheck_preserves_typed_block_context(db_sessions, vacancy, status):
    key = await save(db_sessions, vacancy)
    response = requests.Response()
    response.status_code = status
    source = SimpleNamespace(
        get_vacancy=AsyncMock(
            side_effect=requests.HTTPError("private credentials", response=response)
        )
    )
    with pytest.raises(CollectionStopped) as stopped:
        await check_catalog(source, db_sessions, now=vacancy.collected_at + timedelta(days=3))
    assert stopped.value.code == f"hh_http_{status}"
    assert stopped.value.status_code == status
    assert stopped.value.stage == "catalog_check"
    assert stopped.value.source_vacancy_id == vacancy.source_id
    row = await metadata(db_sessions, key)
    assert row.check_error == "http_error" and row.availability == "unknown" and not row.archived
    source.get_vacancy.assert_awaited_once()


async def test_rechecks_are_bounded_and_expired_excluded(db_sessions, vacancy):
    sessions = db_sessions
    now = vacancy.collected_at + timedelta(days=3)
    for number in range(5):
        item = vacancy.model_copy(
            update={
                "id": uuid4(),
                "source_id": str(number),
                "valid_until": now - timedelta(hours=1) if number == 0 else None,
            }
        )
        await save(sessions, item)
    source = SimpleNamespace(
        get_vacancy=AsyncMock(side_effect=lambda sid: vacancy.model_copy(update={"source_id": sid}))
    )
    assert await check_catalog(source, sessions, now=now) == 2
    assert source.get_vacancy.await_count == 2
    async with sessions() as session:
        expired = await session.scalar(select(VacancyModel).where(VacancyModel.source_id == "0"))
        assert expired.availability == "expired" and not expired.archived
        assert not is_fresh(await VacancyRepository(session).get_by_id(expired.id), now=now)
        checked = await session.scalar(select(VacancyModel).where(VacancyModel.source_id == "1"))
        assert checked.last_seen_at.replace(tzinfo=UTC) == vacancy.collected_at
        assert checked.last_checked_at.replace(tzinfo=UTC) == now


async def test_changed_card_invalidates_analysis_and_supersedes_old_task(db_sessions, vacancy):
    from domain.vacancy_analysis import VacancyAnalysis, VacancyExtraction
    from storage.models import BackgroundJob

    sessions = db_sessions
    vacancy = vacancy.model_copy(
        update={"published_at": datetime.now(UTC) - timedelta(days=20), "valid_until": None}
    )
    source = SimpleNamespace(search=AsyncMock(return_value=[vacancy]))
    recipe = "a" * 64
    await collect_vacancies(source, sessions, SearchQuery(text="first"), analysis_pipeline=recipe)
    async with sessions.begin() as session:
        repo = VacancyRepository(session)
        original = await repo.get_by_source(vacancy.source, vacancy.source_id)
        old_id = original.id
        await repo.set_analysis(
            old_id,
            VacancyAnalysis(
                model="unit-test",
                model_digest="fake-test-digest",
                input_hash=analysis_input_hash(vacancy),
                extraction=VacancyExtraction(
                    category="other",
                    specialization=None,
                    role_evidence=vacancy.title,
                    responsibilities=[],
                    skills=[],
                    alternatives=[],
                    experience_min_years=None,
                    experience_evidence=None,
                ),
            ),
        )
    await collect_vacancies(source, sessions, SearchQuery(text="second"), analysis_pipeline=recipe)
    async with sessions() as session:
        assert (await VacancyRepository(session).get_by_id(old_id)).analysis is not None
        assert await session.scalar(select(func.count()).select_from(BackgroundJob)) == 1
    changed = vacancy.model_copy(
        update={"description": "Новые требования", "raw_payload": {"changed": True}}
    )
    source.search.return_value = [changed]
    await collect_vacancies(source, sessions, SearchQuery(text="third"), analysis_pipeline=recipe)
    async with sessions() as session:
        updated = await VacancyRepository(session).get_by_id(old_id)
        assert updated.analysis is None and updated.description == "Новые требования"
        assert (await session.get(VacancyModel, old_id)).revision_number == 2
        assert sorted(await session.scalars(select(BackgroundJob.state))) == [
            "pending",
            "superseded",
        ]


async def test_normal_ranking_excludes_closed_and_expired_but_keeps_uncertainty(
    db_sessions, vacancy, profile
):
    sessions = db_sessions
    now = datetime.now(UTC)
    for number in range(3):
        await save(
            sessions,
            vacancy.model_copy(
                update={
                    "id": uuid4(),
                    "source_id": str(number),
                    "published_at": now - timedelta(days=100),
                    "collected_at": now - timedelta(days=3),
                    "valid_until": now - timedelta(hours=1) if number == 2 else None,
                }
            ),
        )

    async def fetch(source_id):
        if source_id == "0":
            response = httpx.Response(
                410, request=httpx.Request("GET", "https://example.org/jobs/0")
            )
            raise httpx.HTTPStatusError("gone", request=response.request, response=response)
        raise requests.Timeout()

    await check_catalog(SimpleNamespace(get_vacancy=fetch), sessions, now=now)
    async with sessions() as session:
        matches = await rank_recent(VacancyRepository(session), profile, include_ineligible=True)
        assert [v.source_id for v, _ in matches] == ["1"]


async def test_block_stops_rechecks_before_next_card(db_sessions, vacancy):
    sessions = db_sessions
    for number in range(3):
        await save(sessions, vacancy.model_copy(update={"id": uuid4(), "source_id": str(number)}))
    source = SimpleNamespace(get_vacancy=AsyncMock(side_effect=CollectionStopped("captcha")))
    with pytest.raises(CollectionStopped):
        await check_catalog(source, sessions, limit=3, now=vacancy.collected_at + timedelta(days=3))
    assert source.get_vacancy.await_count == 1


async def test_expiration_metadata_has_bounded_writes_without_http(db_sessions, vacancy):
    sessions = db_sessions
    observed = vacancy.collected_at
    now = observed + timedelta(days=2)
    async with sessions.begin() as session:
        for number in range(101):
            await VacancyRepository(session).save(
                vacancy.model_copy(
                    update={
                        "id": uuid4(),
                        "source_id": str(number),
                        "valid_until": now - timedelta(hours=1),
                    }
                )
            )
    source = SimpleNamespace(get_vacancy=AsyncMock())
    await check_catalog(source, sessions, now=now)
    source.get_vacancy.assert_not_awaited()
    async with sessions() as session:
        count = await session.scalar(
            select(func.count())
            .select_from(VacancyModel)
            .where(VacancyModel.availability == "expired")
        )
        assert count == 100


@pytest.mark.postgres
async def test_catalog_check_row_lock_and_simultaneous_snapshot_do_not_deadlock(
    postgres_sessions, vacancy
):
    import asyncio

    from sqlalchemy import event

    from domain.vacancy_duplicates import cross_source_key

    card = vacancy.model_copy(
        update={"description": "Поддержка систем, разработка и тестирование. " * 4}
    )
    assert cross_source_key(card.model_dump())
    async with postgres_sessions.begin() as session:
        await VacancyRepository(session).save(card)
    attempted_lock = asyncio.Event()

    def observe(conn, cursor, statement, parameters, context, many):
        if "FOR UPDATE" in statement and "vacancies.source_id =" in statement:
            attempted_lock.set()

    async def replay():
        async with postgres_sessions.begin() as session:
            old = card.model_copy(update={"raw_payload": {"html": "old publication"}})
            await VacancyRepository(session).save(old, seen_in_search=False, verified=False)

    engine = postgres_sessions.kw["bind"].sync_engine
    async with asyncio.timeout(8):
        async with postgres_sessions.begin() as online:
            await online.get(VacancyModel, card.id, with_for_update=True)
            event.listen(engine, "before_cursor_execute", observe)
            task = asyncio.create_task(replay())
            try:
                await attempted_lock.wait()
                await VacancyRepository(online).save(card.model_copy(update={"archived": True}))
            finally:
                event.remove(engine, "before_cursor_execute", observe)
        await task
    async with postgres_sessions() as session:
        assert (await session.get(VacancyModel, card.id)).availability == "closed"


async def test_custom_check_interval_drives_display_and_refresh(db_sessions, vacancy, profile):
    from bot.cards import format_card
    from jobs.catalog_freshness import available_for_delivery
    from scoring import score_vacancy

    now = datetime.now(UTC)
    old = vacancy.model_copy(
        update={
            "collected_at": now - timedelta(days=2),
            "published_at": now - timedelta(days=3),
            "valid_until": None,
        }
    )
    async with db_sessions.begin() as session:
        repo = VacancyRepository(session)
        await repo.save(old)
        card = await repo.get_by_id(old.id)
        overdue = set()
        interval = timedelta(days=1)
        assert await available_for_delivery(
            session, [card.id], now=now, interval=interval, overdue=overdue
        ) == {card.id}
        assert overdue == {card.id}
        assert (await session.get(VacancyModel, card.id)).check_requested_at is not None
        match = score_vacancy(card, profile, now=now)
        assert "Последняя проверка" in format_card(card, match, check_interval=interval).render()[0]
        assert (
            "Последняя проверка"
            not in format_card(card, match, check_interval=3 * interval).render()[0]
        )
