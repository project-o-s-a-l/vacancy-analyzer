"""behavioral bounds and PostgreSQL concurrency; SQL is never replaced by a mock."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from sqlalchemy import event, update
from test_candidates import bulk, candidate, job

from domain import CandidateProfile
from domain.retrieval import RetrievalParameters
from jobs.candidates import retrieve_candidates
from jobs.rank import rank_recent
from jobs.vacancy_backfill import backfill_vacancies
from scoring.retrieval import lexical_evidence, profile_query, search_document
from storage.models import VacancyModel as V
from storage.repositories import VacancyRepository
from storage.repositories.candidate_repository import CandidateRepository


@pytest.mark.postgres
async def test_details_bounded_no_catalog_count_and_sql_rank(
    postgres_sessions, vacancy, monkeypatch
):
    cards = [job(vacancy, company=f"Employer {n}") for n in range(300)]
    await bulk(postgres_sessions, cards)
    sql = []
    engine = postgres_sessions.kw["bind"].sync_engine

    def capture(conn, cursor, statement, parameters, context, many):
        sql.append(statement)

    event.listen(engine, "before_cursor_execute", capture)
    monkeypatch.setattr(
        CandidateRepository,
        "diagnostic_counts",
        AsyncMock(side_effect=AssertionError("full catalog COUNT")),
    )
    try:
        async with postgres_sessions() as session:
            stats = {}
            ranked = await rank_recent(
                VacancyRepository(session), candidate(), limit=5, diagnostics=stats
            )
        assert len(ranked) == 5
        assert stats["detail_count"] <= stats["pool_limit"] + stats["repair_limit"]
        assert stats["detail_count"] == 5 and stats["scanned_count"] == 5
        assert stats["repair_count"] == 0 and stats["catalog_counts_sampled"] == 0
        shortlist = next(s for s in sql if "va_lexical_rank_v2" in s)
        assert "LIMIT" in shortlist and "row_number() OVER" in shortlist
        assert "vacancies.description" not in shortlist and "professional_facts" not in shortlist
        assert not any("count(*) AS count_1" in s for s in sql)
    finally:
        event.remove(engine, "before_cursor_execute", capture)


@pytest.mark.postgres
@pytest.mark.parametrize("invalid_count,found", [(3, True), (5, False)])
async def test_residual_refill_or_explicit_budget(postgres_sessions, vacancy, invalid_count, found):
    now = datetime.now(UTC)
    bad = [
        job(vacancy, company=f"bad-{n}", experience="6+", published_at=now - timedelta(hours=1))
        for n in range(invalid_count)
    ]
    good = job(vacancy, company="good", experience=None, published_at=now - timedelta(days=200))
    await bulk(postgres_sessions, [*bad, good])
    stats = {}
    async with postgres_sessions() as s:
        chosen = await retrieve_candidates(
            VacancyRepository(s),
            candidate(experience_years=1),
            parameters=RetrievalParameters(k=1, batch_size=2, pool_factor=4),
            diagnostics=stats,
        )
    assert bool(chosen.items) is found
    if found:
        assert chosen.items[0].id == good.id
    assert stats["pool_budget_exhausted"] == int(not found)
    assert stats["detail_count"] <= 4 and stats["required_excluded"] == min(invalid_count, 4)


@pytest.mark.postgres
@pytest.mark.parametrize("mutation", ["archive", "revision", "delete", "history_group"])
async def test_read_committed_changes_refill(postgres_sessions, vacancy, mutation):
    first = job(vacancy, company="first", published_at=datetime.now(UTC) - timedelta(hours=1))
    second = job(vacancy, company="second", published_at=datetime.now(UTC) - timedelta(days=200))
    await bulk(postgres_sessions, [first, second])
    blocked_key = "f" * 64

    class Racing(CandidateRepository):
        changed = False

        async def load_selected(self, ids):
            if ids and not self.changed:
                self.changed = True
                async with postgres_sessions.begin() as writer:
                    from sqlalchemy import delete

                    values = (
                        {"archived": True}
                        if mutation == "archive"
                        else {"content_revision": "new"}
                        if mutation == "revision"
                        else {"duplicate_key": blocked_key}
                    )
                    await writer.execute(
                        delete(V).where(V.id == first.id)
                        if mutation == "delete"
                        else update(V).where(V.id == first.id).values(**values)
                    )
            return await super().load_selected(ids)

    stats = {}
    async with postgres_sessions() as s:
        result = await retrieve_candidates(
            VacancyRepository(s),
            candidate(),
            parameters=RetrievalParameters(k=1, batch_size=1),
            candidate_source=Racing(s),
            excluded_keys={blocked_key},
            diagnostics=stats,
        )
    assert [v.id for v in result.items] == [second.id]
    assert stats["recheck_excluded"] == 1 and stats["pool_budget_exhausted"] == 0


@pytest.mark.postgres
async def test_stale_budget_and_resumable_rebuild(postgres_sessions, vacancy):
    cards = [job(vacancy, company=f"stale-{n}") for n in range(7)]
    await bulk(postgres_sessions, cards)
    async with postgres_sessions.begin() as s:
        await s.execute(update(V).values(projection_contract="old"))
    stats = {}
    async with postgres_sessions() as s:
        await retrieve_candidates(
            VacancyRepository(s),
            candidate(),
            parameters=RetrievalParameters(k=10, repair_budget=2),
            diagnostics=stats,
        )
    assert stats["repair_count"] == 2 and stats["projection_budget_exhausted"] == 1
    assert stats["source_exhausted"] == 0
    first = await backfill_vacancies(postgres_sessions, batch_size=2)
    second = await backfill_vacancies(postgres_sessions, batch_size=2)
    assert first["cursor"]["after"] != second["cursor"]["after"]
    for _ in range(10):
        if (await backfill_vacancies(postgres_sessions, batch_size=2))["done"]:
            break
    stats = {}
    async with postgres_sessions() as s:
        result = await retrieve_candidates(VacancyRepository(s), candidate(), diagnostics=stats)
    assert (
        len(result.items) == 7
        and stats["repair_count"] == stats["projection_budget_exhausted"] == 0
    )


@pytest.mark.postgres
async def test_large_history_uses_array_bind_before_limit(postgres_sessions, vacancy):
    first, other = job(vacancy, company="blocked"), job(vacancy, company="available")
    await bulk(postgres_sessions, [first, other])
    excluded = {first.id, *[uuid4() for _ in range(10000)]}
    async with postgres_sessions() as s:
        source = CandidateRepository(s)
        result = await retrieve_candidates(
            VacancyRepository(s),
            candidate(),
            parameters=RetrievalParameters(k=1),
            excluded_ids=excluded,
        )
        stmt = source.shortlist_statement(
            candidate(),
            profile_query(candidate()),
            RetrievalParameters(),
            limit=5,
            positive=True,
            now=datetime.now(UTC),
            since=None,
            category=None,
            include_ineligible=False,
            excluded_ids=excluded,
            excluded_keys=(),
        )
        sql = str(stmt.compile(dialect=s.get_bind().dialect))
    assert [v.id for v in result.items] == [other.id]
    assert "ANY" in sql and len(sql) < 10000


def test_broad_profile_and_multiple_intents_do_not_dilute_exact_match():
    document = search_document(
        {"title": "Java Developer", "skills": ["Java", "Spring"], "description": "Java Spring"}
    )

    narrow = candidate()
    broad = candidate(
        skills={"java": {}, "spring": {}, **{f"other{i}": {} for i in range(100)}},
        preferences={"positions": ["Java Developer", "Главный бухгалтер"]},
    )
    params = RetrievalParameters()
    assert (
        lexical_evidence(document, profile_query(broad), params).lexical_score
        == lexical_evidence(document, profile_query(narrow), params).lexical_score
    )


@pytest.mark.postgres
async def test_notification_history_above_postgres_bind_limit(
    postgres_sessions, vacancy, monkeypatch
):
    from jobs.notify import NotificationService
    from storage.repositories.notification_repository import NotificationRepository

    first, other = job(vacancy, company="blocked"), job(vacancy, company="available")
    await bulk(postgres_sessions, [first, other])
    blocked = {first.id, *[uuid4() for _ in range(40000)]}
    monkeypatch.setattr(NotificationRepository, "blocked_ids", AsyncMock(return_value=blocked))
    result = await NotificationService(
        postgres_sessions, candidate(), chat_id=71, min_score=0
    ).pending()
    assert [v.id for v, _ in result] == [other.id]


@pytest.mark.postgres
async def test_sparse_cards_without_duplicate_key_cross_batches(postgres_sessions, vacancy):
    cards = [
        job(vacancy, company=f"sparse-{i}", description=None, description_html=None)
        for i in range(35)
    ]
    await bulk(postgres_sessions, cards)
    async with postgres_sessions() as session:
        result = await retrieve_candidates(
            VacancyRepository(session),
            candidate(),
            parameters=RetrievalParameters(k=35, batch_size=2),
        )
    assert len(result.items) == 35


@pytest.mark.postgres
async def test_required_location_before_pool_remote_exception(postgres_sessions, vacancy):
    cards = [job(vacancy, company=f"bad-{n}", location="Казань", remote=False) for n in range(50)]
    good = job(
        vacancy,
        company="old",
        location="г. Москва",
        remote=False,
        published_at=datetime.now(UTC) - timedelta(days=250),
    )
    remote = job(vacancy, company="remote", location=None, remote=True)
    await bulk(postgres_sessions, [*cards, good, remote])
    profile = CandidateProfile(preferences={"locations": ["Москва"], "locations_required": True})
    stats = {}
    async with postgres_sessions() as s:
        result = await retrieve_candidates(
            VacancyRepository(s),
            profile,
            parameters=RetrievalParameters(k=2, batch_size=1, pool_factor=1),
            diagnostics=stats,
        )
    assert {v.id for v in result.items} == {good.id, remote.id}
    assert stats["required_excluded"] == 0 and stats["detail_count"] == 2
