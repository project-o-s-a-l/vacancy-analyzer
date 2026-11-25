import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import event, func, select

from collectors.base import SearchCard, SearchPage, SearchQuery
from jobs.catalog_checks import check_catalog
from jobs.catalog_freshness import available_for_delivery
from jobs.catalog_metrics import catalog_status
from jobs.collection_budget import BudgetDeferred, BudgetedSource, aware
from jobs.collection_progress import (
    choose_direction,
    crawl_slice,
    direction_key,
    sync_directions,
)
from jobs.collection_scheduler import collection_loop
from storage.models import (
    BackgroundJob,
    CollectionBudget,
    CollectionDirection,
    CollectionObservation,
    VacancyModel,
)
from storage.repositories import VacancyRepository

NOW = datetime(2026, 11, 6, 12, tzinfo=UTC)


class Replay:
    paging_kind = "replay-v1"
    depth_limit = 2000

    def __init__(self, vacancy, pages=None):
        self.vacancy = vacancy
        self.pages = pages or [["701", "702"], ["702", "703"], ["704"]]
        self.calls, self.details = [], []
        self.fail = set()
        self.fingerprint = None

    async def search_page(self, query, page):
        self.calls.append(page)
        return SearchPage(
            cards=[
                SearchCard(source_id=i, published_at=NOW, fingerprint=self.fingerprint)
                for i in self.pages[page]
            ],
            has_more=page < len(self.pages) - 1,
            total=sum(map(len, self.pages)),
        )

    async def get_vacancy(self, source_id):
        self.details.append(source_id)
        if source_id in self.fail:
            raise httpx.ReadTimeout("network")
        return self.vacancy.model_copy(
            update={"id": uuid4(), "source_id": source_id, "collected_at": NOW}
        )


async def test_restart_overlap_equal_times_dedupe_and_jobs(db_sessions, vacancy):
    sessions = db_sessions
    source, query = Replay(vacancy), SearchQuery(text="synthetic", per_page=2)
    await sync_directions(source, sessions, [query], now=NOW)
    for minute in range(5):
        # No process-local cursor; new source objects can continue the same DB state.
        await crawl_slice(
            source,
            sessions,
            query,
            now=NOW + timedelta(minutes=minute),
            analysis_pipeline="f07-test",
        )
    assert source.calls == [0, 0, 1, 1, 2]
    assert source.details == ["701", "702", "703", "704"]
    async with sessions() as session:
        assert await session.scalar(select(func.count()).select_from(VacancyModel)) == 4
        assert await session.scalar(select(func.count()).select_from(BackgroundJob)) == 4
        row = await session.get(CollectionDirection, direction_key(source, query))
        assert row.progress["initial_done"] and row.progress["cycles"] == 1
    # Another sweep within detail TTL does not fetch duplicates or enqueue new work.
    for minute in range(65, 70):
        await crawl_slice(
            Replay(vacancy),
            sessions,
            query,
            now=NOW + timedelta(minutes=minute),
            analysis_pipeline="f07-test",
        )
    report = await catalog_status(sessions, now=NOW + timedelta(minutes=70))
    assert report["directions"][0]["active_cards"] == 4
    assert report["directions"][0]["repeat_ratio"] > 0
    assert report["market_complete"] is False


async def test_page_committed_before_crash_details_resume(db_sessions, vacancy, monkeypatch):
    import jobs.collection_progress as progress

    source, query = Replay(vacancy), SearchQuery(text="safe", per_page=2)
    await sync_directions(source, db_sessions, [query], now=NOW)
    original = progress.drain_details
    monkeypatch.setattr(progress, "drain_details", AsyncMock(side_effect=RuntimeError("crash")))
    with pytest.raises(RuntimeError):
        await crawl_slice(source, db_sessions, query, now=NOW)
    async with db_sessions() as session:
        assert (await session.get(CollectionDirection, direction_key(source, query))).progress[
            "page"
        ] == 1
        assert await session.scalar(select(func.count()).select_from(CollectionObservation)) == 2
        assert await session.scalar(select(func.count()).select_from(VacancyModel)) == 0
    monkeypatch.setattr(progress, "drain_details", original)
    await crawl_slice(source, db_sessions, query, now=NOW + timedelta(minutes=1))
    assert set(source.details) == {"701", "702"}


async def test_failed_commit_rolls_back_page_and_inbox(db_sessions, vacancy):
    source, query = Replay(vacancy), SearchQuery(text="atomic", per_page=2)
    await sync_directions(source, db_sessions, [query], now=NOW)
    from sqlalchemy.orm import Session

    def fail(session):
        if any(isinstance(item, CollectionObservation) for item in session.new):
            raise RuntimeError("crash before commit")

    def fail_flush(session, context, instances):
        fail(session)

    event.listen(Session, "before_flush", fail_flush)
    try:
        with pytest.raises(RuntimeError):
            await crawl_slice(source, db_sessions, query, now=NOW)
    finally:
        event.remove(Session, "before_flush", fail_flush)
    async with db_sessions() as session:
        assert (await session.get(CollectionDirection, direction_key(source, query))).progress[
            "page"
        ] == 0
        assert await session.scalar(select(func.count()).select_from(CollectionObservation)) == 0
    await crawl_slice(source, db_sessions, query, now=NOW + timedelta(minutes=1))
    assert source.calls == [0, 0]


async def test_failed_detail_does_not_rewind_or_block_other_pages(db_sessions, vacancy):
    source, query = Replay(vacancy), SearchQuery(text="errors", per_page=2)
    source.fail = {"701"}
    await sync_directions(source, db_sessions, [query], now=NOW)
    for minute in range(5):
        await crawl_slice(source, db_sessions, query, now=NOW + timedelta(minutes=minute))
    async with db_sessions() as session:
        assert await session.scalar(select(func.count()).select_from(VacancyModel)) == 3
        item = await session.get(CollectionObservation, (direction_key(source, query), "701"))
        assert item.pending and item.attempts == 2
    source.fail.clear()
    await crawl_slice(source, db_sessions, query, now=NOW + timedelta(minutes=10))
    async with db_sessions() as session:
        assert await session.scalar(select(func.count()).select_from(VacancyModel)) == 4


async def test_changed_snippet_waits_three_days_without_disappearance_closure(db_sessions, vacancy):
    source, query = Replay(vacancy, [["701", "702"]]), SearchQuery(text="changes")
    source.fingerprint = "a" * 64
    await sync_directions(source, db_sessions, [query], now=NOW)
    await crawl_slice(source, db_sessions, query, now=NOW)
    source.fingerprint = "b" * 64
    source.vacancy = vacancy.model_copy(
        update={"description": "New tasks", "raw_payload": {"v": 2}}
    )
    source.pages = [["701"]]
    await crawl_slice(source, db_sessions, query, now=NOW + timedelta(hours=1))
    assert source.details == ["701", "702"]
    await crawl_slice(source, db_sessions, query, now=NOW + timedelta(hours=71))
    assert source.details == ["701", "702"]
    await crawl_slice(source, db_sessions, query, now=NOW + timedelta(hours=72))
    assert source.details == ["701", "702", "701"]
    async with db_sessions() as session:
        gone = await session.scalar(select(VacancyModel).where(VacancyModel.source_id == "702"))
        assert not gone.archived  # Search absence is never closure.
        changed = await session.scalar(select(VacancyModel).where(VacancyModel.source_id == "701"))
        assert changed.revision_number == 2
        assert changed.facts_revision == changed.projection_facts_revision


async def test_directions_removed_readded_sort_identity_depth_and_starvation(db_sessions, vacancy):
    source = Replay(vacancy, [["1", "2"], ["3", "4"], ["5"]])
    source.depth_limit = 4
    a, b = SearchQuery(text="popular", per_page=2), SearchQuery(text="rare", per_page=2)
    keys = await sync_directions(source, db_sessions, [a, b], demand={"popular": 100}, now=NOW)
    assert await choose_direction(db_sessions, keys, now=NOW) == direction_key(source, a)
    await crawl_slice(source, db_sessions, a, now=NOW)
    await crawl_slice(source, db_sessions, a, now=NOW + timedelta(minutes=1))
    await crawl_slice(source, db_sessions, a, now=NOW + timedelta(minutes=2))
    assert await choose_direction(db_sessions, keys, now=NOW + timedelta(hours=2)) == direction_key(
        source, b
    )
    await sync_directions(source, db_sessions, [b], now=NOW)
    async with db_sessions() as session:
        row = await session.get(CollectionDirection, direction_key(source, a))
        assert not row.active and row.progress["depth_limited"]
    await sync_directions(source, db_sessions, [a, b], now=NOW)
    source.paging_kind = "different-sort-v2"
    changed = await sync_directions(source, db_sessions, [a], now=NOW)
    assert set(changed).isdisjoint(keys)


async def test_new_only_skips_known_backlog_and_leaves_it_for_normal_collection(
    db_sessions, vacancy, monkeypatch
):
    known_ids = [str(i) for i in range(701, 711)]
    source = Replay(vacancy, [known_ids[:5], known_ids[5:]])
    query = SearchQuery(text="discover", per_page=5)
    now = datetime.now(UTC)
    old = now - timedelta(days=4)
    await sync_directions(source, db_sessions, [query], now=old)
    for minute in range(3):
        await crawl_slice(source, db_sessions, query, now=old + timedelta(minutes=minute))
    # A previous normal sweep filled the entire backpressure allowance with old cards.
    for minute in range(3):
        await crawl_slice(
            source, db_sessions, query, now=now - timedelta(minutes=3 - minute), detail_limit=0
        )
    key = direction_key(source, query)
    async with db_sessions() as session:
        assert all(o.pending for o in await session.scalars(select(CollectionObservation)))
        original = await session.scalar(select(VacancyModel).where(VacancyModel.source_id == "701"))
    source.pages = [["701", "702", "711"]]
    source.fingerprint = "c" * 64
    source.details.clear()
    await collection_loop(
        source,
        db_sessions,
        query,
        False,
        3600,
        query_texts=[query.text],
        new_only=True,
        analysis_pipeline="new-only-test",
    )
    assert source.details == ["711"]
    async with db_sessions() as session:
        row = await session.get(VacancyModel, original.id)
        assert (
            row.last_checked_at,
            row.last_seen_at,
            row.revision_number,
            row.search_fingerprint,
        ) == (
            original.last_checked_at,
            original.last_seen_at,
            original.revision_number,
            original.search_fingerprint,
        )
        assert (await session.get(CollectionObservation, (key, "701"))).pending
        assert await session.scalar(select(func.count()).select_from(BackgroundJob)) == 1
    report = await catalog_status(db_sessions)
    assert report["directions"][0]["pending_details"] == len(known_ids)
    assert report["directions"][0]["pending_new_details"] == 0

    checks = AsyncMock()
    monkeypatch.setattr("jobs.collection_scheduler.check_catalog", checks)

    async def stop(_):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await collection_loop(
            source,
            db_sessions,
            query,
            True,
            3600,
            query_texts=[query.text],
            check_limit=20,
            new_only=True,
            sleep=stop,
        )
    checks.assert_not_awaited()
    await crawl_slice(source, db_sessions, query, now=now + timedelta(hours=2))
    assert set(source.details[1:]) == set(known_ids)


async def test_budget_persists_failures_and_honors_retry_after(db_sessions, vacancy):
    now = [NOW]
    source = Replay(vacancy)
    response = httpx.Response(
        429,
        headers={"Retry-After": "7200"},
        request=httpx.Request("GET", "https://example.org/jobs"),
    )
    source.get_vacancy = AsyncMock(
        side_effect=httpx.HTTPStatusError("limit", request=response.request, response=response)
    )

    def wrap():
        return BudgetedSource(
            source, db_sessions, detail_limit=1, delay=0, now=lambda: now[0], sleep=AsyncMock()
        )

    with pytest.raises(httpx.HTTPStatusError):
        await wrap().get_vacancy("1")
    now[0] += timedelta(hours=1)
    with pytest.raises(BudgetDeferred, match="cooldown"):
        await wrap().get_vacancy("1")
    assert source.get_vacancy.await_count == 1
    now[0] += timedelta(hours=1)
    source.get_vacancy.side_effect = None
    source.get_vacancy.return_value = vacancy
    await wrap().get_vacancy("1")
    with pytest.raises(BudgetDeferred, match="budget"):
        await wrap().get_vacancy("1")


async def test_superjob_paces_requests_across_its_hourly_target(db_sessions, vacancy):
    source = Replay(vacancy)
    source.source_name = "superjob"
    now = NOW

    async def sleep(seconds):
        nonlocal now
        now += timedelta(seconds=seconds)

    async with db_sessions.begin() as session:
        session.add(
            CollectionBudget(
                source="superjob",
                window_at=NOW - timedelta(minutes=59),
                search_used=60,
                detail_used=700,
                next_request_at=NOW,
                blocked=False,
                metrics={},
            )
        )
    budget = BudgetedSource(
        source,
        db_sessions,
        search_limit=None,
        detail_limit=None,
        delay=2,
        request_intervals={"search": 5, "detail": 3600 / 700},
        now=lambda: now,
        sleep=sleep,
    )
    await budget.get_vacancy("701")
    query = SearchQuery(text="synthetic")
    await budget.search_page(query, 0)
    assert now == NOW + timedelta(seconds=2)
    await budget.get_vacancy("702")
    assert now == NOW + timedelta(seconds=3600 / 700)
    await budget.search_page(query, 0)
    assert now == NOW + timedelta(seconds=3600 / 700 + 2)
    async with db_sessions() as session:
        row = await session.get(CollectionBudget, "superjob")
        assert (row.search_used, row.detail_used) == (62, 702)


async def test_fast_slice_reports_end_with_pending_details(db_sessions, vacancy, caplog):
    source = Replay(vacancy, [["701", "702"]])
    query = SearchQuery(text="end")
    keys = await sync_directions(source, db_sessions, [query], now=NOW)
    options = {"detail_limit": 1, "new_only": True, "active_slice_interval": 1}
    with caplog.at_level("INFO", logger="jobs.collection_progress"):
        await crawl_slice(source, db_sessions, query, now=NOW, **options)
    sweep = next(r for r in caplog.records if r.event_code == "collection.sweep_finished")
    assert sweep.event_fields["pending_count"] == 1
    assert sweep.event_fields["total"] == 2

    next_slice = NOW + timedelta(seconds=1)
    assert await choose_direction(db_sessions, keys, now=next_slice) == next(iter(keys))
    assert await crawl_slice(source, db_sessions, query, now=next_slice, **options) == 1
    assert source.calls == [0]
    report = await catalog_status(db_sessions, now=next_slice)
    assert report["directions"][0]["pending_new_details"] == 0
    assert await choose_direction(db_sessions, keys, now=next_slice) is None


async def test_final_freshness_network_failure_and_priority_are_bounded(db_sessions, vacancy):
    old = vacancy.model_copy(update={"collected_at": NOW - timedelta(days=3)})
    async with db_sessions.begin() as session:
        vid = await VacancyRepository(session).save(old)
        assert await available_for_delivery(session, [vid], now=NOW) == {vid}
    source = SimpleNamespace(get_vacancy=AsyncMock(side_effect=httpx.ReadTimeout("offline")))
    await check_catalog(source, db_sessions, now=NOW)
    async with db_sessions.begin() as session:
        row = await session.get(VacancyModel, vid)
        assert aware(row.last_checked_at) == old.collected_at
        assert aware(row.last_check_attempt_at) == NOW
        assert not row.archived and row.check_requested_at
        assert await available_for_delivery(session, [vid], now=NOW) == {vid}
    await check_catalog(source, db_sessions, now=NOW + timedelta(seconds=10))
    assert source.get_vacancy.await_count == 1
    source.get_vacancy.side_effect = None
    source.get_vacancy.return_value = vacancy
    await check_catalog(source, db_sessions, now=NOW + timedelta(hours=1))
    async with db_sessions.begin() as session:
        assert await available_for_delivery(session, [vid], now=NOW + timedelta(hours=1)) == {vid}


async def test_old_schedule_and_delivery_request_wait_for_three_day_boundary(sessions, vacancy):
    async with sessions.begin() as session:
        vid = await VacancyRepository(session).save(
            vacancy.model_copy(update={"collected_at": NOW})
        )
        row = await session.get(VacancyModel, vid)
        row.next_check_at = NOW + timedelta(hours=6)
        row.check_requested_at = NOW  # Persisted by an older release.
    source = SimpleNamespace(get_vacancy=AsyncMock(return_value=vacancy))
    due = NOW + timedelta(days=3)
    assert await check_catalog(source, sessions, now=due - timedelta(seconds=1)) == 0
    assert await check_catalog(source, sessions, now=due) == 1
    source.get_vacancy.assert_awaited_once()


async def test_runtime_calls_persisted_cursor_on_repeat(sessions, vacancy):
    source = Replay(vacancy)
    query = SearchQuery(text="placeholder", per_page=2)
    for _ in range(3):
        await collection_loop(source, sessions, query, False, 3600, query_texts=["Python"])
    assert source.calls == [0, 0, 1]
    assert set(source.details) == {"701", "702", "703"}


async def test_offline_source_never_claims_live_freshness(db_sessions, vacancy):
    async with db_sessions.begin() as session:
        repo = VacancyRepository(session)
        vid = await repo.save(vacancy, verified=False)
        assert (await repo.get_by_id(vid)).last_checked_at is None
        await repo.save(vacancy.model_copy(update={"archived": True}))
        await repo.save(vacancy, verified=False)
        archived = vacancy.model_copy(
            update={"description": "Changed offline", "raw_payload": {"html": "old"}}
        )
        await repo.save(archived, seen_in_search=False, verified=False)
        current = await repo.get_by_id(vid)
        assert current.archived and current.last_checked_at == vacancy.collected_at
        assert current.description == vacancy.description
        assert len(await repo.history(vid)) == 2


async def test_overlap_recovers_shifted_boundary_and_removed_direction_stops(db_sessions, vacancy):
    source, query = (
        Replay(vacancy, [["1", "2"], ["3", "4"]]),
        SearchQuery(text="moving", per_page=2),
    )
    await sync_directions(source, db_sessions, [query], now=NOW)
    await crawl_slice(source, db_sessions, query, now=NOW)
    source.pages = [["9", "1"], ["2", "3"], ["4"]]
    for minute in range(1, 5):
        await crawl_slice(source, db_sessions, query, now=NOW + timedelta(minutes=minute))
    assert set(source.details) == {"1", "2", "3", "4", "9"}
    await sync_directions(source, db_sessions, [], now=NOW)
    count = len(source.calls)
    await crawl_slice(source, db_sessions, query, now=NOW + timedelta(hours=2))
    assert len(source.calls) == count


async def test_budget_block_and_cancel_survive_restart(db_sessions, vacancy):
    source = Replay(vacancy)
    response = httpx.Response(403, request=httpx.Request("GET", "https://example.org/jobs"))
    source.get_vacancy = AsyncMock(
        side_effect=httpx.HTTPStatusError("forbidden", request=response.request, response=response)
    )
    budget = BudgetedSource(source, db_sessions, delay=0, sleep=AsyncMock(), now=lambda: NOW)
    with pytest.raises(httpx.HTTPStatusError):
        await budget.get_vacancy("1")
    with pytest.raises(BudgetDeferred, match="blocked"):
        await BudgetedSource(source, db_sessions, delay=0, now=lambda: NOW).get_vacancy("1")
    from jobs.collection_budget import resume_source

    await resume_source(db_sessions)
    source.get_vacancy.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await budget.get_vacancy("1")
    from storage.models import CollectionBudget

    async with db_sessions() as session:
        row = await session.get(CollectionBudget, "hh")
        assert row.detail_used == 2


async def test_source_job_and_inbox_commit_together(db_sessions, vacancy, monkeypatch):
    import jobs.collection_progress as progress

    source, query = Replay(vacancy, [["701"]]), SearchQuery(text="atomic detail")
    await sync_directions(source, db_sessions, [query], now=NOW)
    original = progress.schedule_collected
    monkeypatch.setattr(
        progress, "schedule_collected", AsyncMock(side_effect=RuntimeError("job failed"))
    )
    with pytest.raises(RuntimeError):
        await crawl_slice(source, db_sessions, query, now=NOW, analysis_pipeline="test")
    async with db_sessions() as session:
        assert await session.scalar(select(func.count()).select_from(VacancyModel)) == 0
        item = await session.get(CollectionObservation, (direction_key(source, query), "701"))
        assert item.pending
    monkeypatch.setattr(progress, "schedule_collected", original)
    await crawl_slice(source, db_sessions, query, now=NOW, analysis_pipeline="test")
    async with db_sessions() as session:
        assert await session.scalar(select(func.count()).select_from(VacancyModel)) == 1
        assert await session.scalar(select(func.count()).select_from(BackgroundJob)) == 1


@pytest.mark.parametrize("verified", [True, False])
async def test_delivery_sends_old_or_unchecked_card_and_requests_check(
    db_sessions, vacancy, profile, verified
):
    from bot.cards import render_card
    from jobs.notify import NotificationService

    old = vacancy.model_copy(update={"collected_at": datetime.now(UTC) - timedelta(days=3)})
    async with db_sessions.begin() as session:
        vid = await VacancyRepository(session).save(old, verified=verified)
    service = NotificationService(db_sessions, profile, chat_id=1, min_score=0)
    sender = SimpleNamespace(send_recommendation=AsyncMock(return_value=101))
    assert (await service.deliver(sender)).sent == 1
    _, sent, match = sender.send_recommendation.await_args.args
    assert sent.id == vid
    label = "Последняя проверка на HH" if verified else "ещё не проверена"
    assert label in render_card(sent, match)
    async with db_sessions() as session:
        row = await session.get(VacancyModel, vid)
        assert row.check_requested_at and row.availability == "active"
        assert (aware(row.last_checked_at) if row.last_checked_at else None) == (
            old.collected_at if verified else None
        )


async def test_final_availability_rechecks_source_state(db_sessions, vacancy):
    async with db_sessions.begin() as session:
        vid = await VacancyRepository(session).save(vacancy)
        row = await session.get(VacancyModel, vid)
        row.availability = "closed"
        assert await available_for_delivery(session, [vid], now=NOW) == set()
        row.availability = "active"
        row.valid_until = NOW
        assert await available_for_delivery(session, [vid], now=NOW) == set()
        row.valid_until = None
        row.archived = True
        assert await available_for_delivery(session, [vid], now=NOW) == set()


@pytest.mark.postgres
async def test_migration_preserves_legacy_source_and_failed_attempt(postgres_sessions):
    from alembic.config import Config
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from alembic import command

    admin = postgres_sessions.kw["bind"]
    schema = "f07_migration_" + uuid4().hex
    engine = create_async_engine(
        admin.url, connect_args={"server_settings": {"search_path": schema}}
    )
    try:
        async with admin.begin() as connection:
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        async with engine.begin() as connection:
            config = Config("alembic.ini")

            def migrate(sync, revision):
                config.attributes["connection"] = sync
                command.upgrade(config, revision)

            await connection.run_sync(migrate, "0017")
            vid = uuid4()
            await connection.execute(
                text("""
                INSERT INTO vacancies (id,source,source_id,url,title,company,skills,published_at,
                  collected_at,last_seen_at,raw_payload,last_checked_at,check_error)
                VALUES (:id,'hh','legacy','https://example.org/jobs/1','original','original','[]',
                  :time,:time,:time,CAST(:payload AS jsonb),:time,'network_error')
            """),
                {"id": vid, "time": NOW, "payload": json.dumps({"retained": True})},
            )
            await connection.run_sync(migrate, "0018")
            row = (
                (
                    await connection.execute(
                        text("""
                SELECT id,title,raw_payload,last_checked_at,last_check_attempt_at FROM vacancies
            """)
                    )
                )
                .mappings()
                .one()
            )
            assert row["id"] == vid and row["title"] == "original"
            assert row["raw_payload"] == {"retained": True}
            assert row["last_checked_at"] is None and row["last_check_attempt_at"] == NOW
    finally:
        await engine.dispose()
        async with admin.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
