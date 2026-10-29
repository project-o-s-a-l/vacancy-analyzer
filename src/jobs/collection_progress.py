"""Restartable page sweeps with overlap, independent head discovery and durable detail inbox."""

import logging
from datetime import UTC, datetime, timedelta

import httpx
import requests
from sqlalchemy import func, select, update

from collectors.errors import CollectionStopped, VacancyParseError
from jobs.analysis_queue import schedule_collected
from jobs.catalog_freshness import check_interval
from jobs.collection_budget import BudgetDeferred, aware
from observability import emit_event
from storage.models import CollectionDirection as Direction
from storage.models import CollectionObservation as Observation
from storage.models import VacancyModel
from storage.payloads import payload_hash
from storage.repositories import VacancyRepository

logger = logging.getLogger(__name__)


def direction_key(source, query):
    values = query.model_dump(exclude={"max_pages"})
    values["text"] = " ".join(query.text.casefold().split())
    return payload_hash({"adapter": source.paging_kind, "query": values, "crawl": "overlap-v1"})


async def sync_directions(source, sessions, queries, *, demand=None, now=None):
    now = now or datetime.now(UTC)
    keys = {direction_key(source, query): query for query in queries}
    async with sessions.begin() as session:
        # An adapter switch or removal suspends work; raw vacancies/history are retained.
        await session.execute(
            update(Direction).where(Direction.key.not_in(keys)).values(active=False)
        )
        for key, query in keys.items():
            row = await session.get(Direction, key)
            if row is None:
                row = Direction(
                    key=key,
                    adapter=source.paging_kind,
                    active=True,
                    demand=1,
                    created_at=now,
                    next_due_at=now,
                    progress={
                        "page": 0,
                        "overlap": False,
                        "initial_done": False,
                        "head_due": now.isoformat(),
                        "cycles": 0,
                    },
                )
                session.add(row)
            if not row.active:
                row.active, row.next_due_at = True, now
            row.demand = max(1, (demand or {}).get(query.text.casefold(), 1))
    return keys


async def choose_direction(sessions, keys, *, now=None):
    now = now or datetime.now(UTC)
    async with sessions() as session:
        rows = list(
            await session.scalars(
                select(Direction).where(
                    Direction.key.in_(keys),
                    Direction.active.is_(True),
                    Direction.next_due_at <= now,
                )
            )
        )

    def priority(row):
        age = (now - aware(row.last_run_at or row.created_at)).total_seconds()
        # A hard aging lane precedes demand. Even rare directions eventually win.
        starved = age >= 7200
        coverage = row.progress.get("unique", 0)
        return (
            not starved,
            -age if starved else 0,
            -(min(row.demand, 10) / (1 + min(coverage, 100) / 20)),
            aware(row.last_run_at or row.created_at),
            row.key,
        )

    return min(rows, key=priority).key if rows else None


async def crawl_slice(
    source, sessions, query, *, analysis_pipeline=None, detail_limit=10, interval=3600, now=None
):
    now = now or datetime.now(UTC)
    key = direction_key(source, query)
    async with sessions.begin() as session:
        row = await session.get(Direction, key, with_for_update=True)
        if row is None or not row.active:
            return 0
        progress = dict(row.progress)
        row.last_run_at = now
        row.next_due_at = now + timedelta(seconds=60)
        pending = await session.scalar(
            select(func.count())
            .select_from(Observation)
            .where(Observation.direction == key, Observation.pending.is_(True))
        )
    # Bounded inbox backpressure. A failed item remains durable without rewinding the page.
    sweep_due = (
        not progress.get("last_sweep_at")
        or progress["page"] > 0
        or datetime.fromisoformat(progress["last_sweep_at"]) + timedelta(seconds=interval) <= now
    )
    if pending < max(2 * query.per_page, detail_limit) and sweep_due:
        head = progress["page"] > 0 and datetime.fromisoformat(progress["head_due"]) <= now
        page = 0 if head else max(0, progress["page"] - int(progress["overlap"]))
        try:
            result = await source.search_page(query, page)
        except BudgetDeferred:
            result = None
        except (requests.RequestException, httpx.HTTPError, CollectionStopped, ValueError) as exc:
            async with sessions.begin() as session:
                row = await session.get(Direction, key, with_for_update=True)
                state = dict(row.progress)
                state["search_errors"] = state.get("search_errors", 0) + 1
                state["error"] = getattr(exc, "code", "search_error")
                row.progress = state
                row.next_due_at = now + timedelta(
                    seconds=min(3600, 60 * 2 ** min(state["search_errors"], 6))
                )
            return 0
        if result is not None:
            async with sessions.begin() as session:
                row = await session.get(Direction, key, with_for_update=True)
                progress = dict(row.progress)
                new = 0
                repeated = len(result.cards) - len({c.source_id for c in result.cards})
                for card in {c.source_id: c for c in result.cards}.values():
                    observation = await session.get(Observation, (key, card.source_id))
                    known = (
                        await session.execute(
                            select(
                                VacancyModel.id,
                                VacancyModel.last_checked_at,
                                VacancyModel.search_fingerprint,
                                VacancyModel.check_error,
                            ).where(
                                VacancyModel.source == "hh",
                                VacancyModel.source_id == card.source_id,
                            )
                        )
                    ).one_or_none()
                    if known:
                        await session.execute(
                            update(VacancyModel)
                            .where(VacancyModel.id == known.id)
                            .values(last_seen_at=now)
                        )
                    repeated += int(known is not None or observation is not None)
                    new += int(known is None and observation is None)
                    # A stable snippet cannot establish unchanged details. TTL is capped at 6h.
                    needs = (
                        not known
                        or known.check_error
                        or not known.last_checked_at
                        or aware(known.last_checked_at) <= now - timedelta(hours=6)
                        or card.fingerprint is not None
                        and card.fingerprint != known.search_fingerprint
                    )
                    if observation is None:
                        observation = Observation(
                            direction=key,
                            source_id=card.source_id,
                            last_seen_at=now,
                            fingerprint=card.fingerprint,
                            pending=bool(needs),
                            attempts=0,
                            retry_at=now,
                        )
                        session.add(observation)
                    else:
                        observation.last_seen_at = now
                        observation.fingerprint = card.fingerprint
                        observation.pending = observation.pending or bool(needs)
                await session.flush()
                progress["unique"] = await session.scalar(
                    select(func.count())
                    .select_from(Observation)
                    .where(Observation.direction == key)
                )
                for name, value in {
                    "search_pages": 1,
                    "new_refs": new,
                    "repeat_refs": repeated,
                    "references": len(result.cards),
                }.items():
                    progress[name] = progress.get(name, 0) + value
                progress.update(
                    total_reported=result.total,
                    last_search_at=now.isoformat(),
                    error=None,
                    search_errors=0,
                )
                cap = source.depth_limit // query.per_page
                if head:
                    progress["head_due"] = (now + timedelta(seconds=interval)).isoformat()
                elif progress["overlap"]:
                    # Overlap observes moving boundaries; it never ends a forward sweep.
                    progress["overlap"] = False
                elif not result.has_more or not result.cards or page + 1 >= cap:
                    progress.update(
                        initial_done=True,
                        page=0,
                        overlap=False,
                        depth_limited=page + 1 >= cap
                        and (
                            result.has_more
                            or result.total is not None
                            and result.total > source.depth_limit
                        ),
                        cycles=progress["cycles"] + 1,
                        last_sweep_at=now.isoformat(),
                    )
                    row.next_due_at = now + timedelta(seconds=interval)
                else:
                    progress["page"] = page + 1
                    # One previous page before every next page, even after a restart.
                    progress["overlap"] = True
                    if page == 0:
                        progress["head_due"] = (now + timedelta(seconds=interval)).isoformat()
                row.progress = progress
    count = await drain_details(
        source, sessions, key, limit=detail_limit, analysis_pipeline=analysis_pipeline, now=now
    )
    async with sessions.begin() as session:
        pending = await session.scalar(
            select(func.count())
            .select_from(Observation)
            .where(Observation.direction == key, Observation.pending.is_(True))
        )
        row = await session.get(Direction, key, with_for_update=True)
        if pending:
            row.next_due_at = now + timedelta(seconds=60)
        elif row.progress.get("last_sweep_at") and row.progress["page"] == 0:
            row.next_due_at = datetime.fromisoformat(row.progress["last_sweep_at"]) + timedelta(
                seconds=interval
            )
    emit_event(logger, "collection.progress", count=count, result="success")
    return count


async def drain_details(source, sessions, key, *, limit, analysis_pipeline, now):
    async with sessions() as session:
        due = list(
            await session.scalars(
                select(Observation)
                .where(
                    Observation.direction == key,
                    Observation.pending.is_(True),
                    Observation.retry_at <= now,
                )
                .order_by(Observation.retry_at, Observation.source_id)
                .limit(limit)
            )
        )
    count = 0
    for observation in due:
        async with sessions() as session:
            known = (
                await session.execute(
                    select(
                        VacancyModel.id,
                        VacancyModel.last_checked_at,
                        VacancyModel.search_fingerprint,
                        VacancyModel.check_error,
                        VacancyModel.last_check_attempt_at,
                    ).where(
                        VacancyModel.source == "hh", VacancyModel.source_id == observation.source_id
                    )
                )
            ).one_or_none()
        if (
            known
            and not known.check_error
            and known.last_checked_at
            and aware(known.last_checked_at) > now - timedelta(hours=6)
            and (
                observation.fingerprint is None
                or observation.fingerprint == known.search_fingerprint
            )
        ):
            async with sessions.begin() as session:
                item = await session.get(Observation, (key, observation.source_id))
                item.pending = False
                direction = await session.get(Direction, key)
                stats = dict(direction.progress)
                stats["details_cached"] = stats.get("details_cached", 0) + 1
                direction.progress = stats
            continue
        state, error, vacancy = None, None, None
        try:
            vacancy = await source.get_vacancy(observation.source_id)
            if vacancy.source != "hh" or vacancy.source_id != observation.source_id:
                raise ValueError("Source identity mismatch")
        except BudgetDeferred:
            break
        except (httpx.HTTPStatusError, requests.HTTPError) as exc:
            status = exc.response.status_code if exc.response is not None else None
            state = "closed" if status in {404, 410} else "unknown"
            error = None if state == "closed" else "http_error"
        except (
            requests.RequestException,
            httpx.RequestError,
            VacancyParseError,
            ValueError,
            CollectionStopped,
        ):
            state, error = "unknown", "detail_error"
        async with sessions.begin() as session:
            item = await session.get(
                Observation, (key, observation.source_id), with_for_update=True
            )
            row = await session.scalar(
                select(VacancyModel)
                .where(VacancyModel.source == "hh", VacancyModel.source_id == observation.source_id)
                .with_for_update()
            )
            # Concurrent source refresh wins. Never overwrite a later observation with old I/O.
            superseded = row and known and row.last_check_attempt_at != known.last_check_attempt_at
            if known and row is None:
                item.pending = False  # A concurrent deletion must not be resurrected by old I/O.
                continue
            superseded = superseded or known is None and row is not None
            if superseded:
                continue
            if vacancy is not None:
                last_seen = (
                    max(aware(row.last_seen_at), aware(observation.last_seen_at))
                    if row
                    else aware(observation.last_seen_at)
                )
                repo = VacancyRepository(session)
                vid = await repo.save(
                    vacancy.model_copy(update={"collected_at": now}), seen_in_search=False
                )
                row = await session.get(VacancyModel, vid, populate_existing=True)
                row.last_seen_at = last_seen
                row.search_fingerprint = observation.fingerprint
                row.next_check_at = now + timedelta(seconds=check_interval(row, now))
                if analysis_pipeline and row.availability == "active":
                    await schedule_collected(
                        session, vacancy.model_copy(update={"id": vid}), pipeline=analysis_pipeline
                    )
                count += 1
            elif row:
                row.last_check_attempt_at, row.check_error = now, error
                if state == "closed":
                    row.archived, row.availability, row.last_checked_at = True, "closed", now
                    row.closed_at = row.closed_at or now
                else:
                    row.availability = "unknown"
            item.pending = error is not None
            item.attempts = item.attempts + 1 if error else 0
            item.retry_at = now + timedelta(seconds=min(21600, 60 * 2 ** min(item.attempts, 9)))
            if state == "closed":
                item.retry_at = now + timedelta(hours=6)
            if row and error:
                row.next_check_at = item.retry_at
            direction = await session.get(Direction, key)
            stats = dict(direction.progress)
            metric = (
                "detail_errors"
                if error
                else "missing"
                if state == "closed"
                else repo.last_save["change_kind"]
            )
            stats[metric] = stats.get(metric, 0) + 1
            direction.progress = stats
    return count
