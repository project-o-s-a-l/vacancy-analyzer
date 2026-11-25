"""Restartable page sweeps with overlap, independent head discovery and durable detail inbox."""

import logging
from datetime import UTC, datetime, timedelta

import httpx
import requests
from sqlalchemy import func, select, update

from collectors.base import SearchCard
from collectors.errors import CollectionStopped, VacancyParseError
from jobs.analysis_queue import schedule_collected
from jobs.catalog_freshness import RECHECK_AFTER, detail_due
from jobs.collection_budget import BudgetDeferred, aware
from jobs.settings import DEFAULT_CHECK_INTERVAL
from observability import emit_event
from storage.models import CollectionDirection as Direction
from storage.models import CollectionObservation as Observation
from storage.models import VacancyModel
from storage.payloads import payload_hash
from storage.repositories import VacancyRepository

logger = logging.getLogger(__name__)


def direction_key(source, query):
    values = query.model_dump(exclude={"max_pages", "window_from", "window_to"})
    for field in ("category", "professional_role", "experience"):
        if values[field] is None:
            del values[field]  # Preserve persisted legacy HH direction hashes.
    values["text"] = " ".join(query.text.casefold().split())
    return payload_hash({"adapter": source.paging_kind, "query": values, "crawl": "overlap-v1"})


def initial_progress(now):
    return {
        "page": 0,
        "overlap": False,
        "initial_done": False,
        "head_due": now.isoformat(),
        "cycles": 0,
    }


async def ensure_direction(session, source, key, now, *, active):
    statement = (
        VacancyRepository(session)
        .insert(Direction)
        .values(
            key=key,
            adapter=source.paging_kind,
            source=getattr(source, "source_name", "hh"),
            active=active,
            demand=1,
            created_at=now,
            next_due_at=now,
            progress=initial_progress(now),
        )
    )
    await session.execute(statement.on_conflict_do_nothing(index_elements=[Direction.key]))


async def sync_directions(source, sessions, queries, *, demand=None, now=None):
    now = now or datetime.now(UTC)
    keys = {direction_key(source, query): query for query in queries}
    source_name = getattr(source, "source_name", "hh")
    async with sessions.begin() as session:
        # An adapter switch or removal suspends work; raw vacancies/history are retained.
        await session.execute(
            update(Direction)
            .where(
                Direction.source == source_name,
                Direction.active.is_(True),
                Direction.key.not_in(keys),
            )
            .values(active=False)
        )
        existing = {
            row.key: row
            for row in await session.scalars(select(Direction).where(Direction.key.in_(keys)))
        }
        for key, query in keys.items():
            row = existing.get(key)
            if row is None:
                await ensure_direction(session, source, key, now, active=True)
                row = await session.get(Direction, key)
            if not row.active:
                row.active, row.next_due_at = True, now
            row.demand = max(1, (demand or {}).get(query.text.casefold(), 1))
    return keys


async def choose_direction(sessions, keys, *, now=None, due_only=True):
    now = now or datetime.now(UTC)
    async with sessions() as session:
        rows = list(
            await session.scalars(
                select(Direction).where(
                    Direction.key.in_(keys),
                    Direction.active.is_(True),
                    Direction.next_due_at <= now if due_only else True,
                )
            )
        )

    def priority(row):
        age = (now - aware(row.last_run_at or row.created_at)).total_seconds()
        # A hard aging lane precedes demand. Even rare directions eventually win.
        starved = age >= 7200
        coverage = row.progress.get("unique", 0)
        return (
            aware(row.next_due_at) > now,
            not starved,
            -age if starved else 0,
            -(min(row.demand, 10) / (1 + min(coverage, 100) / 20)),
            aware(row.last_run_at or row.created_at),
            row.key,
        )

    return min(rows, key=priority).key if rows else None


def _publisher_directions(source):
    return select(Direction.key).where(
        Direction.source == source.source_name, Direction.adapter == source.paging_kind
    )


async def _record_page(session, key, source, result, progress, now, refresh_after, new_only):
    source_name = getattr(source, "source_name", "hh")
    cards = {card.source_id: card for card in result.cards}
    existing = {
        obs.source_id: obs
        for obs in await session.scalars(
            select(Observation).where(
                Observation.direction == key, Observation.source_id.in_(cards)
            )
        )
    }
    known_rows = (
        await session.execute(
            select(
                VacancyModel.id,
                VacancyModel.source_id,
                VacancyModel.last_checked_at,
                VacancyModel.check_error,
                VacancyModel.next_check_at,
            ).where(VacancyModel.source == source_name, VacancyModel.source_id.in_(cards))
        )
    ).all()
    known_by_id = {row.source_id: row for row in known_rows}
    resolved = {}
    unknown = cards.keys() - known_by_id.keys()
    if unknown and getattr(source, "it_search_terms", ()):
        # Overlapping keyword searches may have already resolved this unmodified ad.
        rows = await session.execute(
            select(Observation.source_id, Observation.fingerprint, func.max(Observation.retry_at))
            .where(
                Observation.direction.in_(_publisher_directions(source)),
                Observation.source_id.in_(unknown),
                Observation.pending.is_(False),
                Observation.retry_at > now,
            )
            .group_by(Observation.source_id, Observation.fingerprint)
        )
        resolved = {(sid, fingerprint): deadline for sid, fingerprint, deadline in rows}
    if known_rows and not new_only:
        await session.execute(
            update(VacancyModel)
            .where(
                VacancyModel.id.in_([row.id for row in known_rows]),
                VacancyModel.last_seen_at < now,
            )
            .values(last_seen_at=now)
        )
    new = 0
    repeated = len(result.cards) - len(cards)
    for card in cards.values():
        observation = existing.get(card.source_id)
        known = known_by_id.get(card.source_id)
        repeated += int(known is not None or observation is not None)
        new += int(known is None and observation is None)
        needs = known is None or (not new_only and detail_due(known, now, refresh_after))
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
            if observation.fingerprint != card.fingerprint:
                observation.retry_at = now
            elif known is None and not observation.pending and aware(observation.retry_at) > now:
                # A missing or out-of-scope card was already resolved during this sweep.
                needs = False
            observation.pending = observation.pending or bool(needs)
        if deadline := resolved.get((card.source_id, card.fingerprint)):
            observation.pending, observation.retry_at = False, deadline
        observation.fingerprint = card.fingerprint
        observation.detail_url = card.detail_url
        if observation.pending and card.payload is not None:
            observation.payload = card.payload
    progress["unique"] = progress.get("unique", 0) + len(cards) - len(existing)
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
    progress.pop("search_retry_at", None)


def _advance_sweep(progress, result, query, page, head, windows, source, now, interval):
    """Return True only after every accessible page or time window has been visited."""
    depth_limit = source.depth_limit
    cap = depth_limit // query.per_page if depth_limit else None
    at_cap = cap is not None and page + 1 >= cap
    exceeds_cap = (
        cap is not None and result.total is not None and result.total > cap * query.per_page
    )
    oversized = page == 0 and exceeds_cap
    levels = getattr(source, "experience_levels", ())
    if oversized and levels and query.category == "it" and not query.experience:
        progress.update(experiences=list(levels), page=0, overlap=False)
        return False
    if windows:
        lower, upper = windows[-1]
        progress.setdefault("window_end", windows[0][1])
        progress["windows"] = windows
        if oversized:
            left = query.window_from or datetime(1970, 1, 1, tzinfo=UTC)
            right = query.window_to
            if right - left > timedelta(minutes=10):
                pivot = max(left + (right - left) / 2, right - timedelta(days=30))
                middle = pivot.replace(microsecond=0).isoformat()
                progress.update(
                    windows=[*windows[:-1], [lower, middle], [middle, upper]], page=0, overlap=False
                )
                return False
    if head:
        progress["head_due"] = (now + timedelta(seconds=interval)).isoformat()
        return False
    if progress["overlap"]:
        progress["overlap"] = False
        return False
    if result.has_more and not at_cap:
        progress.update(page=page + 1, overlap=not windows)
        if page == 0:
            progress["head_due"] = (now + timedelta(seconds=interval)).isoformat()
        return False
    truncated = at_cap and (result.has_more or exceeds_cap)
    truncated = progress.pop("window_truncated", False) or truncated
    progress.update(page=0, overlap=False)
    if len(windows) > 1:
        progress.update(windows=windows[:-1], window_truncated=truncated)
        return False
    remaining = progress.pop("experiences", [])[1:]
    if remaining:
        progress.update(experiences=remaining, window_truncated=truncated)
        return False
    progress.update(
        initial_done=True,
        depth_limited=truncated,
        cycles=progress["cycles"] + 1,
        last_sweep_at=now.isoformat(),
    )
    if windows:
        progress.pop("windows", None)
        end = datetime.fromisoformat(progress.pop("window_end"))
        if not truncated:
            progress["watermark"] = (end - timedelta(minutes=5)).isoformat()
    return True


def pending_observations(key, source_name, new_only):
    conditions = [Observation.direction == key, Observation.pending.is_(True)]
    if new_only:
        # Exclude deferred refreshes before applying either backpressure or the detail limit.
        known = select(VacancyModel.id).where(
            VacancyModel.source == source_name, VacancyModel.source_id == Observation.source_id
        )
        conditions.append(~known.exists())
    return conditions


async def crawl_slice(
    source,
    sessions,
    query,
    *,
    analysis_pipeline=None,
    detail_limit=10,
    interval=3600,
    check_interval=DEFAULT_CHECK_INTERVAL,
    new_only=False,
    manual_run_started_at=None,
    active_slice_interval=None,
    now=None,
):
    now = now or datetime.now(UTC)
    key = direction_key(source, query)
    source_name = getattr(source, "source_name", "hh")
    refresh_after = timedelta(seconds=check_interval)
    pending_filter = pending_observations(key, source_name, new_only)
    count_pending = select(func.count()).select_from(Observation).where(*pending_filter)
    async with sessions.begin() as session:
        row = await session.get(Direction, key, with_for_update=True)
        if row is None or not row.active:
            return 0
        progress = dict(row.progress)
        row.last_run_at = now
        row.next_due_at = now + timedelta(seconds=60)
        pending = await session.scalar(count_pending)
    # Bounded inbox backpressure. A failed item remains durable without rewinding the page.
    sweep_due = (
        not progress.get("last_sweep_at")
        or (
            manual_run_started_at is not None
            and datetime.fromisoformat(progress["last_sweep_at"]) < manual_run_started_at
        )
        or progress.get("windows")
        or progress.get("experiences")
        or progress["page"] > 0
        or datetime.fromisoformat(progress["last_sweep_at"]) + timedelta(seconds=interval) <= now
    )
    search_retry = progress.get("search_retry_at")
    result = None
    sweep_finished = False
    if (
        pending < max(2 * query.per_page, detail_limit)
        and sweep_due
        and (not search_retry or datetime.fromisoformat(search_retry) <= now)
    ):
        windowed = (
            bool(getattr(source, "window_kind", None))
            and query.category == "it"
            and not query.period
        )
        scoped_query = query
        windows = []
        if progress.get("experiences"):
            scoped_query = query.model_copy(update={"experience": progress["experiences"][0]})
        if windowed:
            windows = progress.get("windows") or [[progress.get("watermark"), now.isoformat()]]
            lower, upper = windows[-1]
            scoped_query = query.model_copy(
                update={
                    "window_from": datetime.fromisoformat(lower) if lower else None,
                    "window_to": datetime.fromisoformat(upper),
                }
            )
        head = (
            not windowed
            and progress["page"] > 0
            and datetime.fromisoformat(progress["head_due"]) <= now
        )
        page = 0 if head else max(0, progress["page"] - int(progress["overlap"]))
        try:
            result = await source.search_page(scoped_query, page)
        except BudgetDeferred:
            result = None
        except (requests.RequestException, httpx.HTTPError, CollectionStopped, ValueError) as exc:
            async with sessions.begin() as session:
                row = await session.get(Direction, key, with_for_update=True)
                state = dict(row.progress)
                state["search_errors"] = state.get("search_errors", 0) + 1
                state["error"] = getattr(exc, "code", "search_error")
                row.next_due_at = now + timedelta(
                    seconds=min(3600, 60 * 2 ** min(state["search_errors"], 6))
                )
                state["search_retry_at"] = row.next_due_at.isoformat()
                row.progress = state
            result = None
        if result is not None:
            async with sessions.begin() as session:
                row = await session.get(Direction, key, with_for_update=True)
                progress = dict(row.progress)
                await _record_page(
                    session, key, source, result, progress, now, refresh_after, new_only
                )
                sweep_finished = _advance_sweep(
                    progress, result, scoped_query, page, head, windows, source, now, interval
                )
                row.progress = progress
    count = await drain_details(
        source,
        sessions,
        key,
        limit=detail_limit,
        analysis_pipeline=analysis_pipeline,
        now=now,
        refresh_after=refresh_after,
        new_only=new_only,
    )
    async with sessions.begin() as session:
        pending = await session.scalar(count_pending)
        row = await session.get(Direction, key, with_for_update=True)
        slice_interval = (
            active_slice_interval
            if active_slice_interval is not None and (result is not None or count)
            else 60
        )
        row.next_due_at = now + timedelta(seconds=slice_interval)
        if not pending and row.progress.get("search_retry_at"):
            row.next_due_at = datetime.fromisoformat(row.progress["search_retry_at"])
        elif (
            not pending
            and row.progress.get("last_sweep_at")
            and row.progress["page"] == 0
            and not row.progress.get("windows")
            and not row.progress.get("experiences")
        ):
            next_sweep_at = datetime.fromisoformat(row.progress["last_sweep_at"]) + timedelta(
                seconds=interval
            )
            row.next_due_at = (
                max(row.next_due_at, next_sweep_at)
                if active_slice_interval is not None and result is None
                else next_sweep_at
            )
    emit_event(logger, "collection.progress", count=count, result="success")
    if sweep_finished:
        limited = progress["depth_limited"]
        emit_event(
            logger,
            "collection.sweep_finished",
            source_kind=source_name,
            total=result.total,
            pending_count=pending,
            next_run_at=now + timedelta(seconds=interval),
            code="source_depth_limit" if limited else None,
            result="success",
        )
    return count


async def drain_details(
    source,
    sessions,
    key,
    *,
    limit,
    analysis_pipeline,
    now,
    refresh_after=RECHECK_AFTER,
    new_only=False,
):
    source_name = getattr(source, "source_name", "hh")
    async with sessions() as session:
        due = list(
            await session.scalars(
                select(Observation)
                .where(
                    *pending_observations(key, source_name, new_only),
                    Observation.retry_at <= now,
                )
                .order_by(Observation.retry_at, Observation.source_id)
                .limit(limit)
            )
        )
        if not due:
            return 0
        known_by_id = {
            row.source_id: row
            for row in await session.execute(
                select(
                    VacancyModel.source_id,
                    VacancyModel.last_checked_at,
                    VacancyModel.check_error,
                    VacancyModel.last_check_attempt_at,
                    VacancyModel.next_check_at,
                ).where(
                    VacancyModel.source == source_name,
                    VacancyModel.source_id.in_([obs.source_id for obs in due]),
                )
            )
        }
    count = 0
    matches_catalog = getattr(source, "matches_catalog", None)
    for observation in due:
        known = known_by_id.get(observation.source_id)
        if new_only and known:
            continue
        inline = observation.payload is not None and getattr(source, "inline_details", False)
        checked_at = aware(observation.last_seen_at) if inline else now
        newer_detail = (
            inline and known and known.last_checked_at and aware(known.last_checked_at) > checked_at
        )
        if newer_detail or (known and not detail_due(known, now, refresh_after)):
            async with sessions.begin() as session:
                direction = await session.get(Direction, key, with_for_update=True)
                item = await session.get(Observation, (key, observation.source_id))
                item.pending = False
                item.payload = None
                stats = dict(direction.progress)
                stats["details_cached"] = stats.get("details_cached", 0) + 1
                direction.progress = stats
            continue
        state, error, vacancy = None, None, None
        try:
            if observation.payload is not None or observation.detail_url is not None:
                fetched = await source.get_observed(
                    SearchCard(
                        source_id=observation.source_id,
                        detail_url=observation.detail_url,
                        payload=observation.payload,
                    )
                )
            else:
                fetched = await source.get_vacancy(observation.source_id)
            if fetched.source != source_name or fetched.source_id != observation.source_id:
                raise ValueError("Source identity mismatch")
            if matches_catalog and not matches_catalog(fetched):
                state = "filtered"
                vacancy = fetched if known else None
            else:
                vacancy = fetched
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
            direction = await session.get(Direction, key, with_for_update=True)
            item = await session.get(
                Observation, (key, observation.source_id), with_for_update=True
            )
            if not item.pending:
                continue  # Another import or collector completed this detail during I/O.
            row = await session.scalar(
                select(VacancyModel)
                .where(
                    VacancyModel.source == source_name,
                    VacancyModel.source_id == observation.source_id,
                )
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
                    vacancy.model_copy(update={"collected_at": checked_at}),
                    seen_in_search=False,
                    insert_only=new_only,
                )
                if vid is None:
                    continue  # A concurrent writer already stored this publication.
                if new_only and getattr(source, "new_limit", None) is not None:
                    await source.record_new(session)
                row = await session.get(VacancyModel, vid, populate_existing=True)
                if state == "filtered" and not vacancy.archived:
                    row.availability = "out_of_scope"
                row.last_seen_at = last_seen
                row.search_fingerprint = observation.fingerprint
                row.next_check_at = checked_at + refresh_after
                if analysis_pipeline and row.availability == "active":
                    await schedule_collected(
                        session, vacancy.model_copy(update={"id": vid}), pipeline=analysis_pipeline
                    )
                if state != "filtered":
                    count += 1
            elif row and state != "filtered":
                row.last_check_attempt_at, row.check_error = now, error
                if state == "closed":
                    row.archived, row.availability, row.last_checked_at = True, "closed", now
                    row.closed_at = row.closed_at or now
                elif row.availability != "out_of_scope":
                    row.availability = "unknown"
            item.pending = error is not None
            if not item.pending:
                item.payload = None
            item.attempts = item.attempts + 1 if error else 0
            item.retry_at = now + timedelta(seconds=min(21600, 60 * 2 ** min(item.attempts, 9)))
            if state == "closed":
                item.retry_at = now + timedelta(hours=6)
            elif state == "filtered":
                item.retry_at = now + refresh_after
            if state in {"filtered", "closed"} and getattr(source, "it_search_terms", ()):
                # Other searches can already have queued the same ad before this decision.
                await session.execute(
                    update(Observation)
                    .where(
                        Observation.direction.in_(_publisher_directions(source)),
                        Observation.source_id == item.source_id,
                        Observation.fingerprint == item.fingerprint,
                        Observation.pending.is_(True),
                    )
                    .values(pending=False, payload=None, attempts=0, retry_at=item.retry_at)
                )
            if row and error:
                row.next_check_at = item.retry_at
            stats = dict(direction.progress)
            if error:
                metric = "detail_errors"
            elif state in {"filtered", "closed"}:
                metric = "filtered" if state == "filtered" else "missing"
            else:
                metric = repo.last_save["change_kind"]
            stats[metric] = stats.get(metric, 0) + 1
            direction.progress = stats
    return count
