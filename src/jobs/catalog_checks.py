"""Small sequential rechecks through the same rate-limited source and collector lease."""

import asyncio
import logging
import time
from datetime import UTC, datetime, timedelta

import httpx
import requests
from sqlalchemy import or_, select, update

from collectors.errors import CollectionStopped, VacancyParseError
from jobs.analysis_queue import schedule_collected
from jobs.catalog_freshness import check_interval
from jobs.collection_budget import BudgetDeferred
from observability import emit_event, operation
from observability.transactions import after_commit
from storage.models import VacancyModel
from storage.repositories import VacancyRepository

logger = logging.getLogger(__name__)


async def check_catalog(
    source, sessions, *, limit=2, interval=86400, analysis_pipeline=None, now=None
):
    with operation("catalog_check"):
        started = time.monotonic()
        try:
            return await _check_catalog(
                source,
                sessions,
                limit=limit,
                interval=interval,
                analysis_pipeline=analysis_pipeline,
                now=now,
            )
        except BaseException as exc:
            emit_event(
                logger,
                "catalog.check_finished",
                level=logging.INFO if isinstance(exc, asyncio.CancelledError) else logging.ERROR,
                result="cancelled"
                if isinstance(exc, asyncio.CancelledError)
                else "stopped"
                if isinstance(exc, CollectionStopped)
                else "failed",
                code=getattr(exc, "code", None),
                error=exc,
                duration_ms=(time.monotonic() - started) * 1000,
            )
            raise


async def _check_catalog(source, sessions, *, limit, interval, analysis_pipeline, now):
    if not 1 <= limit <= 100 or interval < 3600:
        raise ValueError("Недопустимый бюджет проверки каталога")
    now = now or datetime.now(UTC)
    started = time.monotonic()
    emit_event(logger, "catalog.check_started", total=limit)
    async with sessions.begin() as session:
        expired = (
            select(VacancyModel.id)
            .where(
                VacancyModel.availability.in_(["active", "unknown"]),
                VacancyModel.valid_until <= now,
            )
            .order_by(VacancyModel.valid_until, VacancyModel.id)
            .limit(100)
        )
        expired_result = await session.execute(
            update(VacancyModel).where(VacancyModel.id.in_(expired)).values(availability="expired")
        )
        if expired_result.rowcount:
            after_commit(
                session,
                "catalog.expired",
                count=expired_result.rowcount,
                code="source_expired",
                result="expired",
            )
        due = list(
            (
                await session.execute(
                    select(
                        VacancyModel.id, VacancyModel.source_id, VacancyModel.last_check_attempt_at
                    )
                    .where(
                        VacancyModel.source == "hh",
                        VacancyModel.availability.in_(["active", "unknown"]),
                        or_(VacancyModel.valid_until.is_(None), VacancyModel.valid_until > now),
                        or_(
                            VacancyModel.check_requested_at.is_not(None),
                            VacancyModel.last_checked_at.is_(None),
                            VacancyModel.last_checked_at <= now - timedelta(seconds=interval),
                            VacancyModel.next_check_at <= now,
                        ),
                        or_(
                            VacancyModel.next_check_at.is_(None),
                            VacancyModel.next_check_at <= now,
                            # Priority checks may accelerate a healthy card, never a failed retry.
                            (
                                VacancyModel.check_requested_at.is_not(None)
                                & VacancyModel.check_error.is_(None)
                            ),
                        ),
                    )
                    .order_by(
                        VacancyModel.check_requested_at.asc().nulls_last(),
                        VacancyModel.last_checked_at.asc().nulls_first(),
                        VacancyModel.collected_at,
                        VacancyModel.source_id,
                    )
                    .limit(limit)
                )
            ).all()
        )
    checked = 0
    for key, source_id, previous_attempt in due:
        state, error, stop = None, None, None
        status = None
        try:
            vacancy = await source.get_vacancy(source_id)
            if vacancy.source != "hh" or vacancy.source_id != source_id:
                raise VacancyParseError("Источник вернул другую вакансию")
        except BudgetDeferred:
            break
        except (requests.HTTPError, httpx.HTTPStatusError) as exc:
            status = exc.response.status_code if exc.response is not None else None
            state, error = ("closed", None) if status in {404, 410} else ("unknown", "http_error")
            if status in {403, 429}:
                stop = CollectionStopped(
                    "HH ограничил проверку актуальности",
                    code=f"hh_http_{status}",
                    status_code=status,
                )
        except CollectionStopped as exc:
            state, error, stop = "unknown", "access_blocked", exc
        except (VacancyParseError, ValueError):
            state, error = "unknown", "parse_error"
        except (requests.RequestException, httpx.RequestError):
            state, error = "unknown", "network_error"
        if stop:
            stop.stage = "catalog_check"
            stop.source_vacancy_id = source_id
        async with sessions.begin() as session:
            # No transaction is held during an HTTP request. Re-lock only the
            # affected row; deletion during a request must not resurrect it.
            row = await session.get(VacancyModel, key, with_for_update=True)
            if row is None:
                if stop:
                    raise stop
                continue
            if row.last_check_attempt_at != previous_attempt:
                # A newer source writer committed while this request was in flight.
                continue
            row.last_check_attempt_at = now
            checked += 1
            if state is not None:
                row.availability, row.check_error = state, error
                row.next_check_at = now + timedelta(hours=1)
                if state == "closed":
                    row.last_checked_at = now
                    row.archived = True
                    row.closed_at = row.closed_at or now
                after_commit(
                    session,
                    "catalog.checked",
                    level=logging.WARNING if state == "unknown" else logging.INFO,
                    source_vacancy_id=source_id,
                    vacancy_id=key,
                    result=state,
                    status_code=status or getattr(stop, "status_code", None),
                    code=stop.code if stop else error if error else "source_closed",
                )
            else:
                # Use the actual check time, never the original collection time.
                vacancy = vacancy.model_copy(update={"collected_at": now})
                await VacancyRepository(session).save(vacancy, seen_in_search=False)
                await session.refresh(row)
                row.next_check_at = now + timedelta(seconds=check_interval(row, now, interval))
                if (
                    analysis_pipeline
                    and not vacancy.archived
                    and (vacancy.valid_until is None or vacancy.valid_until > now)
                ):
                    await schedule_collected(
                        session, vacancy.model_copy(update={"id": key}), pipeline=analysis_pipeline
                    )
                after_commit(
                    session,
                    "catalog.checked",
                    source_vacancy_id=source_id,
                    vacancy_id=key,
                    result="closed" if vacancy.archived else "success",
                    code="source_closed" if vacancy.archived else None,
                )
        if stop:
            raise stop
    emit_event(
        logger,
        "catalog.check_finished",
        count=checked,
        result="success",
        duration_ms=(time.monotonic() - started) * 1000,
    )
    return checked
