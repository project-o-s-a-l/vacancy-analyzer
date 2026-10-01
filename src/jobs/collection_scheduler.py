"""Refresh directions requested by users; keep network rate limiting in the source."""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import asdict
from datetime import UTC, datetime

import httpx
import requests
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from collectors.base import SearchQuery, VacancySource
from domain import CandidateProfile
from jobs.catalog_checks import check_catalog
from jobs.collect import collect_vacancies
from jobs.search_plan import PLAN_VERSION, plan_profile, plan_queries
from nlp.ollama import OllamaVacancyAnalyzer
from observability import emit_event, operation
from storage.models import SearchPlanModel
from storage.payloads import payload_hash
from storage.repositories.user_repository import UserRepository

logger = logging.getLogger(__name__)


def query_key(text: str) -> str:
    return " ".join(text.split()).casefold()


def search_key(query: SearchQuery) -> tuple[str, str]:
    return query_key(query.text), query.search_field


def build_queries(texts: Iterable[str], template: SearchQuery) -> list[SearchQuery]:
    unique: dict[str, SearchQuery] = {}
    for text in texts:
        cleaned = " ".join(text.split())
        key = query_key(cleaned)
        if key and key not in unique:
            unique[key] = SearchQuery(**{**template.model_dump(), "text": cleaned})
    return list(unique.values())


def profile_queries(
    profiles: Iterable[CandidateProfile], template: SearchQuery
) -> list[SearchQuery]:
    unique: dict[tuple[str, str], SearchQuery] = {}
    for profile in profiles:
        for query in plan_queries(plan_profile(profile), template):
            unique.setdefault(search_key(query), query)
    return list(unique.values())


async def load_profile_queries(
    sessions: async_sessionmaker[AsyncSession], template: SearchQuery
) -> list[SearchQuery]:
    async with sessions.begin() as session:
        users = await UserRepository(session).search_users(lock=True)
        # Keep only current plans, including when an owner is back in setup.
        await session.execute(
            delete(SearchPlanModel).where(
                SearchPlanModel.user_id.not_in([u.user_id for u in users])
            )
        )
        for user in users:
            queries = [
                asdict(item) | {"search_field": "all"} for item in plan_profile(user.profile)
            ]
            digest = payload_hash({"version": PLAN_VERSION, "queries": queries})
            row = await session.get(SearchPlanModel, user.user_id)
            if row is None:
                session.add(
                    SearchPlanModel(
                        user_id=user.user_id,
                        version=PLAN_VERSION,
                        plan_hash=digest,
                        queries=queries,
                        updated_at=datetime.now(UTC),
                    )
                )
            elif row.plan_hash != digest:
                row.version, row.plan_hash, row.queries, row.updated_at = (
                    PLAN_VERSION,
                    digest,
                    queries,
                    datetime.now(UTC),
                )
    return profile_queries((u.profile for u in users), template)


async def collection_loop(
    source: VacancySource,
    sessions: async_sessionmaker[AsyncSession],
    template: SearchQuery,
    watch: bool,
    interval: float,
    *,
    query_texts: list[str] | None = None,
    profile_check_interval: float = 60,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    analyzer: OllamaVacancyAnalyzer | None = None,
    analysis_pipeline: str | None = None,
    check_limit: int = 0,
    check_interval: float = 86400,
    check_batch_interval: float = 3600,
) -> None:
    """Check profiles between directions and during idle time; schedule each direction fairly."""
    if (
        interval <= 0
        or profile_check_interval <= 0
        or not 0 <= check_limit <= 10
        or check_interval < 3600
        or check_batch_interval < 3600
    ):
        raise ValueError("Интервалы сбора должны быть положительными")
    fixed = build_queries(query_texts, template) if query_texts is not None else None
    next_run: dict[tuple[str, str], float] = {}
    last_plan: frozenset[tuple[str, str]] | None = None
    next_check = clock()
    while True:
        queries = fixed if fixed is not None else await load_profile_queries(sessions, template)
        active = frozenset(search_key(query) for query in queries)
        # Removed directions must stop running, and re-added directions start immediately.
        next_run = {key: due for key, due in next_run.items() if key in active}
        if active != last_plan:
            emit_event(logger, "collection.plan_changed", count=len(active))
            if not active:
                emit_event(logger, "collection.no_profiles", result="skipped")
            last_plan = active

        if watch and check_limit and clock() >= next_check:
            await check_catalog(
                source,
                sessions,
                limit=check_limit,
                interval=check_interval,
                analysis_pipeline=analysis_pipeline,
            )
            next_check = clock() + check_batch_interval

        if watch:
            now = clock()
            pending = [query for query in queries if next_run.get(search_key(query), 0) <= now]
            # Read profiles again after each completed direction, even while other work remains.
            batch = sorted(pending, key=lambda query: next_run.get(search_key(query), 0))[:1]
        else:
            batch = queries

        for query in batch:
            with operation("direction"):
                started = time.monotonic()
                emit_event(logger, "collection.cycle_started", count=len(batch))
                emit_event(
                    logger,
                    "collection.direction_started",
                    scope=query.search_field,
                    page_budget=query.max_pages,
                )
                try:
                    count = await collect_vacancies(
                        source,
                        sessions,
                        query,
                        analyzer=analyzer,
                        analysis_pipeline=analysis_pipeline,
                    )
                    emit_event(
                        logger,
                        "collection.direction_finished",
                        count=count,
                        result="success",
                        duration_ms=(time.monotonic() - started) * 1000,
                    )
                except (requests.RequestException, httpx.RequestError) as exc:
                    emit_event(
                        logger,
                        "collection.direction_finished",
                        result="deferred" if watch else "failed",
                        code="network_error",
                        duration_ms=(time.monotonic() - started) * 1000,
                    )
                    emit_event(
                        logger,
                        "collection.cycle_finished",
                        result="deferred" if watch else "failed",
                        duration_ms=(time.monotonic() - started) * 1000,
                    )
                    if not watch:
                        raise
                    emit_event(
                        logger,
                        "collection.network_deferred",
                        level=logging.WARNING,
                        error=exc,
                        code="network_error",
                        result="deferred",
                    )
                except BaseException as exc:
                    result = "cancelled" if isinstance(exc, asyncio.CancelledError) else "stopped"
                    emit_event(
                        logger,
                        "collection.direction_finished",
                        result=result,
                        code=getattr(exc, "code", None),
                        duration_ms=(time.monotonic() - started) * 1000,
                    )
                    emit_event(
                        logger,
                        "collection.cycle_finished",
                        result=result,
                        duration_ms=(time.monotonic() - started) * 1000,
                    )
                    raise
                else:
                    emit_event(
                        logger,
                        "collection.cycle_finished",
                        count=count,
                        result="success",
                        duration_ms=(time.monotonic() - started) * 1000,
                    )
            next_run[search_key(query)] = clock() + interval

        if not watch:
            return
        if batch:
            continue
        wait = (
            min(profile_check_interval, min(next_run.values()) - clock())
            if next_run
            else profile_check_interval
        )
        if check_limit:
            wait = min(wait, next_check - clock())
        await sleep(max(0, wait))
