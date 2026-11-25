"""Collection lifecycle shared by application hosts and the standalone CLI."""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

import httpx
from asyncpg.exceptions import CannotConnectNowError, PostgresConnectionError
from sqlalchemy import text, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.exc import TimeoutError as PoolTimeout
from sqlalchemy.ext.asyncio import AsyncEngine

from collectors.base import SearchQuery
from collectors.errors import CollectionStopped, VacancyParseError
from collectors.registry import SourceAdapterError, source_adapter
from jobs.collection_budget import BudgetedSource
from jobs.collection_scheduler import collection_loop
from jobs.settings import COLLECTION_SOURCES, Settings
from nlp.analysis_identity import pipeline_hash
from observability import bind_context, emit_event, operation
from storage.database import session_factory
from storage.models import CollectionDirection

logger = logging.getLogger(__name__)
# Session-scoped PostgreSQL lock; all hosts collecting into the same DB share it.
COLLECTION_LOCK = 7_336_507_829_142_601


class CollectionAlreadyRunning(RuntimeError):
    pass


@asynccontextmanager
async def collection_lease(engine: AsyncEngine) -> AsyncIterator[None]:
    if engine.dialect.name != "postgresql":
        # SQLite is used for local tests; production coordination uses PostgreSQL.
        yield
        return
    async with engine.connect() as connection:
        await connection.execution_options(isolation_level="AUTOCOMMIT")
        acquired = None
        try:
            acquired = await connection.scalar(
                text("SELECT pg_try_advisory_lock(:key)"), {"key": COLLECTION_LOCK}
            )
            if not acquired:
                raise CollectionAlreadyRunning(
                    "Сборщик уже работает для этой базы. Остановите его перед ручным сбором."
                )
            yield
        finally:
            if acquired:
                try:
                    await connection.execute(
                        text("SELECT pg_advisory_unlock(:key)"), {"key": COLLECTION_LOCK}
                    )
                except BaseException:
                    # A pooled connection must never retain an untracked session lock.
                    await connection.invalidate()
                    raise
            elif acquired is None:
                await connection.invalidate()


async def run_collection(
    settings: Settings,
    engine: AsyncEngine,
    query: SearchQuery,
    *,
    watch: bool,
    new_only: bool = False,
    query_texts: list[str] | None = None,
    source: str | None = None,
) -> None:
    sources = collection_sources(settings, query, source)
    if any(s not in {"api", "html"} for s in sources) and (
        query.area or query.period or query.search_field == "name"
    ):
        raise ValueError("--area, --days и поиск только по названию поддерживаются с --source hh")
    adapters = {}
    unavailable = {}
    for name in sources:
        try:
            adapters[name] = source_adapter(name)
        except SourceAdapterError as exc:
            unavailable[name] = exc
    if unavailable and not adapters:
        raise next(iter(unavailable.values()))
    sessions = session_factory(engine)
    new_only = new_only or settings.collection_new_only

    async def collect(adapter, kind) -> None:
        hh = kind in {"api", "html"}
        paced = kind in {"superjob", "avito"}
        if kind == "avito":
            request_policy = {
                "new_limit": settings.avito_new_target_per_hour if new_only else None,
                "stop_on_initial_429": source == "avito",
                "search_limit": settings.avito_search_requests_per_hour,
                "detail_limit": settings.avito_detail_requests_per_hour,
                "delay": settings.avito_request_delay,
                "request_intervals": {
                    "search": 3600 / settings.avito_search_requests_per_hour,
                    "detail": 3600 / settings.avito_detail_requests_per_hour,
                },
            }
        elif kind == "superjob":
            request_policy = {
                "search_limit": None,
                "detail_limit": None,
                "delay": settings.superjob_request_delay,
                "request_intervals": {
                    "search": settings.public_request_delay,
                    "detail": 3600 / settings.superjob_detail_target_per_hour,
                },
            }
        else:
            request_policy = {
                "search_limit": settings.hh_search_requests_per_hour
                if hh
                else settings.public_search_requests_per_hour,
                "detail_limit": settings.hh_detail_requests_per_hour
                if hh
                else settings.public_detail_requests_per_hour,
                "delay": settings.hh_request_delay if hh else settings.public_request_delay,
            }
        emit_event(
            logger,
            "component.ready",
            component_kind="collector",
            result="ready",
        )
        await collection_loop(
            BudgetedSource(adapter, sessions, **request_policy)
            if getattr(adapter, "paging_kind", None)
            else adapter,
            sessions,
            query,
            watch,
            settings.hh_poll_interval,
            query_texts=query_texts,
            new_only=new_only,
            profile_check_interval=settings.hh_profile_check_interval,
            active_slice_interval=1 if paced else None,
            check_limit=settings.hh_check_limit,
            check_interval=settings.hh_check_interval,
            check_batch_interval=settings.hh_check_batch_interval,
            analysis_pipeline=pipeline_hash("vacancy_analysis", settings.ollama_model)
            if settings.vacancy_llm_enabled
            else None,
        )

    async def run_source(kind):
        with bind_context(source_kind=kind):
            started = time.monotonic()
            emit_event(logger, "component.started", component_kind="collector")
            try:
                if kind in unavailable:
                    raise unavailable[kind]
                async with adapters[kind].open(settings) as adapter:
                    await collect(adapter, kind)
            except asyncio.CancelledError:
                emit_event(
                    logger,
                    "component.finished",
                    component_kind="collector",
                    result="cancelled",
                    duration_ms=(time.monotonic() - started) * 1000,
                )
                raise
            else:
                emit_event(
                    logger,
                    "component.finished",
                    component_kind="collector",
                    result="success",
                    duration_ms=(time.monotonic() - started) * 1000,
                )

    async with collection_lease(engine):
        enabled = {"hh" if kind in {"html", "api"} else kind for kind in sources}
        disabled = set(COLLECTION_SOURCES) - enabled
        if watch and source is None and query_texts is None and disabled:
            async with sessions.begin() as session:
                await session.execute(
                    update(CollectionDirection)
                    .where(
                        CollectionDirection.source.in_(disabled),
                        CollectionDirection.active.is_(True),
                    )
                    .values(active=False)
                )
        if len(sources) == 1:
            await run_source(sources[0])
            return
        failures = []

        async def independent(kind):
            retry = 60
            while True:
                try:
                    await run_source(kind)
                    return
                except Exception as exc:
                    emit_event(
                        logger,
                        "collection.stopped",
                        level=logging.ERROR,
                        source_kind=kind,
                        error=exc,
                        code=getattr(exc, "code", "source_error"),
                        result="deferred" if watch and kind not in unavailable else "stopped",
                    )
                    if watch and kind in unavailable:
                        return
                    if not watch:
                        failures.append(kind)
                        return
                await asyncio.sleep(retry)
                retry = min(3600, retry * 2)

        async with asyncio.TaskGroup() as group:
            for kind in sources:
                group.create_task(independent(kind), name=f"collect-{kind}")
        if failures:
            raise ValueError("Сбор завершился с ошибкой источника: " + ", ".join(failures))


def collection_sources(settings, query, selected):
    if selected == "all":
        kinds = COLLECTION_SOURCES
    elif selected is None and query.category == "it":
        kinds = settings.collection_sources
    else:
        kinds = (selected or "hh",)
    return list(dict.fromkeys(settings.hh_source if kind == "hh" else kind for kind in kinds))


async def _background_worker(settings: Settings, engine: AsyncEngine) -> None:
    with operation("collection"):
        delay = 5
        while await _run_background_worker(settings, engine):
            await asyncio.sleep(delay)
            delay = min(delay * 2, 300)


async def _run_background_worker(settings: Settings, engine: AsyncEngine) -> bool:
    try:
        query = (
            SearchQuery(category="it")
            if getattr(settings, "collection_mode", "it") == "it"
            else SearchQuery(text="profile")
        )
        await run_collection(settings, engine, query, watch=True)
    except CollectionAlreadyRunning as exc:
        emit_event(
            logger,
            "collection.already_running",
            level=logging.WARNING,
            error=exc,
            code="already_running",
            result="skipped",
        )
        return True
    except (
        PostgresConnectionError,
        CannotConnectNowError,
        DBAPIError,
        PoolTimeout,
        ConnectionError,
        TimeoutError,
        OSError,
    ) as exc:
        if isinstance(exc, DBAPIError):
            code = getattr(exc.orig, "sqlstate", "") or ""
            transient = (
                exc.connection_invalidated
                or code.startswith(("08", "40", "53"))
                or code in {"55P03", "57P01", "57P02", "57P03"}
            )
            if code and not transient:
                emit_event(
                    logger,
                    "collection.stopped",
                    level=logging.ERROR,
                    error=exc,
                    code="unexpected_error",
                    result="stopped",
                )
                return False
        emit_event(
            logger,
            "collection.stopped",
            level=logging.WARNING,
            error=exc,
            code="infrastructure_unavailable",
            result="deferred",
        )
        return True
    except (CollectionStopped, VacancyParseError) as exc:
        emit_event(
            logger,
            "collection.stopped",
            level=logging.ERROR,
            error=exc,
            code="parse_error" if isinstance(exc, VacancyParseError) else exc.code,
            status_code=getattr(exc, "status_code", None),
            stage=getattr(exc, "stage", None),
            source_vacancy_id=getattr(exc, "source_vacancy_id", None),
            result="stopped",
        )
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        emit_event(
            logger,
            "collection.stopped",
            level=logging.ERROR,
            error=exc,
            status_code=status,
            code=f"hh_http_{status}" if status in {403, 429} else "http_error",
            result="stopped",
        )
    except Exception as exc:
        # Do not expose connection strings, tokens or raw HTTP/SQL diagnostics.
        emit_event(
            logger,
            "collection.stopped",
            level=logging.ERROR,
            error=exc,
            code="unexpected_error",
            result="stopped",
        )

    return False


@asynccontextmanager
async def background_collection(settings: Settings, engine: AsyncEngine) -> AsyncIterator[None]:
    """Attach one managed worker to a bot or future web application's lifespan."""
    if not settings.hh_auto_collect:
        yield
        return
    task = asyncio.create_task(_background_worker(settings, engine), name="vacancy-collection")
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
