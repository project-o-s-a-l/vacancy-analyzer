"""Collection lifecycle shared by application hosts and the standalone CLI."""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Literal

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from collectors.base import SearchQuery
from collectors.errors import CollectionStopped, VacancyParseError
from collectors.registry import source_adapter
from jobs.collection_scheduler import collection_loop
from jobs.settings import Settings
from nlp.analysis_identity import pipeline_hash
from observability import emit_event, operation
from storage.database import session_factory

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
    query_texts: list[str] | None = None,
    source: Literal["html", "api"] = "html",
) -> None:
    async def collect(adapter) -> None:
        emit_event(
            logger,
            "component.ready",
            component_kind="collector",
            source_kind=source,
            result="ready",
        )
        await collection_loop(
            adapter,
            session_factory(engine),
            query,
            watch,
            settings.hh_poll_interval,
            query_texts=query_texts,
            profile_check_interval=settings.hh_profile_check_interval,
            check_limit=settings.hh_check_limit,
            check_interval=settings.hh_check_interval,
            check_batch_interval=settings.hh_check_batch_interval,
            analysis_pipeline=pipeline_hash("vacancy_analysis", settings.ollama_model)
            if settings.vacancy_llm_enabled
            else None,
        )

    integration = source_adapter(source)
    async with collection_lease(engine):
        started = time.monotonic()
        emit_event(logger, "component.started", component_kind="collector", source_kind=source)
        try:
            async with integration.open(settings) as adapter:
                await collect(adapter)
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


async def _background_worker(settings: Settings, engine: AsyncEngine) -> None:
    with operation("collection"):
        await _run_background_worker(settings, engine)


async def _run_background_worker(settings: Settings, engine: AsyncEngine) -> None:
    try:
        await run_collection(settings, engine, SearchQuery(text="profile"), watch=True)
    except CollectionAlreadyRunning as exc:
        emit_event(
            logger,
            "collection.already_running",
            level=logging.WARNING,
            error=exc,
            code="already_running",
            result="skipped",
        )
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
