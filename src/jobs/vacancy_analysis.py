"""Shared analysis lifecycle for the collector, CLI, and future web host."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from domain.vacancy_analysis import is_fresh
from jobs.model_telemetry import ObservedOllamaClient
from jobs.settings import Settings
from nlp.ollama import OllamaVacancyAnalyzer
from nlp.ollama_runtime import OllamaRuntime
from storage.models import VacancyModel
from storage.repositories import VacancyRepository


@asynccontextmanager
async def vacancy_analyzer(settings: Settings) -> AsyncIterator[OllamaVacancyAnalyzer | None]:
    if not settings.vacancy_llm_enabled:
        yield None
        return
    async with ObservedOllamaClient(
        model=settings.ollama_model,
        job_kind="vacancy_analysis",
        base_url=settings.ollama_base_url,
        timeout=httpx.Timeout(settings.ollama_timeout, connect=5),
        trust_env=False,
        follow_redirects=False,
    ) as client:
        yield OllamaVacancyAnalyzer(
            client,
            model=settings.ollama_model,
            runtime=OllamaRuntime(
                client,
                model=settings.ollama_model,
                keep_alive=settings.ollama_keep_alive,
                min_available_mb=settings.inference_min_available_mb,
                request_seconds=settings.ollama_timeout,
            ),
        )


async def analyze_catalog(
    sessions: async_sessionmaker[AsyncSession],
    analyzer: OllamaVacancyAnalyzer,
    *,
    days: int | None = None,
    limit: int = 20,
    force: bool = False,
) -> dict[str, int]:
    """Explicit one-shot caller; production CLI only schedules persisted work."""
    from jobs.analysis_queue import enqueue_vacancy
    from jobs.analysis_worker import AnalysisWorker
    from nlp.analysis_identity import pipeline_hash

    if (days is not None and days < 1) or not 1 <= limit <= 500:
        raise ValueError("Invalid analysis window or limit")
    now = datetime.now(UTC)
    async with sessions() as session:
        statement = (
            select(VacancyModel.id)
            .order_by(
                VacancyModel.published_at.desc(),
                VacancyModel.source_id,
            )
            .limit(5000)
        )
        if days is not None:
            statement = statement.where(VacancyModel.published_at >= now - timedelta(days=days))
        keys = list((await session.scalars(statement)).all())
    report = {"analyzed": 0, "cached": 0, "failed": 0, "changed": 0, "queued": 0}
    worker = AnalysisWorker(sessions, vacancy_analyzer=analyzer, allow_resumes=False)
    attempted = 0
    for key in keys:
        if attempted >= limit:
            break
        async with sessions.begin() as session:
            await session.get(VacancyModel, key, with_for_update=True)
            vacancy = await VacancyRepository(session).get_by_id(key)
            if vacancy is None or not is_fresh(vacancy, now=now, days=days):
                continue
            job = await enqueue_vacancy(
                session,
                vacancy,
                pipeline=pipeline_hash("vacancy_analysis", analyzer.model),
                force=force,
            )
        outcome = await worker.run_once(job_id=job.id)
        key = outcome if outcome in report else "queued"
        report[key] += 1
        if outcome != "cached":
            attempted += 1
    return report
