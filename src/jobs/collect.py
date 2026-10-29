import logging
import time
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from collectors.base import SearchQuery, VacancySource
from domain.vacancy_analysis import is_fresh
from jobs.analysis_queue import schedule_collected
from nlp.analysis_identity import pipeline_hash
from nlp.ollama import OllamaVacancyAnalyzer
from observability import emit_event
from storage.repositories import VacancyRepository

logger = logging.getLogger(__name__)


async def collect_vacancies(
    source: VacancySource,
    sessions: async_sessionmaker[AsyncSession],
    query: SearchQuery,
    *,
    analyzer: OllamaVacancyAnalyzer | None = None,
    analysis_pipeline: str | None = None,
) -> int:
    started = time.monotonic()
    vacancies = await source.search(query)
    now = datetime.now(UTC)
    count = 0
    recipe = analysis_pipeline
    changes = {"new": 0, "changed": 0, "unchanged": 0}
    skipped = 0
    for vacancy in vacancies:
        if not is_fresh(vacancy, now=now, days=query.period):
            if vacancy.archived or vacancy.valid_until is not None and vacancy.valid_until <= now:
                async with sessions.begin() as session:
                    repository = VacancyRepository(session)
                    if await repository.get_by_source(vacancy.source, vacancy.source_id):
                        await repository.save(vacancy)
            skipped += 1
            code = (
                "source_closed"
                if vacancy.archived
                else (
                    "source_expired"
                    if vacancy.valid_until and vacancy.valid_until <= now
                    else "source_future"
                    if vacancy.published_at > now
                    else "source_age"
                )
            )
            emit_event(
                logger,
                "catalog.skipped",
                source_vacancy_id=vacancy.source_id,
                result="skipped",
                code=code,
            )
            continue
        # Commit source and task together before any model request.
        async with sessions.begin() as session:
            repository = VacancyRepository(session)
            vacancy_id = await repository.save(vacancy)
            if recipe is None and analyzer is not None:
                recipe = pipeline_hash("vacancy_analysis", analyzer.model)
            if recipe is not None:
                task = await schedule_collected(
                    session,
                    vacancy.model_copy(update={"id": vacancy_id}),
                    pipeline=recipe,
                )
        count += 1
        changes[repository.last_save["change_kind"]] += 1
        if analyzer is not None and task is not None:
            # Explicit one-shot callers use the same durable worker. Application
            # collection passes only a recipe and never waits for inference.
            from jobs.analysis_worker import AnalysisWorker

            await AnalysisWorker(sessions, vacancy_analyzer=analyzer, allow_resumes=False).run_once(
                job_id=task.id
            )
    emit_event(
        logger,
        "catalog.batch_finished",
        count=count,
        new_count=changes["new"],
        changed_count=changes["changed"],
        unchanged_count=changes["unchanged"],
        skipped_count=skipped,
        duration_ms=(time.monotonic() - started) * 1000,
        result="success",
    )
    return count
