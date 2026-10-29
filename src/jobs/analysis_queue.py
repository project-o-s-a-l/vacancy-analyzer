"""Common persisted analysis scheduling; independent of any user interface."""

import logging
from datetime import UTC, datetime

from sqlalchemy import or_, select
from sqlalchemy.orm import load_only

from domain.vacancy_analysis import analysis_input_hash
from jobs.queue_policy import QueueFull
from nlp.analysis_identity import pipeline_hash
from observability import emit_event
from storage.models import BackgroundJob, VacancyModel
from storage.repositories import VacancyRepository
from storage.repositories.job_repository import JobRepository

logger = logging.getLogger(__name__)


async def schedule_collected(session, vacancy, *, pipeline):
    """Admission failure must not roll back a collected source or its deterministic facts."""
    try:
        return await enqueue_vacancy(session, vacancy, pipeline=pipeline)
    except QueueFull:
        emit_event(logger, "analysis.deferred", result="deferred", code="queue_full")
        return None


async def enqueue_vacancy(session, vacancy, *, pipeline, force=False):
    revision = analysis_input_hash(vacancy)
    source = (
        await session.execute(
            select(
                VacancyModel.id,
                VacancyModel.content_revision,
            )
            .where(VacancyModel.id == vacancy.id)
            .with_for_update()
        )
    ).one_or_none()
    if source is None:
        raise ValueError("Сначала сохраните исходную вакансию")
    current_revision = source.content_revision
    if current_revision is None:
        current = await VacancyRepository(session).get_by_id(vacancy.id)
        current_revision = analysis_input_hash(current)
    if current_revision != revision:
        raise ValueError("Содержимое вакансии изменилось до постановки задания")
    return await JobRepository(session).enqueue(
        kind="vacancy_analysis",
        entity_key=str(vacancy.id),
        vacancy_id=vacancy.id,
        revision=revision,
        pipeline=pipeline,
        force=force,
        payload={"force": force},
    )


async def enqueue_catalog(sessions, *, model, limit=500, force=False, missing_only=False):
    if not 1 <= limit <= 5000:
        raise ValueError("Лимит должен быть 1..5000")
    now = datetime.now(UTC)
    recipe = pipeline_hash("vacancy_analysis", model)
    async with sessions() as session:
        statement = select(VacancyModel.id)
        if missing_only:
            known = (
                select(BackgroundJob.id)
                .where(
                    BackgroundJob.vacancy_id == VacancyModel.id,
                    BackgroundJob.kind == "vacancy_analysis",
                    BackgroundJob.revision == VacancyModel.content_revision,
                    BackgroundJob.pipeline == recipe,
                    BackgroundJob.state.not_in(["cancelled", "superseded"]),
                )
                .exists()
            )
            statement = statement.where(
                ~known,
                VacancyModel.archived.is_(False),
                VacancyModel.published_at <= now,
                or_(VacancyModel.valid_until.is_(None), VacancyModel.valid_until > now),
            )
        keys = list(
            (
                await session.scalars(
                    statement.order_by(
                        VacancyModel.published_at.asc()
                        if missing_only
                        else VacancyModel.published_at.desc(),
                        VacancyModel.source_id,
                    ).limit(limit)
                )
            ).all()
        )
    count = 0
    for key in keys:
        async with sessions.begin() as session:
            row = await session.scalar(
                select(VacancyModel)
                .options(
                    load_only(
                        VacancyModel.id,
                        VacancyModel.content_revision,
                    )
                )
                .where(
                    VacancyModel.id == key,
                    VacancyModel.archived.is_(False),
                    VacancyModel.published_at <= now,
                    or_(VacancyModel.valid_until.is_(None), VacancyModel.valid_until > now),
                )
                .with_for_update()
            )
            if row is None:
                continue
            revision = row.content_revision
            if revision is None:
                # Legacy rows are backfilled individually. Never load hundreds
                # of raw HTML payloads in memory just to schedule their IDs.
                vacancy = await VacancyRepository(session).get_by_id(key)
                revision = analysis_input_hash(vacancy)
                row.content_revision = revision
            try:
                await JobRepository(session).enqueue(
                    kind="vacancy_analysis",
                    entity_key=str(key),
                    vacancy_id=key,
                    revision=revision,
                    pipeline=recipe,
                    force=force,
                    payload={"force": force},
                )
            except QueueFull:
                break
        count += 1
    return count
