"""Idempotent projection repairs for F06. Normal writes materialize in their source transaction."""

from domain.professional import ProfessionalFacts
from scoring.fact_projection import projection_contract, projection_revision
from storage.models import VacancyModel
from storage.repositories.job_repository import JobRepository
from storage.repositories.vacancy_repository import VacancyRepository


async def enqueue_projection(session, vacancy_id, *, force=False):
    row = await session.get(VacancyModel, vacancy_id, with_for_update=True, populate_existing=True)
    if row is None:
        raise ValueError("Vacancy not found")
    repo = VacancyRepository(session)
    vacancy = await repo.get_by_id(vacancy_id)
    return await JobRepository(session).enqueue(
        kind="vacancy_projection",
        entity_key=str(vacancy_id),
        vacancy_id=vacancy_id,
        revision=projection_revision(vacancy, vacancy.professional_facts),
        pipeline=projection_contract(),
        force=force,
    )


async def process_projection(sessions, job):
    async with sessions.begin() as session:
        # Same lock order as source writers and inference completion.
        row = await session.get(VacancyModel, job.vacancy_id, with_for_update=True)
        jobs = JobRepository(session)
        owned = await jobs.owned(job)
        if owned is None:
            return "changed"
        if row is None:
            await jobs.finish(owned, state="superseded", code="input_changed")
            return "changed"
        repo = VacancyRepository(session)
        vacancy = await repo.get_by_id(job.vacancy_id)
        facts = vacancy.professional_facts
        if job.pipeline != projection_contract() or job.revision != projection_revision(
            vacancy, facts
        ):
            await jobs.finish(owned, state="superseded", code="input_changed")
            await enqueue_projection(session, job.vacancy_id)
            return "changed"
        await repo.refresh_facts(job.vacancy_id)
        await jobs.finish(
            owned,
            state="succeeded",
            result={
                "facts_revision": ProfessionalFacts.model_validate(row.professional_facts).revision,
                "projection_revision": row.projection_revision,
            },
        )
    return "projected"
