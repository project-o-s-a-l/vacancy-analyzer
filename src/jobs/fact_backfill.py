"""Bounded keyset backfill. Data and durable cursor commit together; retries are harmless."""

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from domain.professional import PROFILE_SCHEMA
from domain.resume import ResumeDraft
from jobs.resumes import profile_hash
from nlp.professional_facts import profile_input_revision, resume_facts
from storage.models import FactBackfillModel, ResumeModel, TelegramUserModel, VacancyModel
from storage.repositories.fact_repository import FactRepository
from storage.repositories.user_repository import read_profile
from storage.repositories.vacancy_repository import VacancyRepository

BACKFILL_NAME = "f02-professional-facts-v1"


async def backfill_batch(sessions, *, batch_size=100, restart=False):
    if not 1 <= batch_size <= 500:
        raise ValueError("Backfill batch must be in 1..500")
    counts = dict(users=0, vacancies=0, drafts=0)
    async with sessions.begin() as session:
        insert = pg_insert if session.get_bind().dialect.name == "postgresql" else sqlite_insert
        await session.execute(
            insert(FactBackfillModel)
            .values(
                name=BACKFILL_NAME,
                cursor={"stage": "users", "after": None},
                updated_at=datetime.now(UTC),
            )
            .on_conflict_do_nothing(index_elements=["name"])
        )
        checkpoint = await session.scalar(
            select(FactBackfillModel)
            .where(FactBackfillModel.name == BACKFILL_NAME)
            .with_for_update()
        )
        cursor = {"stage": "users", "after": None} if restart else dict(checkpoint.cursor)
        stage, after = cursor["stage"], cursor["after"]
        if stage == "users":
            statement = select(TelegramUserModel).order_by(TelegramUserModel.user_id)
            if after is not None:
                statement = statement.where(TelegramUserModel.user_id > int(after))
            rows = list(await session.scalars(statement.limit(batch_size).with_for_update()))
            for row in rows:
                profile = read_profile(row)
                if row.profile_schema_version not in {None, PROFILE_SCHEMA}:
                    raise ValueError("Unsupported profile schema in backfill")
                row.profile_schema_version = PROFILE_SCHEMA
                facts = profile.professional_facts
                if row.professional_facts != facts.model_dump(mode="json"):
                    row.professional_facts = facts.model_dump(mode="json")
                    row.facts_input_revision = profile_input_revision(profile)
                    counts["users"] += 1
                await FactRepository(session).record("candidate", row.user_id, facts)
                resume = await session.get(ResumeModel, row.user_id)
                if (
                    resume
                    and resume.draft
                    and "extraction" in resume.draft
                    and not resume.draft.get("professional_facts")
                ):
                    draft = ResumeDraft.model_validate(resume.draft)
                    draft.professional_facts = resume_facts(
                        draft.text, draft.extraction, row.user_id, version=draft.version
                    )
                    if draft.profile_hash == profile_hash(profile):
                        draft.profile_revision = row.profile_revision
                    await FactRepository(session).record(
                        "resume_proposal", row.user_id, draft.professional_facts
                    )
                    resume.draft = draft.model_dump(mode="json")
                    counts["drafts"] += 1
            cursor = (
                {"stage": "users", "after": rows[-1].user_id}
                if len(rows) == batch_size
                else {"stage": "vacancies", "after": None}
            )
        elif stage == "vacancies":
            statement = select(VacancyModel.id).order_by(VacancyModel.id)
            if after is not None:
                statement = statement.where(VacancyModel.id > UUID(after))
            keys = list(await session.scalars(statement.limit(batch_size)))
            for key in keys:
                await VacancyRepository(session).refresh_facts(key)
                counts["vacancies"] += 1
            cursor = (
                {"stage": "vacancies", "after": str(keys[-1])}
                if len(keys) == batch_size
                else {"stage": "done", "after": None}
            )
        checkpoint.cursor = cursor
        checkpoint.updated_at = datetime.now(UTC)
    return dict(**counts, cursor=cursor, done=cursor["stage"] == "done")
