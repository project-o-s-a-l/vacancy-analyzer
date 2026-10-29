"""Persist only validated blocks, under the same source/generation fence as final writes."""

from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from nlp.inference_metrics import stage
from storage.models import ExtractionCache, ResumeModel, VacancyModel
from storage.repositories.job_repository import JobRepository
from storage.repositories.user_repository import UserRepository


class ValidatedCache:
    def __init__(self, sessions, *, job=None, user_id=None, generation=None, profile_revision=None):
        self.sessions, self.job = sessions, job
        self.user_id = user_id if user_id is not None else job.user_id
        self.vacancy_id = job.vacancy_id if job else None
        self.generation, self.profile_revision = generation, profile_revision
        self.scope = (
            f"resume:{self.user_id}" if self.user_id is not None else f"vacancy:{self.vacancy_id}"
        )

    async def get(self, key):
        if self.job and (self.job.payload or {}).get("force"):
            return None
        async with self.sessions() as session:
            return await session.scalar(
                select(ExtractionCache.value).where(
                    ExtractionCache.scope == self.scope,
                    ExtractionCache.key == key,
                    ExtractionCache.expires_at > datetime.now(UTC),
                )
            )

    async def put(self, key, value):
        now = datetime.now(UTC)
        with stage("cache_write"):
            async with self.sessions.begin() as session:
                if self.user_id is not None:
                    user = await UserRepository(session).get(self.user_id, lock=True)
                    resume = await session.get(ResumeModel, self.user_id)
                    if (
                        user is None
                        or resume is None
                        or (
                            self.profile_revision is not None
                            and user.revision != self.profile_revision
                        )
                    ):
                        return
                    if self.job:
                        if resume.processing_job_id != self.job.id:
                            return
                    elif (resume.draft or {}).get("preparing_token") != self.generation:
                        return
                else:
                    vacancy = await session.get(VacancyModel, self.vacancy_id, with_for_update=True)
                    if vacancy is None or vacancy.content_revision != self.job.revision:
                        return
                if self.job and await JobRepository(session).owned(self.job) is None:
                    return
                insert = (
                    pg_insert if session.get_bind().dialect.name == "postgresql" else sqlite_insert
                )
                await session.execute(
                    insert(ExtractionCache)
                    .values(
                        scope=self.scope,
                        key=key,
                        user_id=self.user_id,
                        vacancy_id=self.vacancy_id,
                        value=value,
                        created_at=now,
                        expires_at=now + timedelta(days=7),
                    )
                    .on_conflict_do_update(
                        index_elements=["scope", "key"],
                        set_={
                            "value": value,
                            "created_at": now,
                            "expires_at": now + timedelta(days=7),
                        },
                    )
                )
                # Per-owner bounds prevent unbounded private history. Shared cache is
                # bounded by catalog size ×128 blocks; expired rows are reclaimed in batches.
                obsolete = (
                    select(ExtractionCache.key)
                    .where(ExtractionCache.scope == self.scope)
                    .order_by(ExtractionCache.created_at.desc(), ExtractionCache.key)
                    .offset(128)
                )
                await session.execute(
                    delete(ExtractionCache).where(
                        ExtractionCache.scope == self.scope, ExtractionCache.key.in_(obsolete)
                    )
                )


async def prune_cache(sessions):
    async with sessions.begin() as session:
        from sqlalchemy import tuple_

        keys = (
            select(ExtractionCache.scope, ExtractionCache.key)
            .where(ExtractionCache.expires_at <= datetime.now(UTC))
            .limit(128)
        )
        await session.execute(
            delete(ExtractionCache).where(
                tuple_(ExtractionCache.scope, ExtractionCache.key).in_(keys)
            )
        )
