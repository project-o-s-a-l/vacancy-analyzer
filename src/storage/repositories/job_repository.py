"""Atomic claims and fenced completion; caller commits short transactions."""

import logging
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from uuid import uuid4

from sqlalchemy import and_, case, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from observability.transactions import after_commit
from storage.models import BackgroundJob

TERMINAL = {"succeeded", "failed", "superseded", "cancelled"}


class JobRepository:
    def __init__(self, session):
        self.session = session
        self.insert = (
            pg_insert if session.get_bind().dialect.name == "postgresql" else sqlite_insert
        )

    def _event(self, name, row, *, code=None, **fields):
        after_commit(
            self.session,
            name,
            level=logging.ERROR
            if row.state == "failed"
            else logging.WARNING
            if name == "job.retry_scheduled"
            else logging.INFO,
            job_id=row.id,
            job_kind=row.kind,
            job_state=row.state,
            attempt=row.attempts,
            max_attempts=row.max_attempts,
            vacancy_id=row.vacancy_id,
            pipeline=row.pipeline,
            code=code,
            **fields,
        )

    async def _bulk_facts(self, condition, *, state, code=None, pipeline=None):
        rows = list(
            (
                await self.session.execute(
                    select(
                        BackgroundJob.id,
                        BackgroundJob.kind,
                        BackgroundJob.attempts,
                        BackgroundJob.vacancy_id,
                        BackgroundJob.pipeline,
                    )
                    .where(condition)
                    .order_by(BackgroundJob.id)
                    .with_for_update()
                    .limit(65)
                )
            ).all()
        )
        for row in rows[:64]:
            after_commit(
                self.session,
                "job.finished",
                level=logging.ERROR if state == "failed" else logging.INFO,
                job_id=row.id,
                job_kind=row.kind,
                job_state=state,
                attempt=row.attempts,
                vacancy_id=row.vacancy_id,
                pipeline=row.pipeline,
                result=state if state != "succeeded" else "success",
                code="pipeline_changed" if pipeline and row.pipeline != pipeline else code,
            )
        if len(rows) > 64:
            after_commit(self.session, "job.bulk_changed", count=64, job_state=state)

    async def enqueue(
        self,
        *,
        kind,
        entity_key,
        revision,
        pipeline,
        vacancy_id=None,
        user_id=None,
        payload=None,
        force=False,
    ):
        now = datetime.now(UTC)
        key = sha256(f"{kind}:{entity_key}:{revision}:{pipeline}".encode()).hexdigest()
        obsolete = and_(
            BackgroundJob.kind == kind,
            BackgroundJob.entity_key == entity_key,
            BackgroundJob.dedupe_key != key,
            BackgroundJob.state.in_(["pending", "running"]),
        )
        await self._bulk_facts(
            obsolete, state="superseded", code="input_changed", pipeline=pipeline
        )
        # A newer source/request/pipeline invalidates queued or in-flight old work.
        await self.session.execute(
            update(BackgroundJob)
            .where(
                BackgroundJob.kind == kind,
                BackgroundJob.entity_key == entity_key,
                BackgroundJob.dedupe_key != key,
                BackgroundJob.state.in_(["pending", "running"]),
            )
            .values(
                state="superseded",
                payload=None,
                updated_at=now,
                error_code=case(
                    (BackgroundJob.pipeline != pipeline, "pipeline_changed"), else_="input_changed"
                ),
            )
        )
        statement = (
            self.insert(BackgroundJob)
            .values(
                id=uuid4(),
                dedupe_key=key,
                kind=kind,
                entity_key=entity_key,
                revision=revision,
                pipeline=pipeline,
                vacancy_id=vacancy_id,
                user_id=user_id,
                state="pending",
                priority=10 if kind == "resume_analysis" else 50,
                attempts=0,
                max_attempts=3,
                next_run_at=now,
                payload=payload,
                created_at=now,
                updated_at=now,
            )
            .on_conflict_do_nothing(index_elements=["dedupe_key"])
        )
        inserted = (
            await self.session.execute(statement.returning(BackgroundJob.id))
        ).scalar_one_or_none()
        row = await self.session.scalar(
            select(BackgroundJob)
            .where(BackgroundJob.dedupe_key == key)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        reopened = (
            row.state in {"succeeded", "superseded", "cancelled"} or force and row.state in TERMINAL
        )
        if reopened:
            row.state, row.attempts, row.next_run_at = "pending", 0, now
            row.payload, row.error_code, row.result = payload, None, None
            row.updated_at = now
        self._event(
            "job.enqueued" if inserted or reopened else "job.reused",
            row,
            next_run_at=row.next_run_at,
            result="ready"
            if row.state == "pending"
            else "deferred"
            if row.state == "running"
            else "skipped",
        )
        return row

    async def claim(self, owner, *, lease_seconds=240, now=None, kinds=None, job_id=None):
        now = now or datetime.now(UTC)
        # A crashed client might leave Ollama generating. Do not open another slot
        # until its bounded request deadline, even after its PG session lock vanished.
        active = await self.session.scalar(
            select(BackgroundJob.id)
            .where(
                BackgroundJob.lease_until > now,
                BackgroundJob.kind != "vacancy_projection",
            )
            .limit(1)
        )
        if active and kinds != ["vacancy_projection"]:
            return None
        ready = or_(
            and_(BackgroundJob.state == "pending", BackgroundJob.next_run_at <= now),
            and_(BackgroundJob.state == "running", BackgroundJob.lease_until <= now),
        )
        await self._bulk_facts(
            and_(ready, BackgroundJob.attempts >= BackgroundJob.max_attempts),
            state="failed",
            code="attempts_exhausted",
        )
        # Exhausted crashed attempts become terminal instead of getting a fourth request.
        await self.session.execute(
            update(BackgroundJob)
            .where(
                ready,
                BackgroundJob.attempts >= BackgroundJob.max_attempts,
            )
            .values(state="failed", error_code="attempts_exhausted", payload=None, updated_at=now)
        )
        statement = select(BackgroundJob.id, BackgroundJob.state).where(
            ready,
            BackgroundJob.attempts < BackgroundJob.max_attempts,
        )
        if kinds is not None:
            statement = statement.where(BackgroundJob.kind.in_(kinds))
        if job_id is not None:
            statement = statement.where(BackgroundJob.id == job_id)
        selected = (
            await self.session.execute(
                statement.order_by(
                    BackgroundJob.priority,
                    BackgroundJob.created_at,
                    BackgroundJob.id,
                )
                .with_for_update(skip_locked=True)
                .limit(1)
            )
        ).one_or_none()
        if selected is None:
            return None
        token = uuid4()
        row = (
            await self.session.execute(
                update(BackgroundJob)
                .where(
                    BackgroundJob.id == selected.id,
                    ready,
                )
                .values(
                    state="running",
                    owner=owner,
                    lease_token=token,
                    lease_until=now + timedelta(seconds=lease_seconds),
                    attempts=BackgroundJob.attempts + 1,
                    updated_at=now,
                )
                .returning(BackgroundJob)
            )
        ).scalar_one_or_none()
        if row is not None:
            self._event(
                "job.lease_recovered" if selected.state == "running" else "job.claimed",
                row,
                result="ready",
            )
        return row

    async def owned(self, job, *, now=None):
        now = now or datetime.now(UTC)
        return await self.session.scalar(
            select(BackgroundJob)
            .where(
                BackgroundJob.id == job.id,
                BackgroundJob.state == "running",
                BackgroundJob.lease_token == job.lease_token,
                BackgroundJob.owner == job.owner,
                BackgroundJob.lease_until > now,
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )

    async def finish(self, row, *, state="succeeded", code=None, result=None):
        row.state, row.error_code, row.result = state, code, result
        row.payload = None
        row.lease_until = row.lease_token = row.owner = None
        row.updated_at = datetime.now(UTC)
        self._event(
            "job.finished", row, code=code, result="success" if state == "succeeded" else state
        )

    async def retry(self, row, code):
        deadline, owner, token = row.lease_until, row.owner, row.lease_token
        if row.attempts >= row.max_attempts:
            await self.finish(row, state="failed", code=code)
            row.lease_until, row.owner, row.lease_token = deadline, owner, token
            return
        row.state, row.error_code = "pending", code
        row.next_run_at = datetime.now(UTC) + timedelta(seconds=30 * 4 ** (row.attempts - 1))
        # A timed-out HTTP connection does not prove Ollama stopped generating.
        # Preserve the global cooldown until the original request deadline.
        row.updated_at = datetime.now(UTC)
        self._event(
            "job.retry_scheduled", row, code=code, next_run_at=row.next_run_at, result="deferred"
        )

    async def cancel_user(self, user_id):
        # Remove even terminal private input. Retain a live request's deadline to
        # prevent a replacement worker from opening a second inference slot.
        await self._bulk_facts(
            and_(BackgroundJob.user_id == user_id, BackgroundJob.state != "cancelled"),
            state="cancelled",
            code="user_cancelled",
        )
        await self.session.execute(
            update(BackgroundJob)
            .where(
                BackgroundJob.user_id == user_id,
            )
            .values(state="cancelled", payload=None, result=None, updated_at=datetime.now(UTC))
        )

    async def release(self, job):
        # An obsolete/cancelled request has now actually returned; its cooldown
        # may be removed, but only by the exact claimant (ABA fencing).
        await self.session.execute(
            update(BackgroundJob)
            .where(
                BackgroundJob.id == job.id,
                BackgroundJob.lease_token == job.lease_token,
                BackgroundJob.owner == job.owner,
                BackgroundJob.state.in_(["superseded", "cancelled"]),
            )
            .values(lease_until=None, lease_token=None, owner=None)
        )
