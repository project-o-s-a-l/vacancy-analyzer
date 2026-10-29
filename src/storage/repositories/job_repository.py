"""Atomic claims and fenced completion; caller commits short transactions."""

import logging
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from uuid import uuid4

from sqlalchemy import and_, case, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from observability.transactions import after_commit
from storage.models import BackgroundJob, InferenceGate

TERMINAL = {"succeeded", "failed", "superseded", "cancelled"}


class JobRepository:
    def __init__(self, session):
        self.session = session
        self.insert = (
            pg_insert if session.get_bind().dialect.name == "postgresql" else sqlite_insert
        )

    async def gate(self):
        await self.session.execute(
            self.insert(InferenceGate)
            .values(name="global", resume_streak=0)
            .on_conflict_do_nothing(index_elements=["name"])
        )
        return await self.session.scalar(
            select(InferenceGate)
            .where(InferenceGate.name == "global")
            .with_for_update()
            .execution_options(populate_existing=True)
        )

    async def clear_gate(self, token):
        await self.session.execute(
            update(InferenceGate)
            .where(InferenceGate.name == "global", InferenceGate.token == token)
            .values(token=None, until=None, runtime=None)
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
        from jobs.queue_policy import QueueFull, SubmissionLimited, queue_settings

        limits = queue_settings()
        # Serialize admission across all producers, including separate containers.
        # Entity/user locks are always acquired before this gate, never in reverse.
        await self.gate()
        existing = await self.session.scalar(
            select(BackgroundJob).where(BackgroundJob.dedupe_key == key)
        )
        if existing is None or existing.state in TERMINAL:
            if user_id is not None:
                recent = await self.session.scalar(
                    select(func.count())
                    .select_from(BackgroundJob)
                    .where(
                        BackgroundJob.user_id == user_id,
                        BackgroundJob.created_at > now - timedelta(minutes=1),
                    )
                )
                if recent >= limits.analysis_user_per_minute:
                    raise SubmissionLimited()
            active_count = await self.session.scalar(
                select(func.count())
                .select_from(BackgroundJob)
                .where(
                    BackgroundJob.state.in_(["pending", "running"]),
                    ~and_(BackgroundJob.kind == kind, BackgroundJob.entity_key == entity_key),
                )
            )
            limit = limits.analysis_queue_limit
            if kind != "resume_analysis":
                limit -= min(limits.analysis_resume_reserve, limit - 1)
            if active_count >= limit:
                raise QueueFull()
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

    async def claim(
        self,
        owner,
        *,
        lease_seconds=240,
        now=None,
        kinds=None,
        job_id=None,
        runtime=None,
        model_work=True,
    ):
        now = now or datetime.now(UTC)
        gate = await self.gate()
        from jobs.queue_policy import queue_settings

        inference = model_work and kinds != ["vacancy_projection"]
        # Native/legacy tests with no remote runtime still retain the original lease fence.
        if inference and gate.token and gate.runtime:
            return None  # Only a verified drain may remove uncertain remote work.
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
        if active and inference:
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
                    case((BackgroundJob.kind != "resume_analysis", 0), else_=1)
                    if inference and gate.resume_streak >= queue_settings().analysis_resume_burst
                    else BackgroundJob.priority,
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
            if inference and row.kind != "vacancy_projection":
                gate.resume_streak = (
                    min(gate.resume_streak + 1, queue_settings().analysis_resume_burst)
                    if row.kind == "resume_analysis"
                    else 0
                )
                gate.token, gate.until, gate.runtime = token, row.lease_until, runtime
            elif inference and row.kind == "vacancy_projection":
                gate.resume_streak = 0
            self._event(
                "job.lease_recovered" if selected.state == "running" else "job.claimed",
                row,
                result="ready",
                wait_ms=max(
                    0,
                    (
                        now
                        - (
                            row.created_at.replace(tzinfo=UTC)
                            if row.created_at.tzinfo is None
                            else row.created_at
                        )
                    ).total_seconds()
                    * 1000,
                ),
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

    async def release(self, job, *, remote_settled=False):
        # An obsolete/cancelled request has now actually returned; its cooldown
        # may be removed, but only by the exact claimant (ABA fencing).
        await self.session.execute(
            update(BackgroundJob)
            .where(
                BackgroundJob.id == job.id,
                BackgroundJob.lease_token == job.lease_token,
                BackgroundJob.owner == job.owner,
                BackgroundJob.state.in_(
                    ["superseded", "cancelled", "pending", "failed", "succeeded"]
                    if remote_settled
                    else ["superseded", "cancelled"]
                ),
            )
            .values(lease_until=None, lease_token=None, owner=None)
        )
