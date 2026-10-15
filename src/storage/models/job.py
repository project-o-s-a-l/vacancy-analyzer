"""Durable, fenced work items. Resume payloads are scrubbed on terminal states."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    JSON,
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from storage.models.vacancy import Base


class BackgroundJob(Base):
    __tablename__ = "background_jobs"
    __table_args__ = (
        CheckConstraint(
            "kind IN ('vacancy_analysis', 'resume_analysis', 'vacancy_projection')",
            name="ck_job_kind",
        ),
        CheckConstraint(
            "state IN ('pending', 'running', 'succeeded', 'failed', 'superseded', 'cancelled')",
            name="ck_job_state",
        ),
        CheckConstraint("attempts >= 0 AND max_attempts > 0", name="ck_job_attempts"),
        Index("ix_job_due", "state", "next_run_at", "priority", "created_at"),
        Index("ix_job_user", "user_id"),
    )
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    dedupe_key: Mapped[str] = mapped_column(String(64), unique=True)
    kind: Mapped[str] = mapped_column(String(32))
    entity_key: Mapped[str] = mapped_column(String(80))
    vacancy_id: Mapped[UUID | None] = mapped_column(
        Uuid, ForeignKey("vacancies.id", ondelete="CASCADE")
    )
    user_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("telegram_users.user_id", ondelete="CASCADE")
    )
    revision: Mapped[str] = mapped_column(String(64))
    pipeline: Mapped[str] = mapped_column(String(64))
    state: Mapped[str] = mapped_column(String(16))
    priority: Mapped[int] = mapped_column(Integer)
    attempts: Mapped[int] = mapped_column(Integer)
    max_attempts: Mapped[int] = mapped_column(Integer)
    next_run_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    owner: Mapped[str | None] = mapped_column(String(64))
    lease_token: Mapped[UUID | None] = mapped_column(Uuid)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(64))
    payload: Mapped[dict | None] = mapped_column(
        JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql")
    )
    result: Mapped[dict | None] = mapped_column(
        JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
