from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from storage.models.exposure import confirmed_context
from storage.models.vacancy import Base


class FeedbackModel(Base):
    __tablename__ = "vacancy_feedback"
    __table_args__ = (
        CheckConstraint(
            "interest IS NULL OR interest IN ('like','dislike')", name="ck_feedback_interest"
        ),
        CheckConstraint(
            "reason IS NULL OR (interest IS NOT NULL AND interest = 'dislike')",
            name="ck_feedback_reason_interest",
        ),
        CheckConstraint(
            "reason IS NULL OR reason IN ('salary','skills','experience','location',"
            "'company','duties','other')",
            name="ck_feedback_reason",
        ),
        CheckConstraint("version > 0", name="ck_feedback_version"),
    )
    user_id: Mapped[int] = mapped_column(
        ForeignKey("telegram_users.user_id", ondelete="CASCADE"), primary_key=True
    )
    vacancy_id: Mapped[UUID] = mapped_column(
        ForeignKey("vacancies.id", ondelete="CASCADE"), primary_key=True
    )
    interest: Mapped[str | None] = mapped_column(String(10))
    saved: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    applied: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    reason: Mapped[str | None] = mapped_column(String(16))
    version: Mapped[int] = mapped_column(Integer)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class FeedbackEventModel(Base):
    __tablename__ = "vacancy_feedback_events"
    __table_args__ = (
        CheckConstraint(
            "action IN ('like','dislike','save','unsave','apply','unapply','reason')",
            name="ck_feedback_event_action",
        ),
        CheckConstraint(
            "reason IS NULL OR reason IN ('salary','skills','experience','location',"
            "'company','duties','other')",
            name="ck_feedback_event_reason",
        ),
    )
    user_id: Mapped[int] = mapped_column(
        ForeignKey("telegram_users.user_id", ondelete="CASCADE"), primary_key=True
    )
    event_key: Mapped[str] = mapped_column(String(128), primary_key=True)
    vacancy_id: Mapped[UUID] = mapped_column(
        ForeignKey("vacancies.id", ondelete="CASCADE"), index=True
    )
    action: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str | None] = mapped_column(String(16))
    snapshot_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("recommendation_exposures.id", name="fk_feedback_snapshot", ondelete="SET NULL")
    )
    legacy_context: Mapped[dict | None] = mapped_column(
        "exposure_context",
        JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql"),
    )
    exposure_context = confirmed_context(legacy_context, snapshot_id, user_id, vacancy_id)
    action_context: Mapped[dict | None] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
