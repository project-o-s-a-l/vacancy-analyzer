"""Durable claims ensure competing workers cannot send the same recommendation."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    false,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from storage.models.vacancy import Base


class NotificationModel(Base):
    __tablename__ = "telegram_notifications"
    __table_args__ = (
        Index("ix_notifications_duplicate", "chat_id", "duplicate_key", "status"),
        Index(
            "uq_notifications_awaiting_rating",
            "chat_id",
            unique=True,
            postgresql_where=text("awaiting_rating IS TRUE"),
            sqlite_where=text("awaiting_rating IS 1"),
        ),
        CheckConstraint(
            "status IN ('sending', 'sent', 'failed', 'uncertain')", name="ck_notification_status"
        ),
    )

    chat_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    vacancy_id: Mapped[UUID] = mapped_column(
        ForeignKey("vacancies.id", ondelete="CASCADE"), primary_key=True
    )
    claim_token: Mapped[UUID]
    status: Mapped[str] = mapped_column(String(20))
    awaiting_rating: Mapped[bool] = mapped_column(Boolean, server_default=false())
    attempted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    history_reset_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    message_id: Mapped[int | None] = mapped_column(BigInteger)
    attempts: Mapped[int] = mapped_column(Integer, default=1)
    error_kind: Mapped[str | None] = mapped_column(String(100))
    duplicate_key: Mapped[str | None] = mapped_column(String(64))
    exposure_context: Mapped[dict | None] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql")
    )
