"""Durable receipt precedes the next Telegram offset acknowledgement."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import JSON, BigInteger, DateTime, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from storage.models.vacancy import Base


class TelegramUpdate(Base):
    __tablename__ = "telegram_updates"
    __table_args__ = (
        Index("ix_updates_ready", "bot_id", "ready_at", "update_id"),
        Index("ix_updates_owner", "bot_id", "owner_id", "received_at", "update_id"),
    )
    bot_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    update_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    owner_id: Mapped[int | None] = mapped_column(BigInteger)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ready_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    claim: Mapped[UUID | None]
    attempts: Mapped[int] = mapped_column(default=0)
    payload: Mapped[dict | None] = mapped_column(
        JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql")
    )
