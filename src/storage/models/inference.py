"""Durable scheduling gate and validated extraction cache; no model weights in workers."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import JSON, BigInteger, DateTime, ForeignKey, Index, Integer, String, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from storage.models.vacancy import Base


class InferenceGate(Base):
    __tablename__ = "inference_gate"
    name: Mapped[str] = mapped_column(String(32), primary_key=True)
    resume_streak: Mapped[int] = mapped_column(Integer, default=0)
    token: Mapped[UUID | None] = mapped_column(Uuid)
    until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    runtime: Mapped[dict | None] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))


class ExtractionCache(Base):
    __tablename__ = "extraction_cache"
    __table_args__ = (Index("ix_extraction_cache_expiry", "expires_at"),)
    scope: Mapped[str] = mapped_column(String(80), primary_key=True)
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("telegram_users.user_id", ondelete="CASCADE")
    )
    vacancy_id: Mapped[UUID | None] = mapped_column(
        Uuid, ForeignKey("vacancies.id", ondelete="CASCADE")
    )
    value: Mapped[dict] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
