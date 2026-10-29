"""Durable crawl progress and budget; no profile owner information in metrics."""

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from storage.models.vacancy import Base


class CollectionDirection(Base):
    __tablename__ = "collection_directions"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    adapter: Mapped[str] = mapped_column(String(64))
    active: Mapped[bool] = mapped_column(Boolean)
    demand: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_due_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    # Query identity is hashed; query text is provided by current plans, not copied here.
    progress: Mapped[dict] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))


class CollectionObservation(Base):
    __tablename__ = "collection_observations"
    __table_args__ = (Index("ix_collection_pending", "direction", "pending", "retry_at"),)
    direction: Mapped[str] = mapped_column(
        ForeignKey("collection_directions.key", ondelete="CASCADE"), primary_key=True
    )
    source_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    fingerprint: Mapped[str | None] = mapped_column(String(64))
    pending: Mapped[bool] = mapped_column(Boolean)
    attempts: Mapped[int] = mapped_column(Integer)
    retry_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class CollectionBudget(Base):
    __tablename__ = "collection_budgets"
    source: Mapped[str] = mapped_column(String(50), primary_key=True)
    window_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    search_used: Mapped[int] = mapped_column(Integer)
    detail_used: Mapped[int] = mapped_column(Integer)
    next_request_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    blocked: Mapped[bool] = mapped_column(Boolean)
    error: Mapped[str | None] = mapped_column(String(64))
    metrics: Mapped[dict] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
