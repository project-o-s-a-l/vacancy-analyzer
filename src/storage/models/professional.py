"""Append-only fact revisions and a transactionally advanced bounded backfill cursor."""

from datetime import datetime

from sqlalchemy import JSON, DateTime, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from storage.models.vacancy import Base


class FactRevisionModel(Base):
    __tablename__ = "professional_fact_revisions"

    subject_kind: Mapped[str] = mapped_column(String(16), primary_key=True)
    subject_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    revision: Mapped[str] = mapped_column(String(64), primary_key=True)
    input_revision: Mapped[str] = mapped_column(String(64), index=True)
    schema_version: Mapped[str] = mapped_column(String(40))
    payload: Mapped[dict] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class FactBackfillModel(Base):
    __tablename__ = "professional_fact_backfills"

    name: Mapped[str] = mapped_column(String(80), primary_key=True)
    cursor: Mapped[dict] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
