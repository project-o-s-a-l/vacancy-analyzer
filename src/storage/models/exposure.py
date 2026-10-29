"""Immutable confirmed message snapshots; retries/reset never overwrite earlier shows."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import JSON, BigInteger, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from storage.models.vacancy import Base


class ExposureModel(Base):
    __tablename__ = "recommendation_exposures"
    __table_args__ = (UniqueConstraint("chat_id", "message_id", name="uq_exposure_message"),)
    id: Mapped[UUID] = mapped_column(primary_key=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, index=True)
    vacancy_id: Mapped[UUID] = mapped_column(ForeignKey("vacancies.id", ondelete="CASCADE"))
    message_id: Mapped[int] = mapped_column(BigInteger)
    confirmed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    context: Mapped[dict | None] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
