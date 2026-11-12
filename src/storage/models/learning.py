"""Private review access, artifact lineage and bounded-window serving telemetry."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import JSON, BigInteger, Boolean, DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from storage.models.vacancy import Base


class LearningAccess(Base):
    __tablename__ = "learning_access"
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("telegram_users.user_id", ondelete="CASCADE"), primary_key=True
    )
    allowed: Mapped[bool] = mapped_column(Boolean)
    consent_ref: Mapped[str] = mapped_column(String(64))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class LearningArtifact(Base):
    __tablename__ = "learning_artifacts"
    id: Mapped[UUID] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(24))
    # Owner IDs remain private. After erasure keep artifact lineage but remove owner IDs.
    owners: Mapped[list] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
    parents: Mapped[list] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
    manifest: Mapped[dict] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class SelectionObservation(Base):
    __tablename__ = "selection_observations"
    id: Mapped[UUID] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("telegram_users.user_id", ondelete="CASCADE"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    metrics: Mapped[dict] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
