from datetime import datetime
from uuid import UUID

from sqlalchemy import JSON, BigInteger, DateTime, ForeignKey, String, Text, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from storage.models.vacancy import Base


class ResumeModel(Base):
    __tablename__ = "user_resumes"

    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("telegram_users.user_id", ondelete="CASCADE"), primary_key=True
    )
    text: Mapped[str | None] = mapped_column(Text)
    parser_version: Mapped[str | None] = mapped_column(String(40))
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    draft: Mapped[dict | None] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
    processing_job_id: Mapped[UUID | None] = mapped_column(Uuid)


class ResumeSourceModel(Base):
    """Immutable normalized texts of confirmed imports, addressed by content revision."""

    __tablename__ = "resume_sources"
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("telegram_users.user_id", ondelete="CASCADE"), primary_key=True
    )
    revision: Mapped[str] = mapped_column(String(64), primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
