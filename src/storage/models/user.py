from datetime import datetime

from sqlalchemy import JSON, BigInteger, DateTime, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from storage.models.vacancy import Base


class TelegramUserModel(Base):
    __tablename__ = "telegram_users"

    user_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    profile: Mapped[dict] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
    form_state: Mapped[str | None] = mapped_column(String(40))
    history_reset_token: Mapped[str | None] = mapped_column(String(32))
    history_reset_before: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    profile_revision: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    profile_schema_version: Mapped[str | None] = mapped_column(String(40))
    professional_facts: Mapped[dict | None] = mapped_column(
        JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql")
    )
    facts_input_revision: Mapped[str | None] = mapped_column(String(64))


class ProfileEditModel(Base):
    """One durable, private working copy; never used by matching before save."""

    __tablename__ = "profile_edits"
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("telegram_users.user_id", ondelete="CASCADE"), primary_key=True
    )
    token: Mapped[str] = mapped_column(String(32))
    base_revision: Mapped[int] = mapped_column(Integer)
    payload: Mapped[dict] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
