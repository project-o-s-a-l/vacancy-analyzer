from datetime import datetime

from sqlalchemy import JSON, BigInteger, DateTime, String
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
