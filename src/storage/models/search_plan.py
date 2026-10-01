from datetime import datetime

from sqlalchemy import JSON, BigInteger, DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from storage.models.vacancy import Base


class SearchPlanModel(Base):
    """Latest profile-to-query mapping; removed together with its owner."""

    __tablename__ = "search_plans"
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("telegram_users.user_id", ondelete="CASCADE"), primary_key=True
    )
    version: Mapped[str] = mapped_column(String(32))
    plan_hash: Mapped[str] = mapped_column(String(64))
    queries: Mapped[list] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
