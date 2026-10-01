"""One bounded, replaceable scoring pool per owner; independent of deliveries."""

from datetime import datetime

from sqlalchemy import JSON, BigInteger, DateTime, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from storage.models.vacancy import Base


class SavedSelectionModel(Base):
    __tablename__ = "saved_selections"

    owner_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    request_revision: Mapped[str] = mapped_column(String(64))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    context: Mapped[dict] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
    entries: Mapped[list] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
