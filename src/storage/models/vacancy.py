from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

SEARCH_REPAIR_PREDICATE = (
    "search_version IS NULL OR search_version <> 'lexical-v1' OR search_revision IS NULL "
    "OR content_revision IS NULL OR search_revision <> content_revision "
    "OR search_terms IS NULL OR search_document IS NULL"
)


class Base(DeclarativeBase):
    pass


class VacancyModel(Base):
    __tablename__ = "vacancies"
    __table_args__ = (
        UniqueConstraint("source", "source_id", name="uq_vacancies_source_id"),
        CheckConstraint("salary_from IS NULL OR salary_from >= 0", name="ck_salary_from"),
        CheckConstraint("salary_to IS NULL OR salary_to >= 0", name="ck_salary_to"),
        CheckConstraint(
            "salary_from IS NULL OR salary_to IS NULL OR salary_from <= salary_to",
            name="ck_salary_order",
        ),
        Index("ix_vacancies_published_at", "published_at"),
        Index("ix_vacancies_check_due", "source", "availability", "last_checked_at"),
        Index("ix_vacancies_expiry", "availability", "valid_until"),
        Index("ix_vacancies_duplicate_key", "duplicate_key"),
        Index("ix_vacancies_search_terms", "search_terms", postgresql_using="gin"),
        Index("ix_vacancies_active_date", "archived", "published_at"),
        Index(
            "ix_vacancies_search_repair",
            "id",
            postgresql_where=text(SEARCH_REPAIR_PREDICATE),
            sqlite_where=text(SEARCH_REPAIR_PREDICATE),
        ),
        CheckConstraint(
            "availability IN ('active', 'closed', 'expired', 'unknown')",
            name="ck_vacancy_availability",
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    source: Mapped[str] = mapped_column(String(50))
    source_id: Mapped[str] = mapped_column(String(255))
    url: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text)
    company: Mapped[str] = mapped_column(Text)
    company_url: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    description_html: Mapped[str | None] = mapped_column(Text)
    skills: Mapped[list] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
    salary_from: Mapped[int | None] = mapped_column(Integer)
    salary_to: Mapped[int | None] = mapped_column(Integer)
    currency: Mapped[str | None] = mapped_column(String(10))
    salary_gross: Mapped[bool | None] = mapped_column(Boolean)
    salary_period: Mapped[str | None] = mapped_column(String(50))
    salary_raw: Mapped[str | None] = mapped_column(Text)
    experience: Mapped[str | None] = mapped_column(String(100))
    employment: Mapped[str | None] = mapped_column(Text)
    schedule: Mapped[str | None] = mapped_column(Text)
    work_format: Mapped[str | None] = mapped_column(Text)
    working_hours: Mapped[str | None] = mapped_column(Text)
    location: Mapped[str | None] = mapped_column(Text)
    address: Mapped[str | None] = mapped_column(Text)
    remote: Mapped[bool | None] = mapped_column(Boolean)
    archived: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    valid_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    collected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    raw_payload: Mapped[dict] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
    analysis: Mapped[dict | None] = mapped_column(
        JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql")
    )
    content_revision: Mapped[str | None] = mapped_column(String(64))
    catalog_revision: Mapped[str | None] = mapped_column(String(64))
    revision_number: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    raw_payload_hash: Mapped[str | None] = mapped_column(String(64))
    availability: Mapped[str] = mapped_column(String(16), default="active", server_default="active")
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    check_error: Mapped[str | None] = mapped_column(String(32))
    duplicate_key: Mapped[str | None] = mapped_column(String(64))
    search_version: Mapped[str | None] = mapped_column(String(32))
    search_revision: Mapped[str | None] = mapped_column(String(64))
    search_document: Mapped[dict | None] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
    search_terms: Mapped[list | None] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
    professional_facts: Mapped[dict | None] = mapped_column(
        JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql")
    )
    facts_input_revision: Mapped[str | None] = mapped_column(String(64))
    projection_revision: Mapped[str | None] = mapped_column(String(64))
    projection_contract: Mapped[str | None] = mapped_column(String(64))
    facts_revision: Mapped[str | None] = mapped_column(String(64))
    projection_facts_revision: Mapped[str | None] = mapped_column(String(64))
    facts_status: Mapped[str | None] = mapped_column(String(16))


class VacancySnapshot(Base):
    """One immutable snapshot per distinct raw payload, including the original HTML."""

    __tablename__ = "vacancy_snapshots"
    __table_args__ = (UniqueConstraint("vacancy_id", "payload_hash", name="uq_snapshot_payload"),)

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    vacancy_id: Mapped[UUID] = mapped_column(ForeignKey("vacancies.id", ondelete="CASCADE"))
    payload_hash: Mapped[str] = mapped_column(String(64))
    raw_payload: Mapped[dict] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class VacancyFactSource(Base):
    """Immutable structured source linked to the original raw snapshot and fact input."""

    __tablename__ = "vacancy_fact_sources"
    vacancy_id: Mapped[UUID] = mapped_column(
        ForeignKey("vacancies.id", ondelete="CASCADE"), primary_key=True
    )
    revision: Mapped[str] = mapped_column(String(64), primary_key=True)
    input_revision: Mapped[str] = mapped_column(String(64), index=True)
    content_revision: Mapped[str] = mapped_column(String(64))
    raw_payload_hash: Mapped[str] = mapped_column(String(64))
    source_payload: Mapped[dict] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
