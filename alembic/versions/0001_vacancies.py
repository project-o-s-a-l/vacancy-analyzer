"""Vacancies and immutable raw-payload versions."""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    json_type = sa.JSON().with_variant(JSONB(), "postgresql")
    op.create_table(
        "vacancies",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column("source_id", sa.String(255), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("company", sa.Text(), nullable=False),
        sa.Column("description", sa.Text()),
        sa.Column("skills", json_type, nullable=False),
        sa.Column("salary_from", sa.Integer()),
        sa.Column("salary_to", sa.Integer()),
        sa.Column("currency", sa.String(10)),
        sa.Column("salary_gross", sa.Boolean()),
        sa.Column("salary_period", sa.String(50)),
        sa.Column("experience", sa.String(100)),
        sa.Column("location", sa.Text()),
        sa.Column("remote", sa.Boolean()),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("collected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("raw_payload", json_type, nullable=False),
        sa.UniqueConstraint("source", "source_id", name="uq_vacancies_source_id"),
        sa.CheckConstraint("salary_from IS NULL OR salary_from >= 0", name="ck_salary_from"),
        sa.CheckConstraint("salary_to IS NULL OR salary_to >= 0", name="ck_salary_to"),
        sa.CheckConstraint(
            "salary_from IS NULL OR salary_to IS NULL OR salary_from <= salary_to",
            name="ck_salary_order",
        ),
    )
    op.create_index("ix_vacancies_published_at", "vacancies", ["published_at"])
    op.create_table(
        "vacancy_snapshots",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "vacancy_id",
            sa.Uuid(),
            sa.ForeignKey("vacancies.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("payload_hash", sa.String(64), nullable=False),
        sa.Column("raw_payload", json_type, nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("vacancy_id", "payload_hash", name="uq_snapshot_payload"),
    )


def downgrade() -> None:
    op.drop_table("vacancy_snapshots")
    op.drop_index("ix_vacancies_published_at", table_name="vacancies")
    op.drop_table("vacancies")
