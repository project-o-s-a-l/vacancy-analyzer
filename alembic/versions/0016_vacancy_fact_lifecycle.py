"""Vacancy source provenance and explicit projection freshness; no source rewrites."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0016"
down_revision = "0015"
branch_labels = None
depends_on = None


def upgrade():
    for name in ("facts_revision", "projection_facts_revision"):
        op.add_column("vacancies", sa.Column(name, sa.String(64)))
    op.add_column("vacancies", sa.Column("facts_status", sa.String(16)))
    op.create_table(
        "vacancy_fact_sources",
        sa.Column(
            "vacancy_id",
            sa.Uuid(),
            sa.ForeignKey("vacancies.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("revision", sa.String(64), primary_key=True),
        sa.Column("input_revision", sa.String(64), nullable=False),
        sa.Column("content_revision", sa.String(64), nullable=False),
        sa.Column("raw_payload_hash", sa.String(64), nullable=False),
        sa.Column(
            "source_payload",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=False,
        ),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_vacancy_fact_sources_input_revision", "vacancy_fact_sources", ["input_revision"]
    )
    with op.batch_alter_table("background_jobs") as batch:
        batch.drop_constraint("ck_job_kind", type_="check")
        batch.create_check_constraint(
            "ck_job_kind", "kind IN ('vacancy_analysis', 'resume_analysis', 'vacancy_projection')"
        )


def downgrade():
    # Operational rollback keeps 0016; destructive downgrade only on an exported copy.
    op.execute("DELETE FROM background_jobs WHERE kind = 'vacancy_projection'")
    with op.batch_alter_table("background_jobs") as batch:
        batch.drop_constraint("ck_job_kind", type_="check")
        batch.create_check_constraint(
            "ck_job_kind", "kind IN ('vacancy_analysis', 'resume_analysis')"
        )
    op.drop_table("vacancy_fact_sources")
    for name in ("facts_revision", "projection_facts_revision", "facts_status"):
        op.drop_column("vacancies", name)
