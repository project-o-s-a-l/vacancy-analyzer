"""Durable analysis work with fenced leases and a pending resume pointer."""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "background_jobs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("dedupe_key", sa.String(64), nullable=False, unique=True),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("entity_key", sa.String(80), nullable=False),
        sa.Column("vacancy_id", sa.Uuid(), sa.ForeignKey("vacancies.id", ondelete="CASCADE")),
        sa.Column(
            "user_id", sa.BigInteger(), sa.ForeignKey("telegram_users.user_id", ondelete="CASCADE")
        ),
        sa.Column("revision", sa.String(64), nullable=False),
        sa.Column("pipeline", sa.String(64), nullable=False),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("next_run_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("owner", sa.String(64)),
        sa.Column("lease_token", sa.Uuid()),
        sa.Column("lease_until", sa.DateTime(timezone=True)),
        sa.Column("error_code", sa.String(64)),
        sa.Column(
            "payload",
            sa.JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql"),
        ),
        sa.Column(
            "result",
            sa.JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("kind IN ('vacancy_analysis', 'resume_analysis')", name="ck_job_kind"),
        sa.CheckConstraint(
            "state IN ('pending', 'running', 'succeeded', 'failed', 'superseded', 'cancelled')",
            name="ck_job_state",
        ),
        sa.CheckConstraint("attempts >= 0 AND max_attempts > 0", name="ck_job_attempts"),
    )
    op.create_index(
        "ix_job_due", "background_jobs", ["state", "next_run_at", "priority", "created_at"]
    )
    op.create_index("ix_job_user", "background_jobs", ["user_id"])
    op.add_column("user_resumes", sa.Column("processing_job_id", sa.Uuid()))
    op.add_column("vacancies", sa.Column("content_revision", sa.String(64)))


def downgrade():
    op.drop_column("vacancies", "content_revision")
    op.drop_column("user_resumes", "processing_job_id")
    op.drop_index("ix_job_user", table_name="background_jobs")
    op.drop_index("ix_job_due", table_name="background_jobs")
    op.drop_table("background_jobs")
