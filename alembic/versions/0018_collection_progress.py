"""Persist bounded collection progress, budgets and successful freshness separately."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0018"
down_revision = "0017"
branch_labels = None
depends_on = None


def upgrade():
    json = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
    op.create_table(
        "collection_directions",
        sa.Column("key", sa.String(64), primary_key=True),
        sa.Column("adapter", sa.String(64), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("demand", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_run_at", sa.DateTime(timezone=True)),
        sa.Column("next_due_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("progress", json, nullable=False),
    )
    op.create_table(
        "collection_observations",
        sa.Column(
            "direction",
            sa.String(64),
            sa.ForeignKey("collection_directions.key", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("source_id", sa.String(255), primary_key=True),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("fingerprint", sa.String(64)),
        sa.Column("pending", sa.Boolean(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("retry_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_collection_pending", "collection_observations", ["direction", "pending", "retry_at"]
    )
    op.create_table(
        "collection_budgets",
        sa.Column("source", sa.String(50), primary_key=True),
        sa.Column("window_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("search_used", sa.Integer(), nullable=False),
        sa.Column("detail_used", sa.Integer(), nullable=False),
        sa.Column("next_request_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("blocked", sa.Boolean(), nullable=False),
        sa.Column("error", sa.String(64)),
        sa.Column("metrics", json, nullable=False),
    )
    for name in (
        "last_check_attempt_at",
        "next_check_at",
        "check_requested_at",
        "last_recommended_at",
    ):
        op.add_column("vacancies", sa.Column(name, sa.DateTime(timezone=True)))
    op.add_column("vacancies", sa.Column("search_fingerprint", sa.String(64)))
    for name in ("next_check_at", "check_requested_at"):
        op.create_index(f"ix_vacancies_{name}", "vacancies", [name])
    # Old failures were stamped as checks. Preserve the attempt, invalidate success.
    op.execute("UPDATE vacancies SET last_check_attempt_at=last_checked_at")
    op.execute("UPDATE vacancies SET last_checked_at=NULL WHERE check_error IS NOT NULL")


def downgrade():
    for name in ("next_check_at", "check_requested_at"):
        op.drop_index(f"ix_vacancies_{name}", table_name="vacancies")
    for name in (
        "last_check_attempt_at",
        "next_check_at",
        "check_requested_at",
        "search_fingerprint",
        "last_recommended_at",
    ):
        op.drop_column("vacancies", name)
    op.drop_table("collection_observations")
    op.drop_table("collection_directions")
    op.drop_table("collection_budgets")
