"""Additive professional facts; no parsing or unbounded UPDATE in schema migration."""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade():
    for table in ("telegram_users", "vacancies"):
        op.add_column(
            table,
            sa.Column(
                "professional_facts",
                sa.JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql"),
                nullable=True,
            ),
        )
        op.add_column(table, sa.Column("facts_input_revision", sa.String(64), nullable=True))
    op.add_column(
        "telegram_users",
        sa.Column("profile_revision", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column("vacancies", sa.Column("projection_revision", sa.String(64), nullable=True))
    op.add_column("vacancies", sa.Column("projection_contract", sa.String(64), nullable=True))
    op.add_column(
        "telegram_users", sa.Column("profile_schema_version", sa.String(40), nullable=True)
    )
    op.create_table(
        "professional_fact_revisions",
        sa.Column("subject_kind", sa.String(16), primary_key=True),
        sa.Column("subject_id", sa.String(255), primary_key=True),
        sa.Column("revision", sa.String(64), primary_key=True),
        sa.Column("input_revision", sa.String(64), nullable=False),
        sa.Column("schema_version", sa.String(40), nullable=False),
        sa.Column("payload", sa.JSON().with_variant(JSONB(), "postgresql"), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_professional_fact_revisions_input_revision",
        "professional_fact_revisions",
        ["input_revision"],
    )
    op.create_table(
        "professional_fact_backfills",
        sa.Column("name", sa.String(80), primary_key=True),
        sa.Column("cursor", sa.JSON().with_variant(JSONB(), "postgresql"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade():
    # Operational rollback keeps 0014. Destructive schema downgrade requires a backup.
    op.drop_table("professional_fact_backfills")
    op.drop_index(
        "ix_professional_fact_revisions_input_revision", table_name="professional_fact_revisions"
    )
    op.drop_table("professional_fact_revisions")
    op.drop_column("vacancies", "projection_revision")
    op.drop_column("vacancies", "projection_contract")
    op.drop_column("telegram_users", "profile_revision")
    op.drop_column("telegram_users", "profile_schema_version")
    for table in ("vacancies", "telegram_users"):
        op.drop_column(table, "facts_input_revision")
        op.drop_column(table, "professional_facts")
