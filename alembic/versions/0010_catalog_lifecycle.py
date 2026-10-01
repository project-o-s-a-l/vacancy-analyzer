"""Explainable search plans and bounded catalogue actuality checks."""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "search_plans",
        sa.Column(
            "user_id",
            sa.BigInteger(),
            sa.ForeignKey("telegram_users.user_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("version", sa.String(32), nullable=False),
        sa.Column("plan_hash", sa.String(64), nullable=False),
        sa.Column("queries", sa.JSON().with_variant(JSONB(), "postgresql"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    for column in (
        sa.Column("catalog_revision", sa.String(64)),
        sa.Column("revision_number", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("raw_payload_hash", sa.String(64)),
        sa.Column("availability", sa.String(16), nullable=False, server_default="active"),
        sa.Column("last_checked_at", sa.DateTime(timezone=True)),
        sa.Column("closed_at", sa.DateTime(timezone=True)),
        sa.Column("check_error", sa.String(32)),
    ):
        op.add_column("vacancies", column)
    op.execute(
        "UPDATE vacancies SET availability = CASE WHEN archived THEN 'closed' "
        "WHEN valid_until <= CURRENT_TIMESTAMP THEN 'expired' ELSE 'active' END"
    )
    with op.batch_alter_table("vacancies") as batch:
        batch.create_check_constraint(
            "ck_vacancy_availability",
            "availability IN ('active', 'closed', 'expired', 'unknown')",
        )
    op.create_index(
        "ix_vacancies_check_due", "vacancies", ["source", "availability", "last_checked_at"]
    )
    op.create_index("ix_vacancies_expiry", "vacancies", ["availability", "valid_until"])


def downgrade():
    op.drop_index("ix_vacancies_expiry", table_name="vacancies")
    op.drop_index("ix_vacancies_check_due", table_name="vacancies")
    with op.batch_alter_table("vacancies") as batch:
        batch.drop_constraint("ck_vacancy_availability", type_="check")
    for column in (
        "catalog_revision",
        "revision_number",
        "raw_payload_hash",
        "availability",
        "last_checked_at",
        "closed_at",
        "check_error",
    ):
        op.drop_column("vacancies", column)
    op.drop_table("search_plans")
