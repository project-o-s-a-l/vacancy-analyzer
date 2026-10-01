"""Persist local LLM interpretation without changing original HH data."""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "vacancies",
        sa.Column(
            "analysis",
            sa.JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql"),
        ),
    )


def downgrade() -> None:
    op.drop_column("vacancies", "analysis")
