"""Bounded owner-specific scoring pools, separate from delivery history."""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "saved_selections",
        sa.Column("owner_id", sa.BigInteger(), primary_key=True),
        sa.Column("request_revision", sa.String(64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("context", sa.JSON().with_variant(JSONB(), "postgresql"), nullable=False),
        sa.Column("entries", sa.JSON().with_variant(JSONB(), "postgresql"), nullable=False),
    )


def downgrade():
    op.drop_table("saved_selections")
