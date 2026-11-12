"""Durable private profile editor; active profiles remain in their existing columns."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0023"
down_revision = "0022"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "profile_edits",
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("token", sa.String(32), nullable=False),
        sa.Column("base_revision", sa.Integer(), nullable=False),
        sa.Column(
            "payload", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=False
        ),
        sa.ForeignKeyConstraint(["user_id"], ["telegram_users.user_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("user_id"),
    )


def downgrade():
    op.drop_table("profile_edits")
