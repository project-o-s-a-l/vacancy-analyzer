"""Independent candidate profiles and persistent forms for Telegram users."""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "telegram_users",
        sa.Column("user_id", sa.BigInteger(), primary_key=True),
        sa.Column("profile", sa.JSON().with_variant(JSONB(), "postgresql"), nullable=False),
        sa.Column("form_state", sa.String(40)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("telegram_users")
