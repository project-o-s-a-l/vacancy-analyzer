"""Allow confirmed per-user history resets without dropping delivery or feedback data."""

import sqlalchemy as sa

from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "telegram_notifications", sa.Column("history_reset_at", sa.DateTime(timezone=True))
    )
    op.add_column("telegram_users", sa.Column("history_reset_token", sa.String(32)))
    op.add_column("telegram_users", sa.Column("history_reset_before", sa.DateTime(timezone=True)))


def downgrade() -> None:
    op.drop_column("telegram_users", "history_reset_before")
    op.drop_column("telegram_users", "history_reset_token")
    op.drop_column("telegram_notifications", "history_reset_at")
