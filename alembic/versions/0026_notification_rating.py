"""Keep a single Telegram recommendation open for rating in each chat."""

import sqlalchemy as sa

from alembic import op

revision = "0026"
down_revision = "0025"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "telegram_notifications",
        sa.Column("awaiting_rating", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.create_index(
        "uq_notifications_awaiting_rating",
        "telegram_notifications",
        ["chat_id"],
        unique=True,
        postgresql_where=sa.text("awaiting_rating IS TRUE"),
        sqlite_where=sa.text("awaiting_rating IS 1"),
    )


def downgrade():
    op.drop_index("uq_notifications_awaiting_rating", table_name="telegram_notifications")
    op.drop_column("telegram_notifications", "awaiting_rating")
