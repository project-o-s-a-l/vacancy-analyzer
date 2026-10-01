"""Durable Telegram delivery claims and results."""

import sqlalchemy as sa

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "telegram_notifications",
        sa.Column("chat_id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "vacancy_id",
            sa.Uuid(),
            sa.ForeignKey("vacancies.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("claim_token", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("attempted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("delivered_at", sa.DateTime(timezone=True)),
        sa.Column("message_id", sa.BigInteger()),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("error_kind", sa.String(100)),
        sa.CheckConstraint(
            "status IN ('sending', 'sent', 'failed', 'uncertain')", name="ck_notification_status"
        ),
    )


def downgrade() -> None:
    op.drop_table("telegram_notifications")
