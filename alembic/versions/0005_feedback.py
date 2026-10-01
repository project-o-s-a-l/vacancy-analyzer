"""Personal feedback state, immutable events and recommendation exposure context."""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    json_type = sa.JSON().with_variant(JSONB(), "postgresql")
    op.add_column("telegram_notifications", sa.Column("exposure_context", json_type))
    op.create_table(
        "vacancy_feedback",
        sa.Column(
            "user_id",
            sa.BigInteger(),
            sa.ForeignKey("telegram_users.user_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "vacancy_id",
            sa.Uuid(),
            sa.ForeignKey("vacancies.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("interest", sa.String(10)),
        sa.Column("saved", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("applied", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("reason", sa.String(16)),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "interest IS NULL OR interest IN ('like','dislike')", name="ck_feedback_interest"
        ),
        sa.CheckConstraint(
            "reason IS NULL OR (interest IS NOT NULL AND interest = 'dislike')",
            name="ck_feedback_reason_interest",
        ),
        sa.CheckConstraint(
            "reason IS NULL OR reason IN ('salary','skills','experience','location',"
            "'company','duties','other')",
            name="ck_feedback_reason",
        ),
        sa.CheckConstraint("version > 0", name="ck_feedback_version"),
    )
    op.create_table(
        "vacancy_feedback_events",
        sa.Column(
            "user_id",
            sa.BigInteger(),
            sa.ForeignKey("telegram_users.user_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("event_key", sa.String(128), primary_key=True),
        sa.Column(
            "vacancy_id",
            sa.Uuid(),
            sa.ForeignKey("vacancies.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("action", sa.String(16), nullable=False),
        sa.Column("reason", sa.String(16)),
        sa.Column("exposure_context", json_type),
        sa.Column("action_context", json_type),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "action IN ('like','dislike','save','unsave','apply','unapply','reason')",
            name="ck_feedback_event_action",
        ),
        sa.CheckConstraint(
            "reason IS NULL OR reason IN ('salary','skills','experience','location',"
            "'company','duties','other')",
            name="ck_feedback_event_reason",
        ),
    )
    op.create_index(
        "ix_vacancy_feedback_events_vacancy_id", "vacancy_feedback_events", ["vacancy_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_vacancy_feedback_events_vacancy_id", table_name="vacancy_feedback_events")
    op.drop_table("vacancy_feedback_events")
    op.drop_table("vacancy_feedback")
    op.drop_column("telegram_notifications", "exposure_context")
