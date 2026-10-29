"""Persist inference uncertainty/fairness and bounded, owner-scoped chunk reuse."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0019"
down_revision = "0018"
branch_labels = None
depends_on = None


def upgrade():
    json = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
    op.create_table(
        "inference_gate",
        sa.Column("name", sa.String(32), primary_key=True),
        sa.Column("resume_streak", sa.Integer(), nullable=False),
        sa.Column("token", sa.Uuid()),
        sa.Column("until", sa.DateTime(timezone=True)),
        sa.Column("runtime", json),
    )
    op.create_table(
        "extraction_cache",
        sa.Column("scope", sa.String(80), primary_key=True),
        sa.Column("key", sa.String(64), primary_key=True),
        sa.Column(
            "user_id", sa.BigInteger(), sa.ForeignKey("telegram_users.user_id", ondelete="CASCADE")
        ),
        sa.Column("vacancy_id", sa.Uuid(), sa.ForeignKey("vacancies.id", ondelete="CASCADE")),
        sa.Column("value", json, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_extraction_cache_expiry", "extraction_cache", ["expires_at"])


def downgrade():
    op.drop_table("extraction_cache")
    op.drop_table("inference_gate")
