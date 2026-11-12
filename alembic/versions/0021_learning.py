"""private review grants, lineage and selection observations; no fabricated backfill."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0021"
down_revision = "0020"
branch_labels = None
depends_on = None


def upgrade():
    json = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
    op.create_table(
        "learning_access",
        sa.Column(
            "user_id",
            sa.BigInteger(),
            sa.ForeignKey("telegram_users.user_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("allowed", sa.Boolean(), nullable=False),
        sa.Column("consent_ref", sa.String(64), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "learning_artifacts",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("owners", json, nullable=False),
        sa.Column("parents", json, nullable=False),
        sa.Column("manifest", json, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "selection_observations",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "user_id",
            sa.BigInteger(),
            sa.ForeignKey("telegram_users.user_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metrics", json, nullable=False),
    )
    op.create_index("ix_selection_observations_user_id", "selection_observations", ["user_id"])
    op.create_index(
        "ix_selection_observations_created_at", "selection_observations", ["created_at"]
    )


def downgrade():
    op.drop_table("selection_observations")
    op.drop_table("learning_artifacts")
    op.drop_table("learning_access")
