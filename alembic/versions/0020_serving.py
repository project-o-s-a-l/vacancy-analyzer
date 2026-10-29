"""Revision-fenced serving caches; immutable confirmed delivery snapshots."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0020"
down_revision = "0019"
branch_labels = None
depends_on = None


def upgrade():
    json = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
    op.create_table(
        "recommendation_exposures",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "vacancy_id",
            sa.Uuid(),
            sa.ForeignKey("vacancies.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("message_id", sa.BigInteger(), nullable=False),
        sa.Column("confirmed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("context", json),
        sa.UniqueConstraint("chat_id", "message_id", name="uq_exposure_message"),
    )
    op.create_index("ix_recommendation_exposures_chat_id", "recommendation_exposures", ["chat_id"])
    op.execute(
        sa.text("""INSERT INTO recommendation_exposures
        (id, chat_id, vacancy_id, message_id, confirmed_at, context)
        SELECT claim_token, chat_id, vacancy_id, message_id,
               COALESCE(delivered_at, attempted_at), exposure_context
        FROM telegram_notifications WHERE status = 'sent' AND message_id IS NOT NULL
        ON CONFLICT (chat_id, message_id) DO NOTHING""")
    )
    op.create_table(
        "serving_revisions",
        sa.Column("scope", sa.String(80), primary_key=True),
        sa.Column("revision", sa.BigInteger(), nullable=False),
    )
    op.add_column("saved_selections", sa.Column("candidate_context", json))
    op.add_column("saved_selections", sa.Column("candidate_entries", json))
    op.add_column("saved_selections", sa.Column("candidate_expires_at", sa.DateTime(timezone=True)))
    from storage.serving_revisions import install

    install(op.get_bind())


def downgrade():
    from storage.serving_revisions import uninstall

    uninstall(op.get_bind())
    op.drop_column("saved_selections", "candidate_expires_at")
    op.drop_column("saved_selections", "candidate_entries")
    op.drop_column("saved_selections", "candidate_context")
    op.drop_table("serving_revisions")
    op.drop_table("recommendation_exposures")
