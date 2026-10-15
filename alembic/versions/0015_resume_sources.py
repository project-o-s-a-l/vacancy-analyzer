"""Keep confirmed resume sources; existing current texts remain untouched."""

import sqlalchemy as sa

from alembic import op

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "resume_sources",
        sa.Column(
            "user_id",
            sa.BigInteger(),
            sa.ForeignKey("telegram_users.user_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("revision", sa.String(64), primary_key=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade():
    # Deletes historical text: operational rollback keeps this table and its data.
    op.drop_table("resume_sources")
