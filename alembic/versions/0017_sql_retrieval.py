"""Compact PostgreSQL lexical Top-N; source/fact data and existing GIN are preserved."""

from alembic import op
from storage.retrieval_sql import RANK_FUNCTION

revision = "0017"
down_revision = "0016"
branch_labels = None
depends_on = None


def upgrade():
    if op.get_bind().dialect.name == "postgresql":
        op.execute(RANK_FUNCTION)


def downgrade():
    if op.get_bind().dialect.name == "postgresql":
        op.execute("DROP FUNCTION IF EXISTS va_lexical_rank_v2(jsonb,jsonb,jsonb)")
