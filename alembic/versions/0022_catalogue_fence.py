"""Do not invalidate retrieval for recommendation/check scheduling metadata."""

from alembic import op
from storage.serving_revisions import install

revision = "0022"
down_revision = "0021"
branch_labels = None
depends_on = None


def upgrade():
    install(op.get_bind())


def downgrade():
    # Keep all source/private data; restore conservative earlier invalidation.
    install(op.get_bind(), scheduling_is_content=True)
