"""Deliver a Telegram notice when an outstanding resume import finishes."""

import sqlalchemy as sa

from alembic import op

revision = "0024"
down_revision = "0023"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "user_resumes", sa.Column("notice_due_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.create_index("ix_user_resumes_notice_due_at", "user_resumes", ["notice_due_at"])
    op.execute(
        "UPDATE user_resumes SET notice_due_at = CURRENT_TIMESTAMP "
        "WHERE processing_job_id IN (SELECT id FROM background_jobs "
        "WHERE kind = 'resume_analysis' AND state IN ('pending', 'running'))"
    )


def downgrade():
    op.drop_index("ix_user_resumes_notice_due_at", table_name="user_resumes")
    op.drop_column("user_resumes", "notice_due_at")
