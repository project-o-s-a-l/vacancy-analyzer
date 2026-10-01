"""Fields retained by the primary HTML collector."""

import sqlalchemy as sa

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None
_TEXT_FIELDS = (
    "company_url",
    "description_html",
    "salary_raw",
    "employment",
    "schedule",
    "work_format",
    "working_hours",
    "address",
)


def upgrade() -> None:
    for name in _TEXT_FIELDS:
        op.add_column("vacancies", sa.Column(name, sa.Text()))
    op.add_column("vacancies", sa.Column("valid_until", sa.DateTime(timezone=True)))
    op.add_column(
        "vacancies", sa.Column("archived", sa.Boolean(), nullable=False, server_default=sa.false())
    )


def downgrade() -> None:
    for name in reversed((*_TEXT_FIELDS, "valid_until", "archived")):
        op.drop_column("vacancies", name)
