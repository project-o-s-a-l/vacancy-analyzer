"""Scope crawls to a publisher and retain full API observations across restarts."""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op
from domain.vacancy_duplicates import DUPLICATE_FIELDS, cross_source_key

revision = "0025"
down_revision = "0024"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "collection_directions",
        sa.Column("source", sa.String(50), nullable=False, server_default="hh"),
    )
    op.add_column("collection_observations", sa.Column("detail_url", sa.Text()))
    op.add_column(
        "collection_observations",
        sa.Column("payload", sa.JSON().with_variant(JSONB(), "postgresql")),
    )
    op.add_column("vacancies", sa.Column("cross_source_key", sa.String(64)))
    op.create_index("ix_vacancies_cross_source_key", "vacancies", ["cross_source_key"])
    # Existing HH keys and all historic delivery claims keep their original identity.
    # Index existing publications so the first new source can find its peer immediately.
    conn = op.get_bind()
    vacancies = sa.Table("vacancies", sa.MetaData(), autoload_with=conn)
    cursor = None
    while True:
        stmt = sa.select(vacancies.c.id, *(vacancies.c[n] for n in DUPLICATE_FIELDS))
        if cursor is not None:
            stmt = stmt.where(vacancies.c.id > cursor)
        rows = conn.execute(stmt.order_by(vacancies.c.id).limit(200)).mappings().all()
        if not rows:
            break
        batch = [
            {"row_id": row["id"], "group_hash": key}
            for row in rows
            if (key := cross_source_key(row)) is not None
        ]
        if batch:
            conn.execute(
                vacancies.update()
                .where(vacancies.c.id == sa.bindparam("row_id"))
                .values(cross_source_key=sa.bindparam("group_hash")),
                batch,
            )
        cursor = rows[-1]["id"]


def downgrade():
    op.drop_index("ix_vacancies_cross_source_key", table_name="vacancies")
    op.drop_column("vacancies", "cross_source_key")
    op.drop_column("collection_observations", "payload")
    op.drop_column("collection_observations", "detail_url")
    op.drop_column("collection_directions", "source")
