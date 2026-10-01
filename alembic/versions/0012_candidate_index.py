"""Lexical index over the full catalogue, including exact technology tokens."""

from types import SimpleNamespace

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op
from domain.retrieval import RETRIEVAL_VERSION
from domain.vacancy_analysis import analysis_input_hash
from scoring.retrieval import INDEX_FIELDS, document_terms, search_document
from storage.models.vacancy import SEARCH_REPAIR_PREDICATE

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("vacancies", sa.Column("search_version", sa.String(32)))
    op.add_column("vacancies", sa.Column("search_revision", sa.String(64)))
    op.add_column(
        "vacancies", sa.Column("search_document", sa.JSON().with_variant(JSONB(), "postgresql"))
    )
    op.add_column(
        "vacancies", sa.Column("search_terms", sa.JSON().with_variant(JSONB(), "postgresql"))
    )
    connection = op.get_bind()
    table = sa.Table("vacancies", sa.MetaData(), autoload_with=connection)
    cursor = None
    while True:
        statement = (
            sa.select(table.c.id, table.c.experience, *(table.c[name] for name in INDEX_FIELDS))
            .order_by(table.c.id)
            .limit(32)
        )
        if cursor is not None:
            statement = statement.where(table.c.id > cursor)
        rows = connection.execute(statement).mappings().all()
        if not rows:
            break
        for row in rows:
            document = search_document(row)
            revision_hash = analysis_input_hash(SimpleNamespace(**row))
            connection.execute(
                table.update()
                .where(table.c.id == row["id"])
                .values(
                    search_version=RETRIEVAL_VERSION,
                    content_revision=revision_hash,
                    search_revision=revision_hash,
                    search_document=document,
                    search_terms=document_terms(document),
                )
            )
        cursor = rows[-1]["id"]
    op.create_index(
        "ix_vacancies_search_terms", "vacancies", ["search_terms"], postgresql_using="gin"
    )
    op.create_index("ix_vacancies_active_date", "vacancies", ["archived", "published_at"])
    op.create_index(
        "ix_vacancies_search_repair",
        "vacancies",
        ["id"],
        postgresql_where=sa.text(SEARCH_REPAIR_PREDICATE),
        sqlite_where=sa.text(SEARCH_REPAIR_PREDICATE),
    )


def downgrade():
    op.drop_index("ix_vacancies_search_repair", table_name="vacancies")
    op.drop_index("ix_vacancies_active_date", table_name="vacancies")
    op.drop_index("ix_vacancies_search_terms", table_name="vacancies")
    for name in ("search_terms", "search_document", "search_revision", "search_version"):
        op.drop_column("vacancies", name)
