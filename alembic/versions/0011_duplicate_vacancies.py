"""Group exact copies for recommendation delivery without merging source records."""

import sqlalchemy as sa

from alembic import op
from domain.vacancy_duplicates import DUPLICATE_FIELDS, duplicate_key_v1

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("vacancies", sa.Column("duplicate_key", sa.String(64)))
    op.add_column("telegram_notifications", sa.Column("duplicate_key", sa.String(64)))
    op.create_index("ix_vacancies_duplicate_key", "vacancies", ["duplicate_key"])
    op.create_index(
        "ix_notifications_duplicate",
        "telegram_notifications",
        ["chat_id", "duplicate_key", "status"],
    )
    connection = op.get_bind()
    vacancies = sa.Table("vacancies", sa.MetaData(), autoload_with=connection)
    notifications = sa.Table("telegram_notifications", sa.MetaData(), autoload_with=connection)
    cursor = None
    while True:
        statement = (
            sa.select(vacancies.c.id, *(vacancies.c[name] for name in DUPLICATE_FIELDS))
            .order_by(vacancies.c.id)
            .limit(100)
        )
        if cursor is not None:
            statement = statement.where(vacancies.c.id > cursor)
        rows = connection.execute(statement).mappings().all()
        if not rows:
            break
        for row in rows:
            key = duplicate_key_v1(row)
            if key is not None:
                connection.execute(
                    vacancies.update().where(vacancies.c.id == row["id"]).values(duplicate_key=key)
                )
        cursor = rows[-1]["id"]
    connection.execute(
        notifications.update().values(
            duplicate_key=sa.select(vacancies.c.duplicate_key)
            .where(vacancies.c.id == notifications.c.vacancy_id)
            .scalar_subquery()
        )
    )


def downgrade():
    op.drop_index("ix_notifications_duplicate", table_name="telegram_notifications")
    op.drop_index("ix_vacancies_duplicate_key", table_name="vacancies")
    op.drop_column("telegram_notifications", "duplicate_key")
    op.drop_column("vacancies", "duplicate_key")
