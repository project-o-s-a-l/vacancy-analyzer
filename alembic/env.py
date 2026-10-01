import asyncio

from alembic.util import CommandError

from alembic import context
from jobs.settings import Settings
from storage.database import DATABASE_ERRORS, create_engine
from storage.models import Base

target_metadata = Base.metadata


def configure(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def run_online() -> None:
    engine = create_engine(Settings().database_url)
    try:
        try:
            connection = await engine.connect()
        except DATABASE_ERRORS:
            raise CommandError(
                "Не удалось подключиться к PostgreSQL. Проверьте запуск базы, "
                "POSTGRES_PORT и DATABASE_URL в .env. Сначала дождитесь готовности "
                "docker compose up -d --wait postgres."
            ) from None
        try:
            await connection.run_sync(configure)
        finally:
            await connection.close()
    finally:
        await engine.dispose()


if context.is_offline_mode():
    context.configure(
        url=Settings().database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()
else:
    asyncio.run(run_online())
