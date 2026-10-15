"""Exercise the real Alembic chain in a new schema of a disposable test database."""

import asyncio
import json
import os
from pathlib import Path
from uuid import uuid4

from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import command
from storage.models import Base

ALEMBIC_CONFIG = Path(__file__).resolve().parents[1] / "alembic.ini"


async def check(url: str) -> dict:
    schema = "f00_migration_" + uuid4().hex
    engine = create_async_engine(url)
    config = Config(str(ALEMBIC_CONFIG))
    scoped = None
    try:
        async with engine.begin() as connection:
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        scoped = create_async_engine(url, connect_args={"server_settings": {"search_path": schema}})

        def migrate(connection, target):
            config.attributes["connection"] = connection
            command.upgrade(config, target)

        async with scoped.begin() as connection:
            await connection.run_sync(migrate, "head")

            def verify(connection):
                differences = compare_metadata(
                    MigrationContext.configure(connection, opts={"compare_type": True}),
                    Base.metadata,
                )
                if differences:
                    raise AssertionError(f"Migration/ORM drift: {differences!r}")
                return len(inspect(connection).get_table_names())

            table_count = await connection.run_sync(verify)
            head = await connection.scalar(text("SELECT version_num FROM alembic_version"))
            config.attributes["connection"] = connection.sync_connection
            await connection.run_sync(lambda _: command.downgrade(config, "base"))
            tables = await connection.run_sync(lambda c: inspect(c).get_table_names())
            if tables != ["alembic_version"]:
                raise AssertionError(f"Downgrade left tables: {tables!r}")
            await connection.run_sync(migrate, "head")
            await connection.run_sync(verify)
        return {
            "status": "passed",
            "head": head,
            "tables_including_alembic": table_count,
            "checks": ["upgrade head", "ORM parity", "downgrade base", "upgrade head"],
        }
    finally:
        if scoped is not None:
            await scoped.dispose()
        async with engine.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await engine.dispose()


if __name__ == "__main__":
    database_url = os.environ.get("TEST_DATABASE_URL", "")
    parsed = make_url(database_url)
    if parsed.get_backend_name() != "postgresql" or "test" not in (parsed.database or "").lower():
        raise SystemExit(
            "Set TEST_DATABASE_URL to an explicitly disposable PostgreSQL test database"
        )
    print(json.dumps(asyncio.run(check(database_url)), indent=2))
