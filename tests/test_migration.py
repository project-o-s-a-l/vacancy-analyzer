import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.util import CommandError
from asyncpg import ConnectionDoesNotExistError
from sqlalchemy import MetaData, Table, create_engine, inspect, select

from alembic import context
from storage.models import Base, VacancyModel


def test_migration_chain_matches_orm_and_downgrades(vacancy):
    migrations = []
    for path in sorted((Path(__file__).parents[1] / "alembic/versions").glob("*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        migrations.append(module)
    engine = create_engine("sqlite:///:memory:")
    try:
        with engine.begin() as connection:
            context = MigrationContext.configure(connection)
            with Operations.context(context):
                migrations[0].upgrade()
                values = vacancy.model_dump()
                values.update(url=str(vacancy.url), last_seen_at=vacancy.collected_at)
                original = Table("vacancies", MetaData(), autoload_with=connection)
                values = {key: value for key, value in values.items() if key in original.c}
                values["id"] = vacancy.id.hex  # Reflected SQLite UUID is a CHAR(32).
                connection.execute(original.insert().values(**values))
                for migration in migrations[1:]:
                    migration.upgrade()
                assert compare_metadata(context, Base.metadata) == []
                assert connection.scalar(select(VacancyModel.id)) == vacancy.id
                assert connection.scalar(select(VacancyModel.archived)) is False
                assert connection.scalar(select(VacancyModel.search_version)) == "lexical-v1"
                assert "python" in connection.scalar(select(VacancyModel.search_terms))
                for migration in reversed(migrations[1:]):
                    migration.downgrade()
                assert connection.scalar(select(VacancyModel.id)) == vacancy.id
                migrations[0].downgrade()
                assert inspect(connection).get_table_names() == []
    finally:
        engine.dispose()


def test_alembic_connection_failure_is_actionable_and_closes_engine(monkeypatch):
    from jobs import settings
    from storage import database

    engine = SimpleNamespace(
        connect=AsyncMock(
            side_effect=ConnectionDoesNotExistError("private connection diagnostics")
        ),
        dispose=AsyncMock(),
    )
    monkeypatch.setattr(database, "create_engine", lambda url: engine)
    monkeypatch.setattr(settings, "Settings", lambda: SimpleNamespace(database_url="test"))
    monkeypatch.setattr(context, "config", SimpleNamespace(attributes={}), raising=False)
    monkeypatch.setattr(context, "is_offline_mode", lambda: False)
    path = Path(__file__).parents[1] / "alembic/env.py"
    spec = importlib.util.spec_from_file_location("failed_connection_migration_env", path)
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(CommandError, match="POSTGRES_PORT") as error:
        spec.loader.exec_module(module)
    assert "private connection diagnostics" not in str(error.value)
    engine.dispose.assert_awaited_once()
