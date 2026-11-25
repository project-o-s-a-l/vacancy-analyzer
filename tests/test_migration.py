import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.util import CommandError
from asyncpg import ConnectionDoesNotExistError
from sqlalchemy import MetaData, Table, create_engine, inspect, select

from alembic import command, context
from storage.models import Base, VacancyModel, VacancySnapshot


def test_migration_chain_matches_orm_and_downgrades(vacancy):
    config = Config(str(Path(__file__).parents[1] / "alembic.ini"))
    engine = create_engine("sqlite:///:memory:")
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql("PRAGMA foreign_keys=ON")
            config.attributes["connection"] = connection
            command.upgrade(config, "0001")
            values = vacancy.model_dump()
            values.update(url=str(vacancy.url), last_seen_at=vacancy.collected_at)
            original = Table("vacancies", MetaData(), autoload_with=connection)
            values = {key: value for key, value in values.items() if key in original.c}
            values["id"] = vacancy.id.hex  # Reflected SQLite UUID is a CHAR(32).
            connection.execute(original.insert().values(**values))
            command.upgrade(config, "0026")
            snapshot = uuid4()
            connection.execute(
                VacancySnapshot.__table__.insert().values(
                    id=snapshot,
                    vacancy_id=vacancy.id,
                    payload_hash="historical",
                    raw_payload={"observed": True},
                    observed_at=vacancy.collected_at,
                )
            )
            historical = select(VacancySnapshot.raw_payload).where(VacancySnapshot.id == snapshot)
            command.upgrade(config, "head")
            assert compare_metadata(MigrationContext.configure(connection), Base.metadata) == []
            assert connection.scalar(select(VacancyModel.id)) == vacancy.id
            assert connection.scalar(select(VacancyModel.archived)) is False
            assert connection.scalar(select(VacancyModel.search_version)) == "lexical-v1"
            assert "python" in connection.scalar(select(VacancyModel.search_terms))
            assert connection.scalar(historical) == {"observed": True}
            command.downgrade(config, "0026")
            assert connection.scalar(historical) == {"observed": True}
            assert connection.exec_driver_sql("PRAGMA foreign_key_check").first() is None
            command.downgrade(config, "0001")
            assert connection.scalar(select(VacancyModel.id)) == vacancy.id
            command.downgrade(config, "base")
            assert inspect(connection).get_table_names() == ["alembic_version"]
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
