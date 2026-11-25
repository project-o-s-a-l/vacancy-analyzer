import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from domain import CandidateProfile, Vacancy
from storage.database import create_engine
from storage.models import Base


@pytest.fixture(scope="session", autouse=True)
def isolated_model_slot(tmp_path_factory):
    """Fake HTTP tests must not wait for a user's running Ollama process."""
    from nlp import ollama_slot

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ollama_slot, "LOCK_PATH", tmp_path_factory.mktemp("model-slot") / "test.lock")
        patch.setenv("LOG_DIRECTORY", str(tmp_path_factory.mktemp("logs")))
        yield


@pytest.fixture(scope="session", autouse=True)
def offline_salary_rates():
    from jobs import exchange_rates

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(exchange_rates._cache, "next_refresh", float("inf"))
        yield


@pytest.fixture
def sample_data():
    path = Path(__file__).parent / "fixtures" / "vacancies.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def vacancy(sample_data):
    return Vacancy.model_validate(sample_data["vacancies"][0])


@pytest.fixture
def profile(sample_data):
    return CandidateProfile.model_validate(sample_data["profile"])


@pytest.fixture
def now():
    return datetime(2026, 10, 16, tzinfo=UTC)


@pytest.fixture
async def sessions():
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


@pytest.fixture
async def session(sessions):
    async with sessions() as session:
        yield session


@pytest.fixture(
    params=[
        pytest.param("sessions", id="sqlite"),
        pytest.param("postgres_sessions", id="postgres", marks=pytest.mark.postgres),
    ]
)
def db_sessions(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
async def postgres_sessions():
    database_url = os.environ.get("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("Set TEST_DATABASE_URL to a disposable PostgreSQL database")
    schema = "test_" + uuid4().hex
    admin = create_async_engine(database_url)
    engine = None
    created = False
    try:
        async with admin.begin() as connection:
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        created = True
        engine = create_async_engine(
            database_url, connect_args={"server_settings": {"search_path": schema}}
        )
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        if engine is not None:
            await engine.dispose()
        if created:
            async with admin.begin() as connection:
                await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin.dispose()
