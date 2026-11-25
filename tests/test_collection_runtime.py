import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import create_async_engine
from test_collection_scheduler import candidate, store

from collectors.base import SearchQuery
from collectors.errors import CollectionStopped, VacancyParseError
from jobs import collection_runtime
from jobs.collection_runtime import (
    CollectionAlreadyRunning,
    background_collection,
    collection_lease,
    run_collection,
)
from jobs.settings import Settings
from storage.models import VacancyModel


def settings(**kwargs):
    return Settings(_env_file=None, **kwargs)


async def test_bot_queues_resumes_and_closes_executor_before_collection(monkeypatch):
    from bot import runtime

    order = []

    @asynccontextmanager
    async def analysis_context(config, engine):
        assert config.resume_llm_enabled
        try:
            yield
        finally:
            order.append("analysis")

    @asynccontextmanager
    async def collection_context(*args):
        try:
            yield
        finally:
            order.append("collection")

    def service(*args, **kwargs):
        assert kwargs["queued_resumes"] is True
        assert kwargs["resume_model"] == "qwen3:4b"
        assert "resume_analyzer" not in kwargs
        return SimpleNamespace(check_ready=AsyncMock())

    bot = SimpleNamespace(
        session=SimpleNamespace(close=AsyncMock(side_effect=lambda: order.append("bot"))),
        get_webhook_info=AsyncMock(return_value=SimpleNamespace(url="")),
        set_my_commands=AsyncMock(),
    )
    engine = SimpleNamespace(dispose=AsyncMock(side_effect=lambda: order.append("engine")))
    monkeypatch.setattr(runtime, "background_analysis", analysis_context)
    monkeypatch.setattr(runtime, "background_collection", collection_context)
    monkeypatch.setattr(runtime, "UserService", service)
    monkeypatch.setattr(runtime, "create_bot", lambda _: bot)
    monkeypatch.setattr(runtime, "create_engine", lambda _: engine)
    monkeypatch.setattr(runtime, "session_factory", lambda _: None)
    monkeypatch.setattr(
        runtime, "build_dispatcher", lambda _: SimpleNamespace(start_polling=AsyncMock())
    )
    await runtime.run_bot(settings(resume_llm_enabled=True))
    assert order == ["analysis", "collection", "bot", "engine"]


@pytest.fixture
def isolated_collection_lock(monkeypatch):
    # Test schemas share a DB with the local app; advisory locks are database-wide.
    monkeypatch.setattr(collection_runtime, "COLLECTION_LOCK", uuid4().int % (2**63))


async def test_application_worker_collects_profiles_into_shared_catalog(
    sessions, vacancy, monkeypatch
):
    engine = sessions.kw["bind"]
    vacancy = vacancy.model_copy(update={"published_at": datetime.now(UTC) - timedelta(hours=1)})
    await store(sessions, 1, candidate("Python developer", "python"))
    client = SimpleNamespace(close=Mock())
    source = SimpleNamespace(search=AsyncMock(return_value=[vacancy]))

    @asynccontextmanager
    async def connect(_settings):
        try:
            yield source
        finally:
            client.close()

    monkeypatch.setattr(
        collection_runtime, "source_adapter", lambda _: SimpleNamespace(open=connect)
    )
    idle = asyncio.Event()
    original_loop = collection_runtime.collection_loop

    async def sleep(delay):
        assert delay == 60
        idle.set()
        await asyncio.Event().wait()

    async def loop(*args, **kwargs):
        assert args[3] is True
        await original_loop(*args, **kwargs, sleep=sleep)

    monkeypatch.setattr(collection_runtime, "collection_loop", loop)
    async with background_collection(settings(collection_mode="profiles"), engine):
        async with asyncio.timeout(3):
            await idle.wait()
        client.close.assert_not_called()
        async with sessions() as session:
            assert await session.scalar(select(func.count()).select_from(VacancyModel)) == 1
        query = source.search.await_args.args[0]
        assert query.text == "Python developer" and query.search_field == "all"
        assert query.max_pages == 1 and query.period is None and query.per_page == 20
    client.close.assert_called_once()


async def test_disabled_background_collection_does_not_start_worker(monkeypatch):
    collect = AsyncMock()
    monkeypatch.setattr(collection_runtime, "run_collection", collect)
    async with background_collection(settings(hh_auto_collect=False), None):
        await asyncio.sleep(0)
    collect.assert_not_awaited()


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (CollectionStopped("HH показал капчу"), "CollectionStopped"),
        (VacancyParseError("Дата публикации отсутствует"), "VacancyParseError"),
        (ValueError("private credentials"), "ValueError"),
        (RuntimeError("private credentials"), "RuntimeError"),
        (CollectionAlreadyRunning(), "CollectionAlreadyRunning"),
    ],
)
async def test_worker_failure_keeps_application_alive_without_retry(
    monkeypatch, caplog, error, expected
):
    completed = asyncio.Event()

    async def fail(*args, **kwargs):
        completed.set()
        raise error

    collect = AsyncMock(side_effect=fail)
    monkeypatch.setattr(collection_runtime, "run_collection", collect)
    async with background_collection(settings(), None):
        async with asyncio.timeout(3):
            await completed.wait()
        await asyncio.sleep(0)
        assert any(r.event_error["type"] == expected for r in caplog.records)
        assert "private credentials" not in caplog.text
    collect.assert_awaited_once()


@pytest.mark.parametrize("polling_fails", [False, True])
async def test_bot_starts_common_worker_and_stops_it_before_closing_resources(
    monkeypatch, polling_fails
):
    from bot import runtime

    started = asyncio.Event()
    stopped = asyncio.Event()
    order = []

    async def collect(*args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
            order.append("collector")

    async def poll(*args, **kwargs):
        async with asyncio.timeout(3):
            await started.wait()
        assert not stopped.is_set()
        if polling_fails:
            raise RuntimeError("polling stopped")

    bot = SimpleNamespace(
        session=SimpleNamespace(close=AsyncMock(side_effect=lambda: order.append("bot"))),
        get_webhook_info=AsyncMock(return_value=SimpleNamespace(url="")),
        set_my_commands=AsyncMock(),
    )
    engine = SimpleNamespace(dispose=AsyncMock(side_effect=lambda: order.append("engine")))
    service = SimpleNamespace(check_ready=AsyncMock())
    monkeypatch.setattr(collection_runtime, "run_collection", collect)
    monkeypatch.setattr(runtime, "create_bot", lambda _: bot)
    monkeypatch.setattr(runtime, "create_engine", lambda _: engine)
    monkeypatch.setattr(runtime, "session_factory", lambda _: None)
    monkeypatch.setattr(runtime, "UserService", lambda *args, **kwargs: service)
    monkeypatch.setattr(runtime, "build_dispatcher", lambda _: SimpleNamespace(start_polling=poll))
    if polling_fails:
        with pytest.raises(RuntimeError, match="polling stopped"):
            await runtime.run_bot(settings())
    else:
        await runtime.run_bot(settings())
    assert order == ["collector", "bot", "engine"]


@pytest.mark.parametrize("source", ["html", "api"])
async def test_standalone_command_uses_same_coordinator(monkeypatch, source):
    from jobs import cli

    config = settings()
    engine = SimpleNamespace(dispose=AsyncMock())
    collect = AsyncMock()
    monkeypatch.setattr(cli, "Settings", lambda: config)
    monkeypatch.setattr(cli, "create_engine", lambda _: engine)
    monkeypatch.setattr(cli, "session_factory", lambda _: None)
    monkeypatch.setattr(cli, "run_collection", collect)
    args = cli.parser().parse_args(
        [
            "collect",
            "--query",
            "Бухгалтер",
            "--source",
            source,
            "--pages",
            "2",
            "--watch",
            "--new-only",
        ]
    )
    await cli.run_command(args)
    assert collect.await_args.args[:2] == (config, engine)
    assert collect.await_args.args[2].max_pages == 2
    assert collect.await_args.kwargs == {
        "watch": True,
        "new_only": True,
        "query_texts": ["Бухгалтер"],
        "source": source,
    }
    engine.dispose.assert_awaited_once()


@pytest.mark.postgres
async def test_database_lease_blocks_another_process_before_network_and_releases_on_error(
    postgres_sessions, monkeypatch, isolated_collection_lock
):
    engine = postgres_sessions.kw["bind"]
    other_engine = create_async_engine(engine.url)
    monkeypatch.setattr(
        collection_runtime,
        "source_adapter",
        lambda _: SimpleNamespace(open=lambda *_: pytest.fail("Duplicate source connection")),
    )
    try:
        with pytest.raises(ValueError, match="worker failed"):
            async with collection_lease(engine):
                with pytest.raises(CollectionAlreadyRunning):
                    await run_collection(
                        settings(), other_engine, SearchQuery(text="x"), watch=False
                    )
                raise ValueError("worker failed")
        async with collection_lease(other_engine):
            pass
    finally:
        await other_engine.dispose()


@pytest.mark.postgres
async def test_database_lease_is_released_when_worker_is_cancelled(
    postgres_sessions, isolated_collection_lock
):
    engine = postgres_sessions.kw["bind"]
    other_engine = create_async_engine(engine.url)
    acquired = asyncio.Event()

    async def hold():
        async with collection_lease(engine):
            acquired.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(hold())
    try:
        async with asyncio.timeout(5):
            await acquired.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            async with collection_lease(other_engine):
                pass
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await other_engine.dispose()


async def test_background_collection_recovers_after_lock_and_transient_error(monkeypatch):
    run = AsyncMock(side_effect=[CollectionAlreadyRunning(), ConnectionError(), None])
    pause = AsyncMock()
    monkeypatch.setattr(collection_runtime, "run_collection", run)
    monkeypatch.setattr(collection_runtime.asyncio, "sleep", pause)
    await collection_runtime._background_worker(settings(), None)
    assert run.await_count == 3
    assert [call.args[0] for call in pause.await_args_list] == [5, 10]


@pytest.mark.parametrize("missing", [False, True], ids=["failed", "uninstalled"])
async def test_one_failing_source_keeps_other_sources_running(
    postgres_sessions, monkeypatch, missing
):
    completed = []

    def provider(name):
        if missing and name == "superjob":
            raise collection_runtime.SourceAdapterError("superjob")

        @asynccontextmanager
        async def connect(settings):
            if name == "superjob":
                raise ValueError("Source unavailable")
            completed.append(name)
            yield SimpleNamespace(paging_kind=None)

        return SimpleNamespace(open=connect)

    monkeypatch.setattr(collection_runtime, "source_adapter", provider)
    monkeypatch.setattr(collection_runtime, "collection_loop", AsyncMock())
    config = Settings(_env_file=None, collection_sources=("hh", "superjob", "trudvsem"))
    with pytest.raises(ValueError, match="superjob"):
        await run_collection(
            config, postgres_sessions.kw["bind"], SearchQuery(category="it"), watch=False
        )
    assert set(completed) == {"html", "trudvsem"}
