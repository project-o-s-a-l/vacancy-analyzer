from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select, update
from test_candidates import candidate
from test_saved_selections import seed

from jobs.users import UserService
from storage.models import ServingRevisionModel, VacancyModel
from storage.repositories.candidate_repository import CandidateRepository
from storage.repositories.user_repository import RegisteredUser, UserRepository


async def test_service_metadata_preserves_cache_but_content_and_freshness_still_fence(
    db_sessions, vacancy, monkeypatch
):
    sessions = db_sessions
    await seed(sessions, vacancy, n=2)
    users = UserService(sessions, min_score=0)
    await users.register(42)
    async with sessions.begin() as session:
        repo = UserRepository(session)
        user = await repo.get(42)
        await repo.save(RegisteredUser(42, candidate(), None, user.revision))

    calls = []
    original = CandidateRepository.shortlist

    async def shortlist(self, *args, **kwargs):
        calls.append(1)
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(CandidateRepository, "shortlist", shortlist)
    first, message = await (await users.notifications(42))._selection()
    assert first and not message and calls
    count = len(calls)
    async with sessions() as session:
        epoch = await session.get(ServingRevisionModel, "catalogue")
        revision = epoch.revision
    async with sessions.begin() as session:
        await session.execute(update(VacancyModel).values(last_seen_at=datetime.now(UTC)))
    second, message = await (await users.notifications(42))._selection()
    assert [v.id for v, _ in second] == [v.id for v, _ in first]
    assert not message and len(calls) == count
    async with sessions() as session:
        assert (await session.get(ServingRevisionModel, "catalogue")).revision == revision
    # A source health change still fences, even if neither raw payload nor facts changed.
    async with sessions.begin() as session:
        await session.execute(
            update(VacancyModel).values(last_checked_at=datetime.now(UTC) - timedelta(days=2))
        )
    selected, message = await (await users.notifications(42))._selection()
    assert [v.id for v, _ in selected] == [v.id for v, _ in first]
    assert message is None and len(calls) > count
    count = len(calls)
    async with sessions.begin() as session:
        await session.execute(update(VacancyModel).values(archived=True))
    selected, _ = await (await users.notifications(42))._selection()
    assert not selected and len(calls) > count


@pytest.mark.postgres
async def test_metadata_write_in_same_transaction_does_not_hide_subsequent_content_change(
    postgres_sessions, vacancy
):
    from jobs.candidate_cache import identity
    from storage.repositories import VacancyRepository

    async with postgres_sessions.begin() as s:
        await VacancyRepository(s).save(vacancy)
    async with postgres_sessions.begin() as s:
        before = await identity(s, 42)
        await s.execute(update(VacancyModel).values(last_recommended_at=datetime.now(UTC)))
        assert await identity(s, 42) == before
        await s.execute(update(VacancyModel).values(archived=True))
        assert (await identity(s, 42))[0] < 0
    async with postgres_sessions() as s:
        assert (await identity(s, 42))[0] > before[0]
        assert await s.scalar(select(VacancyModel.archived))


@pytest.mark.postgres
async def test_populated_0022_rollback_restores_coarse_fence_and_upgrade_preserves_data(
    postgres_sessions, vacancy
):
    import importlib.util
    from pathlib import Path

    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    from storage.repositories import VacancyRepository

    sessions = postgres_sessions
    async with sessions.begin() as s:
        await VacancyRepository(s).save(vacancy)
    path = Path(__file__).parents[1] / "alembic/versions/0022_catalogue_fence.py"
    spec = importlib.util.spec_from_file_location("migration_0022", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def apply(connection, operation):
        with Operations.context(MigrationContext.configure(connection)):
            operation()

    for operation, bumps_revision in ((module.downgrade, True), (module.upgrade, False)):
        async with sessions.kw["bind"].begin() as c:
            await c.run_sync(apply, operation)
        async with sessions() as s:
            before = (await s.get(ServingRevisionModel, "catalogue")).revision
        async with sessions.begin() as s:
            await s.execute(update(VacancyModel).values(last_recommended_at=datetime.now(UTC)))
        async with sessions() as s:
            after = (await s.get(ServingRevisionModel, "catalogue")).revision
            assert (after > before) == bumps_revision
            row = await s.get(VacancyModel, vacancy.id)
            assert row.title == vacancy.title and row.raw_payload == vacancy.raw_payload
