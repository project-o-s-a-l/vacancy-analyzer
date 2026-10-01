from datetime import timedelta
from uuid import uuid4

import pytest
from sqlalchemy import func, select

from domain import Vacancy
from storage.models import VacancyModel, VacancySnapshot
from storage.repositories import VacancyRepository


async def test_round_trip_preserves_domain_and_raw_json(session, vacancy):
    repository = VacancyRepository(session)
    async with session.begin():
        identifier = await repository.save(vacancy)
    loaded = await repository.get_by_id(identifier)
    assert loaded == vacancy
    assert await repository.get_by_id(uuid4()) is None


async def test_repeated_collection_keeps_uuid_first_seen_and_one_snapshot(session, vacancy):
    repository = VacancyRepository(session)
    async with session.begin():
        identifier = await repository.save(vacancy)
        data = vacancy.model_dump()
        data.update(id=uuid4(), collected_at=vacancy.collected_at + timedelta(hours=1))
        assert await repository.save(Vacancy(**data)) == identifier
    result = await repository.get_by_id(identifier)
    assert result.collected_at == vacancy.collected_at
    assert await repository.history(identifier) == [vacancy.raw_payload]
    assert await session.scalar(select(func.count()).select_from(VacancyModel)) == 1


async def test_changed_payload_retains_old_version_and_updates_latest(session, vacancy):
    repository = VacancyRepository(session)
    async with session.begin():
        identifier = await repository.save(vacancy)
        data = vacancy.model_dump()
        data.update(
            id=uuid4(),
            title="Updated title",
            raw_payload={**vacancy.raw_payload, "new_api_field": {"text": "  untouched  "}},
            collected_at=vacancy.collected_at + timedelta(hours=1),
        )
        updated = Vacancy(**data)
        await repository.save(updated)
    latest = await repository.get_by_id(identifier)
    assert latest.title == "Updated title"
    assert latest.raw_payload == updated.raw_payload
    assert await repository.history(identifier) == [vacancy.raw_payload, updated.raw_payload]


async def test_same_source_id_from_different_source_is_separate(session, vacancy):
    repository = VacancyRepository(session)
    async with session.begin():
        first = await repository.save(vacancy)
        data = vacancy.model_dump()
        data.update(id=uuid4(), source="other")
        second = await repository.save(Vacancy(**data))
    assert first != second


async def test_transaction_failure_rolls_back_vacancy_and_snapshot(session, vacancy):
    repository = VacancyRepository(session)
    with pytest.raises(RuntimeError):
        async with session.begin():
            await repository.save(vacancy)
            raise RuntimeError("simulate pipeline failure")
    assert await session.scalar(select(func.count()).select_from(VacancyModel)) == 0
    assert await session.scalar(select(func.count()).select_from(VacancySnapshot)) == 0


async def test_recent_list_filters_and_limits(session, vacancy):
    repository = VacancyRepository(session)
    async with session.begin():
        await repository.save(vacancy)
    assert await repository.list_recent(since=vacancy.published_at + timedelta(days=1)) == []
    assert len(await repository.list_recent(since=vacancy.published_at, limit=1)) == 1
    with pytest.raises(ValueError):
        await repository.list_recent(since=vacancy.published_at, limit=0)


async def test_json_key_order_does_not_add_duplicate_snapshot(session, vacancy):
    repository = VacancyRepository(session)
    async with session.begin():
        await repository.save(vacancy)
        data = vacancy.model_dump()
        data["raw_payload"] = dict(reversed(list(vacancy.raw_payload.items())))
        await repository.save(Vacancy(**data))
    assert len(await repository.history(vacancy.id)) == 1
