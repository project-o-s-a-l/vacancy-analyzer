"""Real PostgreSQL contracts, isolated in a disposable schema per test."""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from bot.delivery import TelegramSender
from domain import Vacancy
from jobs.notify import NotificationService
from jobs.users import UserService
from storage.repositories import VacancyRepository


@pytest.mark.postgres
async def test_postgres_upsert_jsonb_and_history(postgres_sessions, vacancy):
    async with postgres_sessions() as session:
        repository = VacancyRepository(session)
        async with session.begin():
            identifier = await repository.save(vacancy)
            data = vacancy.model_dump()
            data.update(id=uuid4(), title="Updated", raw_payload={"changed": True})
            assert await repository.save(Vacancy(**data)) == identifier
        assert (await repository.get_by_id(identifier)).title == "Updated"
        assert len(await repository.history(identifier)) == 2


@pytest.mark.postgres
async def test_competing_workers_send_one_card(postgres_sessions, vacancy, profile):
    vacancy.published_at = datetime.now(UTC)
    async with postgres_sessions.begin() as session:
        await VacancyRepository(session).save(vacancy)
    service = NotificationService(postgres_sessions, profile, chat_id=42, top=1, min_score=0)
    sender = SimpleNamespace(send_message=AsyncMock(return_value=SimpleNamespace(message_id=123)))
    reports = await asyncio.gather(*(service.deliver(TelegramSender(sender)) for _ in range(8)))
    assert sum(report.sent for report in reports) == 1
    sender.send_message.assert_awaited_once()
    assert await service.delivery_counts() == {"sent": 1}


@pytest.mark.postgres
async def test_users_registered_concurrently_keep_separate_profiles(postgres_sessions):
    service = UserService(postgres_sessions)
    registrations = await asyncio.gather(*(service.register(42) for _ in range(8)))
    assert all(user.user_id == 42 for user in registrations)
    await service.register(99)
    await service.accept_answer(42, "Python, Docker")
    await service.accept_answer(99, "Java, Kafka")
    # A newly created service reads the same persisted state, not a process-local profile.
    other = UserService(postgres_sessions)
    assert set((await other.get(42)).profile.skills) == {"python", "docker"}
    assert set((await other.get(99)).profile.skills) == {"java", "kafka"}
    assert (await other.get(42)).form_state == "setup:experience"
