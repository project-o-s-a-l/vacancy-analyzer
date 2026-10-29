"""Real PostgreSQL and a killed child; Telegram is entirely simulated."""

import asyncio
import os
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select, text
from test_candidates import candidate
from test_saved_selections import seed

from jobs.users import UserService
from storage.models import ExposureModel, NotificationModel
from storage.repositories.user_repository import UserRepository


@pytest.mark.postgres
async def test_process_killed_after_claim_never_automatically_repeats_unknown_send(
    postgres_sessions, vacancy
):
    sessions = postgres_sessions
    await seed(sessions, vacancy, n=1)
    await UserService(sessions).register(42)
    async with sessions.begin() as s:
        repo = UserRepository(s)
        user = await repo.get(42)
        await repo.save(replace(user, profile=candidate(), form_state=None))
        schema = await s.scalar(text("SELECT current_schema()"))
    code = """
import asyncio, os, sys
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from jobs.users import UserService
async def main():
    engine = create_async_engine(os.environ['TEST_DATABASE_URL'],
        connect_args={'server_settings': {'search_path': sys.argv[1]}})
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    class Sender:
        async def send_recommendation(self, *args):
            print('CLAIM_COMMITTED', flush=True)
            await asyncio.Event().wait()
    service = await UserService(sessions, min_score=0).notifications(42)
    await service.deliver(Sender())
asyncio.run(main())
"""
    child = await asyncio.to_thread(
        subprocess.Popen,
        [sys.executable, "-c", code, schema],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
    )
    try:
        marker = await asyncio.wait_for(asyncio.to_thread(child.stdout.readline), timeout=20)
        assert marker.strip() == "CLAIM_COMMITTED"
        child.terminate()
        await asyncio.to_thread(child.wait, 10)
        sender = SimpleNamespace(send_recommendation=AsyncMock(return_value=999))
        service = await UserService(sessions, min_score=0).notifications(42)
        assert (await service.deliver(sender)).sent == 0
        sender.send_recommendation.assert_not_called()
        async with sessions() as s:
            row = await s.scalar(select(NotificationModel))
            assert row.status == "sending" and row.attempts == 1
            assert await s.scalar(select(ExposureModel)) is None
    finally:
        if child.poll() is None:
            child.kill()
            await asyncio.to_thread(child.wait, 10)
        child.stdout.close()
        child.stderr.close()


@pytest.mark.postgres
@pytest.mark.parametrize("stage", ["claim", "finish"])
async def test_real_database_disconnect_between_claim_send_finish(
    postgres_sessions, vacancy, monkeypatch, stage
):
    from sqlalchemy.exc import SQLAlchemyError

    from jobs.notify import NotificationService
    from storage.repositories.notification_repository import NotificationRepository

    sessions = postgres_sessions
    await seed(sessions, vacancy, n=1)
    service = NotificationService(sessions, candidate(), chat_id=42, min_score=0)
    sender = SimpleNamespace(send_recommendation=AsyncMock(return_value=99))
    original = getattr(NotificationRepository, stage)

    async def disconnect(repository, *args, **kwargs):
        result = await original(repository, *args, **kwargs)
        pid = await repository.session.scalar(text("SELECT pg_backend_pid()"))
        async with sessions() as killer:
            await killer.scalar(text("SELECT pg_terminate_backend(:pid)"), {"pid": pid})
        return result

    monkeypatch.setattr(NotificationRepository, stage, disconnect)
    with pytest.raises((SQLAlchemyError, ConnectionError)):
        await service.deliver(sender)
    monkeypatch.setattr(NotificationRepository, stage, original)
    async with sessions() as s:
        row = await s.scalar(select(NotificationModel))
        assert await s.scalar(select(ExposureModel)) is None
        if stage == "claim":
            assert row is None
            sender.send_recommendation.assert_not_called()
        else:
            assert row.status == "sending"
            sender.send_recommendation.assert_called_once()
    if stage == "finish":
        assert (await service.deliver(sender)).sent == 0
        sender.send_recommendation.assert_called_once()
