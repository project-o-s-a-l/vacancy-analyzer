"""PG + fake Telegram load: selection, queue, pacing and HTTP are separate clocks."""

import argparse
import asyncio
import json
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from aiogram.types import Update
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from bot.rate_limits import TelegramLimits
from bot.update_inbox import UpdateInbox
from domain import CandidateProfile, Vacancy
from jobs.notify import NotificationService
from storage.models import Base
from storage.repositories import VacancyRepository


async def measure(output):
    output.mkdir(parents=True, exist_ok=False)
    url = os.environ["TEST_DATABASE_URL"]
    if "test" not in url.rsplit("/", 1)[-1] or not url.startswith("postgresql+asyncpg:"):
        raise ValueError("Disposable PostgreSQL required")
    schema = "f09_delivery_" + uuid4().hex
    admin = create_async_engine(url)
    engine = None
    results = []
    try:
        async with admin.begin() as s:
            await s.execute(text(f'CREATE SCHEMA "{schema}"'))
        engine = create_async_engine(
            url,
            pool_size=5,
            max_overflow=0,
            connect_args={"server_settings": {"search_path": schema}},
        )
        async with engine.begin() as s:
            await s.run_sync(Base.metadata.create_all)
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        data = json.loads(
            await asyncio.to_thread(
                Path("tests/fixtures/vacancies.json").read_text, encoding="utf-8"
            )
        )
        card = Vacancy.model_validate(data["vacancies"][0]).model_copy(
            update={
                "title": "Java Developer",
                "skills": ["Java", "Spring"],
                "description": "Разработка сервисов Java Spring",
                "description_html": None,
                "experience": None,
                "valid_until": None,
                "archived": False,
                "published_at": datetime.now(UTC) - timedelta(days=1),
            }
        )
        profile = CandidateProfile(skills={"java": {}, "spring": {}}, experience_years=3)
        async with sessions.begin() as s:
            await VacancyRepository(s).save(card)
        for mode in ("serial", "concurrent"):
            for delay in (0, 0.1, 1):
                for repeat in range(5):
                    group = len(results) + 1
                    queue = UpdateInbox(sessions, bot_id=group)
                    limits = TelegramLimits()
                    records, network = [], []
                    issued = time.perf_counter()

                    async def transport(bot, method, delay=delay, network=network):
                        before = time.perf_counter()
                        await asyncio.sleep(delay if method.chat_id % 100 == 0 else 0)
                        network.append(
                            {
                                "owner": method.chat_id % 100,
                                "http_ms": (time.perf_counter() - before) * 1000,
                            }
                        )
                        return 100 + method.chat_id

                    class Sender:
                        def __init__(self, limiter=limits, sender=transport):
                            self.limiter, self.sender = limiter, sender

                        async def send_recommendation(self, owner, vacancy, match):
                            return await self.limiter(
                                self.sender, None, SimpleNamespace(chat_id=owner)
                            )

                    async def request(owner, issued=issued, records=records):
                        began = time.perf_counter()
                        service = NotificationService(
                            sessions, profile, chat_id=owner, top=1, min_score=0
                        )
                        report = await service.deliver(Sender())
                        records.append(
                            dict(
                                owner=owner % 100,
                                queue_ms=(began - issued) * 1000,
                                selection_ms=report.selection_ms,
                                delivery_ms=report.delivery_ms,
                                total_ms=(time.perf_counter() - issued) * 1000,
                                sent=report.sent,
                            )
                        )

                    for owner in range(group * 100, group * 100 + 5):
                        command = Update.model_validate(
                            {
                                "update_id": owner,
                                "message": {
                                    "message_id": 1,
                                    "date": datetime.now(UTC),
                                    "chat": {"id": owner, "type": "private"},
                                    "text": "/latest",
                                    "from": {
                                        "id": owner,
                                        "is_bot": False,
                                        "first_name": "Benchmark",
                                    },
                                },
                            }
                        )
                        await queue.receive(owner, command)

                    async def feed(bot, update, request=request):
                        await request(update.message.chat.id)

                    async def deliver_queued(queue=queue, feed=feed):
                        dispatcher = SimpleNamespace(feed_update=feed)
                        while claim := await queue.claim():
                            await queue.handle(claim, dispatcher, None, {})

                    await asyncio.gather(
                        *(deliver_queued() for _ in range(1 if mode == "serial" else 4))
                    )
                    if len(records) != 5:
                        raise RuntimeError("An accepted command did not finish")
                    results.append(
                        dict(
                            mode=mode,
                            slow_owner_http_seconds=delay,
                            repeat=repeat,
                            owners=sorted(records, key=lambda x: x["owner"]),
                            network=network,
                        )
                    )
                    (output / "report.json").write_text(
                        json.dumps(results, indent=2), encoding="utf-8"
                    )
                    print(
                        mode, delay, repeat, round(max(x["total_ms"] for x in records)), flush=True
                    )
    finally:
        if engine is not None:
            await engine.dispose()
        async with admin.begin() as s:
            await s.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(measure(parser.parse_args().output))
