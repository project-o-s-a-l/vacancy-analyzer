import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import HttpUrl
from sqlalchemy import select

from domain.vacancy_duplicates import vacancy_duplicate_key
from jobs.feedback import FeedbackService
from jobs.notify import NotificationService
from storage.models import NotificationModel
from storage.repositories import VacancyRepository
from storage.repositories.notification_repository import NotificationRepository
from storage.repositories.user_repository import RegisteredUser, UserRepository


async def seed(sessions, vacancy, profile, copies=6):
    cards = [
        vacancy.model_copy(
            update={
                "id": uuid4(),
                "source_id": str(1000 + n),
                "url": HttpUrl(f"https://example.org/jobs/{1000 + n}"),
                "published_at": datetime.now(UTC) - timedelta(days=n),
                "valid_until": None,
            }
        )
        for n in range(copies)
    ]
    async with sessions.begin() as session:
        for card in cards:
            await VacancyRepository(session).save(card)
        await UserRepository(session).register(42)
        await UserRepository(session).save(RegisteredUser(42, profile, None))
    return cards


def test_identical_content_ignores_ids_dates_markup_and_skill_order(vacancy):
    first = vacancy.model_copy(
        update={"description": "<p>Разработка API</p>", "skills": ["Python", "SQL"]}
    )
    copy = first.model_copy(
        update={
            "id": uuid4(),
            "source_id": "other",
            "url": HttpUrl("https://example.org/jobs/2"),
            "description": "РАЗРАБОТКА  api",
            "skills": ["sql", "python", "python"],
            "published_at": datetime.now(UTC),
            "raw_payload": {"different": True},
        }
    )
    assert vacancy_duplicate_key(first) == vacancy_duplicate_key(copy)


@pytest.mark.parametrize(
    "change",
    [
        {"company": "Другой работодатель"},
        {"location": "Другой город"},
        {"salary_from": 1, "salary_to": 2},
        {"remote": False},
        {"description": "Совсем другие обязанности"},
        {"experience": "6+"},
        {"company_url": "https://hh.ru/employer/999"},
    ],
)
def test_different_employer_or_conditions_are_separate(vacancy, change):
    changed = type(vacancy).model_validate(vacancy.model_dump() | change)
    assert vacancy_duplicate_key(vacancy) != vacancy_duplicate_key(changed)


def test_sparse_cards_are_not_merged(vacancy):
    assert (
        vacancy_duplicate_key(
            vacancy.model_copy(update={"description": None, "description_html": None})
        )
        is None
    )


async def test_six_source_ids_give_one_recommendation_across_restarts(
    db_sessions, vacancy, profile
):
    sessions = db_sessions
    await seed(sessions, vacancy, profile)
    service = NotificationService(sessions, profile, chat_id=42, min_score=0)
    assert len(await service.pending()) == 1
    sender = SimpleNamespace(send_recommendation=AsyncMock(return_value=1))
    assert (await service.deliver(sender)).sent == 1
    assert (
        await NotificationService(sessions, profile, chat_id=42, min_score=0).deliver(sender)
    ).sent == 0
    sender.send_recommendation.assert_awaited_once()
    # A different user still receives their own recommendation.
    assert len(await NotificationService(sessions, profile, chat_id=43, min_score=0).pending()) == 1


@pytest.mark.parametrize("status", ["sending", "sent", "uncertain", "failed"])
async def test_group_claim_respects_delivery_outcome(db_sessions, vacancy, profile, status):
    sessions = db_sessions
    cards = await seed(sessions, vacancy, profile, copies=2)
    async with sessions.begin() as session:
        repo = NotificationRepository(session)
        token = await repo.claim(42, cards[0].id)
        if status != "sending":
            await repo.finish(
                42, cards[0].id, token, status, message_id=1 if status == "sent" else None
            )
    async with sessions.begin() as session:
        claim = await NotificationRepository(session).claim(42, cards[1].id)
        assert (claim is not None) == (status == "failed")
    pending = await NotificationService(sessions, profile, chat_id=42, min_score=0).pending()
    assert pending == []


async def test_history_reset_allows_one_copy_again(db_sessions, vacancy, profile):
    sessions = db_sessions
    cards = await seed(sessions, vacancy, profile)
    async with sessions.begin() as session:
        repo = NotificationRepository(session)
        token = await repo.claim(42, cards[0].id)
        await repo.finish(42, cards[0].id, token, "sent", message_id=1)
        assert await repo.reset_history(42, before=datetime.now(UTC) + timedelta(minutes=1)) == 1
    service = NotificationService(sessions, profile, chat_id=42, min_score=0)
    assert len(await service.pending()) == 1
    sender = SimpleNamespace(send_recommendation=AsyncMock(return_value=2))
    assert (await service.deliver(sender)).sent == 1
    assert await service.pending() == []


@pytest.mark.parametrize("action", ["dislike", "apply"])
async def test_feedback_blocks_copies_even_after_history_reset(
    db_sessions, vacancy, profile, action
):
    sessions = db_sessions
    cards = await seed(sessions, vacancy, profile)
    async with sessions.begin() as session:
        repo = NotificationRepository(session)
        token = await repo.claim(42, cards[0].id)
        await repo.finish(42, cards[0].id, token, "sent", message_id=1)
    await FeedbackService(sessions).record(42, cards[0].id, action, event_key="feedback")
    async with sessions.begin() as session:
        await NotificationRepository(session).reset_history(
            42, before=datetime.now(UTC) + timedelta(minutes=1)
        )
    assert await NotificationService(sessions, profile, chat_id=42, min_score=0).pending() == []


async def test_delivery_remembers_old_content_after_source_update(db_sessions, vacancy, profile):
    sessions = db_sessions
    cards = await seed(sessions, vacancy, profile, copies=2)
    original_key = vacancy_duplicate_key(cards[0])
    async with sessions.begin() as session:
        repo = NotificationRepository(session)
        token = await repo.claim(42, cards[0].id)
        await repo.finish(42, cards[0].id, token, "sent", message_id=1)
        await VacancyRepository(session).save(
            cards[0].model_copy(update={"description": "Теперь иная работа"})
        )
    async with sessions() as session:
        row = await session.get(NotificationModel, (42, cards[0].id))
        assert row.duplicate_key == original_key
    assert await NotificationService(sessions, profile, chat_id=42, min_score=0).pending() == []


@pytest.mark.postgres
async def test_competing_claims_on_different_source_ids_send_only_one(
    postgres_sessions, vacancy, profile
):
    cards = await seed(postgres_sessions, vacancy, profile)

    async def claim(index):
        async with postgres_sessions.begin() as session:
            return await NotificationRepository(session).claim(42, cards[index % len(cards)].id)

    claims = await asyncio.gather(*(claim(index) for index in range(12)))
    assert sum(token is not None for token in claims) == 1
    async with postgres_sessions() as session:
        rows = list(await session.scalars(select(NotificationModel)))
        assert len(rows) == 1 and rows[0].status == "sending"


def test_migration_backfills_existing_copies_and_delivery_history(vacancy):
    import importlib.util
    from pathlib import Path

    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from sqlalchemy import MetaData, Table, create_engine

    modules = []
    for path in sorted((Path(__file__).parents[1] / "alembic/versions").glob("*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
        if module.revision == "0011":
            break  # Test this migration's data transition, independent of later revisions.
    engine = create_engine("sqlite:///:memory:")
    try:
        with engine.begin() as connection:
            with Operations.context(MigrationContext.configure(connection)):
                for module in modules[:-1]:
                    module.upgrade()
                vacancies = Table("vacancies", MetaData(), autoload_with=connection)
                notifications = Table(
                    "telegram_notifications", MetaData(), autoload_with=connection
                )
                ids = [uuid4().hex, uuid4().hex]
                for index, key in enumerate(ids):
                    values = vacancy.model_dump()
                    values.update(
                        id=key,
                        source_id=str(index),
                        url=str(vacancy.url),
                        company_url=str(vacancy.company_url) if vacancy.company_url else None,
                        last_seen_at=vacancy.collected_at,
                    )
                    connection.execute(
                        vacancies.insert().values(
                            **{k: v for k, v in values.items() if k in vacancies.c}
                        )
                    )
                    connection.execute(
                        notifications.insert().values(
                            chat_id=42,
                            vacancy_id=key,
                            claim_token=uuid4().hex,
                            status="sent",
                            attempted_at=vacancy.collected_at,
                            delivered_at=vacancy.collected_at,
                            message_id=index + 1,
                            attempts=1,
                        )
                    )
                modules[-1].upgrade()
                vacancies = Table("vacancies", MetaData(), autoload_with=connection)
                notifications = Table(
                    "telegram_notifications", MetaData(), autoload_with=connection
                )
                source_keys = list(connection.scalars(select(vacancies.c.duplicate_key)))
                delivered_keys = list(connection.scalars(select(notifications.c.duplicate_key)))
                assert source_keys == delivered_keys == [vacancy_duplicate_key(vacancy)] * 2
                assert len(list(connection.scalars(select(notifications.c.message_id)))) == 2
    finally:
        engine.dispose()
