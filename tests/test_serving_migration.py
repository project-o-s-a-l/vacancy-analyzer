from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import func, select
from test_candidates import job

from storage.models import ExposureModel, NotificationModel, TelegramUserModel, VacancyModel
from storage.repositories import VacancyRepository
from storage.repositories.notification_repository import NotificationRepository
from storage.repositories.user_repository import UserRepository


@pytest.mark.postgres
async def test_0020_populated_upgrade_preserves_profiles_claims_and_only_confirmed_exposures(
    postgres_sessions, vacancy
):
    sessions = postgres_sessions
    cards = [job(vacancy, title=f"Java Developer {i}") for i in range(4)]
    async with sessions.begin() as s:
        await UserRepository(s).register(42)
        notifications = NotificationRepository(s)
        for index, card in enumerate(cards):
            await VacancyRepository(s).save(card)
            token = await notifications.claim(42, card.id, exposure_context={"legacy": index})
            if index != 3:
                await notifications.finish(
                    42,
                    card.id,
                    token,
                    ("sent", "failed", "uncertain")[index],
                    message_id=100 if index == 0 else None,
                )
        profile = (await s.get(TelegramUserModel, 42)).profile
    from alembic.config import Config

    from alembic import command

    config = Config(str(Path(__file__).parents[1] / "alembic.ini"))

    def roundtrip(connection):
        config.attributes["connection"] = connection
        command.stamp(config, "head")
        command.downgrade(config, "0019")
        command.upgrade(config, "head")

    async with sessions.kw["bind"].begin() as connection:
        await connection.run_sync(roundtrip)
    async with sessions() as s:
        assert await s.scalar(select(func.count()).select_from(VacancyModel)) == 4
        assert (await s.get(TelegramUserModel, 42)).profile == profile
        states = list(await s.scalars(select(NotificationModel.status)))
        assert sorted(states) == ["failed", "sending", "sent", "uncertain"]
        snapshots = list(await s.scalars(select(ExposureModel)))
        assert len(snapshots) == 1 and snapshots[0].message_id == 100
        assert snapshots[0].context == {"legacy": 0}
        assert snapshots[0].confirmed_at <= datetime.now(UTC)
