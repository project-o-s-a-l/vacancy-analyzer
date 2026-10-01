from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from domain import CandidateProfile
from storage.models import TelegramUserModel


@dataclass(frozen=True)
class RegisteredUser:
    user_id: int
    profile: CandidateProfile
    form_state: str | None


class UserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        dialect = session.get_bind().dialect.name
        if dialect not in {"postgresql", "sqlite"}:
            raise ValueError(f"Unsupported database dialect: {dialect}")
        self.insert = pg_insert if dialect == "postgresql" else sqlite_insert

    async def register(self, user_id: int) -> RegisteredUser:
        if user_id < 1:
            raise ValueError("Telegram user ID must be positive")
        now = datetime.now(UTC)
        statement = (
            self.insert(TelegramUserModel)
            .values(
                user_id=user_id,
                profile=CandidateProfile().model_dump(mode="json"),
                form_state="setup:skills",
                created_at=now,
                updated_at=now,
            )
            .on_conflict_do_nothing(index_elements=["user_id"])
        )
        await self.session.execute(statement)
        return await self.get(user_id)

    async def get(self, user_id: int, *, lock: bool = False) -> RegisteredUser | None:
        statement = select(TelegramUserModel).where(TelegramUserModel.user_id == user_id)
        if lock:
            statement = statement.with_for_update()
        row = await self.session.scalar(statement.execution_options(populate_existing=True))
        return (
            RegisteredUser(
                row.user_id, CandidateProfile.model_validate(row.profile), row.form_state
            )
            if row
            else None
        )

    async def save(self, user: RegisteredUser) -> None:
        row = await self.session.get(TelegramUserModel, user.user_id)
        if row is None:
            raise ValueError("Сначала зарегистрируйтесь командой /start")
        row.profile = user.profile.model_dump(mode="json")
        row.form_state = user.form_state
        row.updated_at = datetime.now(UTC)

    async def search_profiles(self) -> list[CandidateProfile]:
        return [user.profile for user in await self.search_users()]

    async def search_users(self, *, lock: bool = False) -> list[RegisteredUser]:
        """Ready profiles, including profiles whose individual fields are being edited."""
        statement = select(
            TelegramUserModel.user_id, TelegramUserModel.profile, TelegramUserModel.form_state
        ).order_by(TelegramUserModel.user_id)
        if lock:
            statement = statement.with_for_update()
        rows = await self.session.execute(statement)
        profiles = []
        for user_id, data, state in rows:
            if state and state.startswith("setup:"):
                continue
            profile = CandidateProfile.model_validate(data)
            if profile.skills:
                profiles.append(RegisteredUser(user_id, profile, state))
        return profiles
