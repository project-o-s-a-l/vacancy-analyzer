from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from domain import CandidateProfile
from domain.professional import PROFILE_SCHEMA, ProfessionalFacts
from nlp.professional_facts import candidate_facts, profile_input_revision, profile_payload
from storage.models import TelegramUserModel
from storage.repositories.fact_repository import FactRepository


class StaleProfileWrite(ValueError):
    """Caller must reread and reapply its change; never silently overwrite another edit."""


def read_profile(row) -> CandidateProfile:
    profile = CandidateProfile.model_validate(row.profile)
    current = profile_input_revision(profile)
    if row.professional_facts and row.facts_input_revision == current:
        facts = ProfessionalFacts.model_validate(row.professional_facts)
        if facts.owner_id != row.user_id or facts.input_sha256 != current:
            raise ValueError("Profile facts have an invalid source/owner")
    else:
        # Includes writes by an older application after an operational rollback.
        timestamp = row.updated_at
        timestamp = timestamp.replace(tzinfo=UTC) if timestamp.tzinfo is None else timestamp
        previous = (
            ProfessionalFacts.model_validate(row.professional_facts)
            if row.professional_facts
            else None
        )
        if previous and previous.owner_id != row.user_id:
            raise ValueError("Profile facts have an invalid owner")
        facts = candidate_facts(profile, row.user_id, previous=previous, reference_time=timestamp)
    return profile.model_copy(update={"professional_facts": facts})


@dataclass(frozen=True)
class RegisteredUser:
    user_id: int
    profile: CandidateProfile
    form_state: str | None
    revision: int = 0
    input_revision: str | None = None
    facts_revision: str | None = None


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
        profile = CandidateProfile()
        facts = candidate_facts(profile, user_id, reference_time=now)
        statement = (
            self.insert(TelegramUserModel)
            .values(
                user_id=user_id,
                profile=profile_payload(profile),
                profile_schema_version=PROFILE_SCHEMA,
                professional_facts=facts.model_dump(mode="json"),
                facts_input_revision=facts.input_sha256,
                form_state="setup:skills",
                created_at=now,
                updated_at=now,
            )
            .on_conflict_do_nothing(index_elements=["user_id"])
        )
        inserted = await self.session.scalar(statement.returning(TelegramUserModel.user_id))
        if inserted is not None:
            await FactRepository(self.session).record("candidate", user_id, facts)
        return await self.get(user_id)

    async def get(self, user_id: int, *, lock: bool = False) -> RegisteredUser | None:
        statement = select(TelegramUserModel).where(TelegramUserModel.user_id == user_id)
        if lock:
            statement = statement.with_for_update()
        row = await self.session.scalar(statement.execution_options(populate_existing=True))
        profile = read_profile(row) if row else None
        return (
            RegisteredUser(
                row.user_id,
                profile,
                row.form_state,
                row.profile_revision,
                profile_input_revision(CandidateProfile.model_validate(row.profile)),
                profile.professional_facts.revision,
            )
            if row
            else None
        )

    async def save(self, user: RegisteredUser, *, confirmed_fields=()) -> RegisteredUser:
        row = await self.session.get(TelegramUserModel, user.user_id, populate_existing=True)
        if row is None:
            raise ValueError("Сначала зарегистрируйтесь командой /start")
        original = read_profile(row)
        if row.profile_schema_version not in {None, PROFILE_SCHEMA}:
            raise ValueError("Unsupported profile schema; update application before writing")
        if (
            (
                user.facts_revision is not None
                and original.professional_facts.revision != user.facts_revision
            )
            or row.profile_revision != user.revision
            or (
                user.input_revision is not None
                and profile_input_revision(original) != user.input_revision
            )
        ):
            raise StaleProfileWrite("Профиль изменился; перечитайте его перед сохранением")
        profile = CandidateProfile.model_validate(user.profile.model_dump(mode="json"))
        revision = profile_input_revision(profile)
        facts = profile.professional_facts
        if facts is not None and (facts.subject != "candidate" or facts.owner_id != user.user_id):
            raise ValueError("Profile facts belong to another owner")
        if facts is None or facts.input_sha256 != revision or confirmed_fields:
            facts = candidate_facts(
                profile,
                user.user_id,
                previous=facts or original.professional_facts,
                confirmed_fields=confirmed_fields,
                reference_time=datetime.now(UTC) if confirmed_fields else None,
            )
        if facts.subject != "candidate" or facts.owner_id != user.user_id:
            raise ValueError("Profile facts belong to another owner")
        statement = (
            update(TelegramUserModel)
            .where(
                TelegramUserModel.user_id == user.user_id,
                TelegramUserModel.profile_revision == user.revision,
                TelegramUserModel.profile == row.profile,
                TelegramUserModel.professional_facts == row.professional_facts,
            )
            .values(
                profile=profile_payload(profile),
                profile_schema_version=PROFILE_SCHEMA,
                form_state=user.form_state,
                updated_at=datetime.now(UTC),
                profile_revision=user.revision + 1,
                professional_facts=facts.model_dump(mode="json"),
                facts_input_revision=revision,
            )
        )
        changed = await self.session.execute(statement.execution_options(synchronize_session=False))
        if changed.rowcount != 1:
            raise StaleProfileWrite("Профиль изменился; перечитайте его перед сохранением")
        history = FactRepository(self.session)
        await history.record("candidate", user.user_id, original.professional_facts)
        await history.record("candidate", user.user_id, facts)
        return RegisteredUser(
            user.user_id,
            profile.model_copy(update={"professional_facts": facts}),
            user.form_state,
            user.revision + 1,
            revision,
            facts.revision,
        )

    async def search_profiles(self) -> list[CandidateProfile]:
        return [user.profile for user in await self.search_users()]

    async def search_users(self, *, lock: bool = False) -> list[RegisteredUser]:
        """Ready profiles, including profiles whose individual fields are being edited."""
        statement = select(TelegramUserModel).order_by(TelegramUserModel.user_id)
        if lock:
            statement = statement.with_for_update()
        rows = await self.session.scalars(statement.execution_options(populate_existing=True))
        profiles = []
        for row in rows:
            user_id, state = row.user_id, row.form_state
            if state and state.startswith("setup:"):
                continue
            profile = read_profile(row)
            if profile.skills:
                profiles.append(
                    RegisteredUser(
                        user_id,
                        profile,
                        state,
                        row.profile_revision,
                        profile_input_revision(profile),
                        profile.professional_facts.revision,
                    )
                )
        return profiles
