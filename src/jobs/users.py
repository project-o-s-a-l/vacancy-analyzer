from dataclasses import replace

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from jobs.feedback import FeedbackService
from jobs.notify import NotificationService
from jobs.profiles import EDIT_FIELDS, FIELDS, update_field
from jobs.resumes import ResumeService
from jobs.skill_gap import analyze_skill_gap
from nlp.ollama_resume import OllamaResumeAnalyzer
from observability.actions import observed
from observability.transactions import after_commit
from storage.models import (
    BackgroundJob,
    FeedbackEventModel,
    FeedbackModel,
    NotificationModel,
    ResumeModel,
    SavedSelectionModel,
    SearchPlanModel,
    TelegramUserModel,
    VacancyModel,
)
from storage.repositories import VacancyRepository
from storage.repositories.user_repository import RegisteredUser, UserRepository


class UserService:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        *,
        top: int = 5,
        min_score: float = 50,
        fresh_days: int | None = None,
        resume_analyzer: OllamaResumeAnalyzer | None = None,
        queued_resumes: bool = False,
        resume_model: str = "rules",
        ranker_policy=None,
        saved_selections_enabled: bool = True,
    ) -> None:
        self.sessions = sessions
        self.top = top
        self.min_score = min_score
        self.fresh_days = fresh_days
        self.ranker_policy = ranker_policy
        self.saved_selections_enabled = saved_selections_enabled
        self.feedback = FeedbackService(sessions)
        self.resumes = ResumeService(
            sessions, analyzer=resume_analyzer, queued=queued_resumes, model=resume_model
        )

    async def check_ready(self) -> None:
        async with self.sessions() as session:
            await session.execute(select(SavedSelectionModel.owner_id).limit(1))
            await session.execute(select(TelegramUserModel.user_id).limit(1))
            await session.execute(select(TelegramUserModel.history_reset_token).limit(1))
            await session.execute(select(FeedbackModel.user_id).limit(1))
            await session.execute(select(FeedbackEventModel.action_context).limit(1))
            await session.execute(select(NotificationModel.exposure_context).limit(1))
            await session.execute(select(NotificationModel.history_reset_at).limit(1))
            await session.execute(select(ResumeModel.user_id).limit(1))
            await session.execute(select(ResumeModel.processing_job_id).limit(1))
            await session.execute(select(BackgroundJob.id).limit(1))
            await session.execute(select(VacancyModel.analysis).limit(1))
            await session.execute(
                select(VacancyModel.availability, VacancyModel.last_checked_at).limit(1)
            )
            await session.execute(select(SearchPlanModel.user_id).limit(1))
            await session.execute(
                select(VacancyModel.search_version, VacancyModel.search_terms).limit(1)
            )
            await session.execute(
                select(VacancyModel.duplicate_key, NotificationModel.duplicate_key)
                .select_from(VacancyModel)
                .join(NotificationModel, NotificationModel.vacancy_id == VacancyModel.id)
                .limit(1)
            )

    async def register(self, user_id: int) -> RegisteredUser:
        async with self.sessions.begin() as session:
            return await UserRepository(session).register(user_id)

    async def get(self, user_id: int) -> RegisteredUser | None:
        async with self.sessions() as session:
            return await UserRepository(session).get(user_id)

    async def begin_edit(self, user_id: int, field: str) -> RegisteredUser:
        if field not in EDIT_FIELDS:
            raise ValueError("Неизвестное поле профиля")
        async with self.sessions.begin() as session:
            repository = UserRepository(session)
            user = await repository.get(user_id, lock=True)
            if user is None:
                raise ValueError("Сначала зарегистрируйтесь командой /start")
            user = replace(user, form_state="edit:" + field)
            await repository.save(user)
            return user

    @observed("history")
    async def cancel(self, user_id: int) -> RegisteredUser:
        async with self.sessions.begin() as session:
            repository = UserRepository(session)
            user = await repository.get(user_id, lock=True)
            if user is None:
                raise ValueError("Сначала зарегистрируйтесь командой /start")
            user = replace(user, form_state=None)
            await repository.save(user)
            row = await session.get(TelegramUserModel, user_id)
            if row.history_reset_token is not None:
                after_commit(
                    session, "history.cancelled", result="cancelled", code="user_cancelled"
                )
            row.history_reset_token = None
            row.history_reset_before = None
            return user

    async def accept_answer(
        self,
        user_id: int,
        text: str,
        *,
        expected_field: str | None = None,
    ) -> RegisteredUser:
        async with self.sessions.begin() as session:
            repository = UserRepository(session)
            user = await repository.get(user_id, lock=True)
            if user is None or not user.form_state:
                raise ValueError("Откройте /profile и выберите поле для изменения.")
            mode, field = user.form_state.split(":", 1)
            if expected_field is not None and expected_field != field:
                raise ValueError("Эта кнопка относится к предыдущему шагу. Откройте /profile.")
            profile = update_field(user.profile, field, text)
            next_state = None
            if mode == "setup":
                index = FIELDS.index(field)
                if index + 1 < len(FIELDS):
                    next_state = "setup:" + FIELDS[index + 1]
            user = replace(user, profile=profile, form_state=next_state)
            await repository.save(user)
            return user

    async def notifications(
        self,
        user_id: int,
        *,
        require_profile: bool = True,
    ) -> NotificationService:
        user = await self.get(user_id)
        if user is None:
            raise ValueError("Сначала зарегистрируйтесь командой /start")
        if require_profile and not user.profile.skills:
            raise ValueError("Сначала укажите навыки в /profile.")
        return NotificationService(
            self.sessions,
            user.profile,
            chat_id=user.user_id,
            top=self.top,
            min_score=self.min_score,
            fresh_days=self.fresh_days,
            ranker_policy=self.ranker_policy,
            saved_selections_enabled=self.saved_selections_enabled,
        )

    async def skill_gap(self, user_id: int):
        user = await self.get(user_id)
        if user is None:
            raise ValueError("Сначала зарегистрируйтесь командой /start")
        async with self.sessions() as session:
            return await analyze_skill_gap(VacancyRepository(session), profile=user.profile)
