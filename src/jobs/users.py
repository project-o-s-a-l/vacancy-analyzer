from dataclasses import replace

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from jobs.catalog_freshness import RECHECK_AFTER
from jobs.feedback import FeedbackService
from jobs.notify import NotificationService
from jobs.profile_editor import ProfileEditor
from jobs.profiles import (
    EDIT_FIELDS,
    LEGACY_SETUP_FIELDS,
    SALARY_CURRENCIES,
    SETUP_FIELDS,
    update_field,
)
from jobs.resumes import ResumeService
from jobs.skill_gap import analyze_skill_gap
from nlp.ollama_resume import OllamaResumeAnalyzer
from observability.actions import observed
from observability.transactions import after_commit
from storage.models import (
    BackgroundJob,
    ExposureModel,
    FactBackfillModel,
    FactRevisionModel,
    FeedbackEventModel,
    FeedbackModel,
    LearningAccess,
    LearningArtifact,
    NotificationModel,
    ProfileEditModel,
    ResumeModel,
    ResumeSourceModel,
    SavedSelectionModel,
    SearchPlanModel,
    SelectionObservation,
    ServingRevisionModel,
    TelegramUpdate,
    TelegramUserModel,
    VacancyModel,
)
from storage.models.selection import RecommendationLease
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
        check_interval=RECHECK_AFTER,
        resume_analyzer: OllamaResumeAnalyzer | None = None,
        queued_resumes: bool = False,
        resume_model: str = "rules",
        ranker_policy=None,
        saved_selections_enabled: bool = True,
        profile_editor_enabled: bool = True,
    ) -> None:
        self.sessions = sessions
        self.top = top
        self.min_score = min_score
        self.fresh_days = fresh_days
        self.check_interval = check_interval
        self.ranker_policy = ranker_policy
        self.saved_selections_enabled = saved_selections_enabled
        self.feedback = FeedbackService(sessions)
        self.editor = ProfileEditor(sessions, enabled=profile_editor_enabled)
        self.resumes = ResumeService(
            sessions, analyzer=resume_analyzer, queued=queued_resumes, model=resume_model
        )

    async def check_ready(self) -> None:
        async with self.sessions() as session:
            await session.execute(select(ProfileEditModel.token).limit(1))
            # telemetry is part of every registered selection, not an optional ML extra.
            await session.execute(select(SelectionObservation.metrics).limit(1))
            await session.execute(select(LearningAccess.allowed).limit(1))
            await session.execute(select(LearningArtifact.manifest).limit(1))
            await session.execute(
                select(
                    TelegramUserModel.profile_revision,
                    TelegramUserModel.profile_schema_version,
                    TelegramUserModel.professional_facts,
                    TelegramUserModel.facts_input_revision,
                ).limit(1)
            )
            await session.execute(
                select(
                    VacancyModel.professional_facts,
                    VacancyModel.facts_input_revision,
                    VacancyModel.projection_revision,
                    VacancyModel.projection_contract,
                    VacancyModel.locations,
                    VacancyModel.source_observed_at,
                ).limit(1)
            )
            await session.execute(select(FactRevisionModel.revision).limit(1))
            await session.execute(select(FactBackfillModel.name).limit(1))
            await session.execute(
                select(
                    SavedSelectionModel.owner_id,
                    SavedSelectionModel.candidate_context,
                    SavedSelectionModel.candidate_entries,
                    SavedSelectionModel.candidate_expires_at,
                ).limit(1)
            )
            await session.execute(select(ServingRevisionModel.revision).limit(1))
            await session.execute(select(TelegramUpdate.update_id).limit(1))
            await session.execute(select(RecommendationLease.owner_id).limit(1))
            await session.execute(select(ExposureModel.id).limit(1))
            await session.execute(select(TelegramUserModel.user_id).limit(1))
            await session.execute(select(TelegramUserModel.history_reset_token).limit(1))
            await session.execute(select(FeedbackModel.user_id).limit(1))
            await session.execute(
                select(FeedbackEventModel.action_context, FeedbackEventModel.snapshot_id).limit(1)
            )
            await session.execute(select(NotificationModel.exposure_context).limit(1))
            await session.execute(
                select(NotificationModel.history_reset_at, NotificationModel.awaiting_rating).limit(
                    1
                )
            )
            await session.execute(select(ResumeModel.user_id).limit(1))
            await session.execute(select(ResumeSourceModel.user_id).limit(1))
            await session.execute(
                select(ResumeModel.processing_job_id, ResumeModel.notice_due_at).limit(1)
            )
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
            return await repository.save(user)

    @observed("history")
    async def cancel(self, user_id: int) -> RegisteredUser:
        async with self.sessions.begin() as session:
            repository = UserRepository(session)
            user = await repository.get(user_id, lock=True)
            if user is None:
                raise ValueError("Сначала зарегистрируйтесь командой /start")
            user = replace(user, form_state=None)
            user = await repository.save(user)
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
        expected_revision: int | None = None,
    ) -> RegisteredUser:
        async with self.sessions.begin() as session:
            repository = UserRepository(session)
            user = await repository.get(user_id, lock=True)
            if user is None or not user.form_state:
                raise ValueError("Откройте /profile и выберите поле для изменения.")
            if expected_revision is not None and expected_revision != user.revision:
                raise ValueError("Кнопка устарела. Откройте /start или /profile.")
            mode, field, *selection = user.form_state.split(":")
            if expected_field is not None and expected_field != field:
                raise ValueError("Эта кнопка относится к предыдущему шагу. Откройте /profile.")
            if field == "salary" and text in SALARY_CURRENCIES:
                return await repository.save(replace(user, form_state=f"{mode}:salary:{text}"))
            profile = update_field(
                user.profile, field, text, salary_currency=selection[0] if selection else None
            )
            next_state = None
            if mode == "setup":
                if field == "skills" and not profile.skills:
                    raise ValueError("Укажите хотя бы один навык или загрузите резюме.")
                steps = LEGACY_SETUP_FIELDS if field in LEGACY_SETUP_FIELDS else SETUP_FIELDS
                if field in steps[:-1]:
                    next_state = "setup:" + steps[steps.index(field) + 1]
            user = replace(user, profile=profile, form_state=next_state)
            skipped_experience = (
                mode == "setup" and field == "experience" and profile.experience_years is None
            )
            return await repository.save(
                user, confirmed_fields=() if skipped_experience else {field}
            )

    async def notifications(
        self,
        user_id: int,
        *,
        require_profile: bool = True,
        wait_for_rating: bool = False,
    ) -> NotificationService:
        from storage.repositories.selection_repository import SelectionRepository

        # Capture profile and deletion epoch under the same short owner lock.
        # Separate reads can accidentally bind an old profile to a new ABA epoch.
        async with self.sessions.begin() as session:
            user = await UserRepository(session).get(user_id, lock=True)
            if user is None:
                raise ValueError("Сначала зарегистрируйтесь командой /start")
            if require_profile and not user.profile.skills:
                raise ValueError("Сначала укажите навыки в /profile.")
            epoch = await SelectionRepository(session).owner_epoch(user_id)
        return NotificationService(
            self.sessions,
            user.profile,
            chat_id=user.user_id,
            top=self.top,
            wait_for_rating=wait_for_rating,
            min_score=self.min_score,
            fresh_days=self.fresh_days,
            check_interval=self.check_interval,
            ranker_policy=self.ranker_policy,
            saved_selections_enabled=self.saved_selections_enabled,
            expected_profile_revision=user.revision,
            expected_owner_epoch=epoch,
        )

    async def skill_gap(self, user_id: int):
        user = await self.get(user_id)
        if user is None:
            raise ValueError("Сначала зарегистрируйтесь командой /start")
        async with self.sessions() as session:
            return await analyze_skill_gap(
                VacancyRepository(session), profile=user.profile, release_read=True
            )
