import logging
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from domain import CandidateProfile, Vacancy
from domain.feedback import FeedbackAction, FeedbackState, HistoryResetRequest
from observability import emit_event
from observability.actions import observed
from observability.transactions import after_commit
from storage.models import TelegramUserModel
from storage.payloads import payload_hash
from storage.repositories import VacancyRepository
from storage.repositories.feedback_repository import FeedbackRepository
from storage.repositories.notification_repository import NotificationRepository
from storage.repositories.user_repository import UserRepository

logger = logging.getLogger(__name__)


class FeedbackService:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def _history_user(self, session: AsyncSession, user_id: int) -> TelegramUserModel:
        user = await session.scalar(
            select(TelegramUserModel).where(TelegramUserModel.user_id == user_id).with_for_update()
        )
        if user is None:
            raise ValueError("Сначала зарегистрируйтесь командой /start")
        return user

    @observed("history")
    async def request_history_reset(self, user_id: int) -> HistoryResetRequest | None:
        async with self.sessions.begin() as session:
            user = await self._history_user(session, user_id)
            before = datetime.now(UTC)
            count = await NotificationRepository(session).history_count(user_id, before=before)
            user.history_reset_token = uuid4().hex if count else None
            user.history_reset_before = before if count else None
            after_commit(
                session, "history.requested", count=count, result="ready" if count else "skipped"
            )
            return HistoryResetRequest(user.history_reset_token, count) if count else None

    @observed("history")
    async def confirm_history_reset(self, user_id: int, token: str) -> int:
        async with self.sessions.begin() as session:
            user = await self._history_user(session, user_id)
            if user.history_reset_token != token or user.history_reset_before is None:
                emit_event(
                    logger, "history.rejected", result="rejected", code="stale_history_request"
                )
                raise ValueError("Запрос устарел или уже выполнен. Откройте /history заново.")
            before = user.history_reset_before
            if before.tzinfo is None:  # SQLite stores this application's UTC values without a zone.
                before = before.replace(tzinfo=UTC)
            count = await NotificationRepository(session).reset_history(user_id, before=before)
            user.history_reset_token = None
            user.history_reset_before = None
            after_commit(session, "history.cleared", count=count, result="success")
            return count

    @observed("history")
    async def cancel_history_reset(self, user_id: int, token: str) -> None:
        async with self.sessions.begin() as session:
            user = await self._history_user(session, user_id)
            if user.history_reset_token != token:
                emit_event(
                    logger, "history.rejected", result="rejected", code="stale_history_request"
                )
                raise ValueError("Запрос устарел или уже выполнен. Откройте /history заново.")
            user.history_reset_token = None
            user.history_reset_before = None
            after_commit(session, "history.cancelled", result="cancelled")

    async def card(self, user_id: int, vacancy_id: UUID) -> tuple[Vacancy, FeedbackState]:
        async with self.sessions() as session:
            state = await FeedbackRepository(session).state(user_id, vacancy_id)
            vacancy = await VacancyRepository(session).get_by_id(vacancy_id)
            if vacancy is None:
                raise ValueError("Вакансия больше недоступна")
            return vacancy, state

    @observed("feedback")
    async def record(
        self,
        user_id: int,
        vacancy_id: UUID,
        action: FeedbackAction,
        *,
        event_key: str,
        reason: str | None = None,
        expected_version: int | None = None,
        exposure_message_id: int | None = None,
    ) -> FeedbackState:
        try:
            return await self._record(
                user_id,
                vacancy_id,
                action,
                event_key=event_key,
                reason=reason,
                expected_version=expected_version,
                exposure_message_id=exposure_message_id,
            )
        except ValueError:
            emit_event(
                logger,
                "feedback.rejected",
                vacancy_id=vacancy_id,
                feedback_action=action,
                result="rejected",
                code="feedback_not_allowed",
            )
            raise

    async def _record(
        self,
        user_id: int,
        vacancy_id: UUID,
        action: FeedbackAction,
        *,
        event_key: str,
        reason: str | None = None,
        expected_version: int | None = None,
        exposure_message_id: int | None = None,
    ) -> FeedbackState:
        async with self.sessions.begin() as session:
            # Serializes concurrent callbacks and form changes for this user on PostgreSQL.
            user = await UserRepository(session).get(user_id, lock=True)
            if user is None:
                raise ValueError("Сначала зарегистрируйтесь командой /start")
            await FeedbackRepository(session).state(user_id, vacancy_id)
            vacancy = await VacancyRepository(session).get_by_id(vacancy_id)
            if vacancy is None:
                raise ValueError("Вакансия больше недоступна")
            return await FeedbackRepository(session).record(
                user_id,
                vacancy_id,
                action,
                event_key=event_key,
                reason=reason,
                expected_version=expected_version,
                exposure_message_id=exposure_message_id,
                action_context={
                    "version": "feedback-action-v2",
                    "profile_revision": user.revision,
                    "candidate_facts_revision": user.profile.professional_facts.revision,
                    "vacancy_facts_revision": vacancy.professional_facts.revision,
                    "profile": user.profile.model_dump(mode="json"),
                    "payload_hash": payload_hash(vacancy.raw_payload),
                },
            )

    async def page(
        self, user_id: int, kind: str, page: int = 0
    ) -> tuple[CandidateProfile, list[tuple[Vacancy, FeedbackState]]]:
        async with self.sessions() as session:
            user = await UserRepository(session).get(user_id)
            if user is None:
                raise ValueError("Сначала зарегистрируйтесь командой /start")
            return user.profile, await FeedbackRepository(session).page(user_id, kind, page)
