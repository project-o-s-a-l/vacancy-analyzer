"""Select and deliver new recommendations through a durable claim per vacancy/chat."""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from domain import CandidateProfile, Vacancy
from domain.delivery import DeliveryFailure, RecommendationSender
from domain.matching import MatchResult
from domain.serving_ranker import RankerPolicy
from domain.vacancy_duplicates import vacancy_duplicate_key
from jobs.admission import admit
from jobs.rank import rank_recent
from jobs.saved_selections import vacancy_revision
from observability import bind_context, emit_event, operation
from observability.actions import observed
from scoring.retrieval import retrieval_eligibility
from scoring.serving_features import shown_context
from storage.models import TelegramUserModel, VacancyModel
from storage.payloads import payload_hash
from storage.repositories import VacancyRepository
from storage.repositories.candidate_repository import CandidateRepository
from storage.repositories.feedback_repository import FeedbackRepository
from storage.repositories.notification_repository import DeliveryStatus, NotificationRepository

logger = logging.getLogger(__name__)


@dataclass
class DeliveryReport:
    sent: int = 0
    failed: int = 0
    uncertain: int = 0
    empty_message: str | None = None


class NotificationService:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        profile: CandidateProfile,
        *,
        chat_id: int,
        top: int = 5,
        min_score: float = 50,
        fresh_days: int | None = None,
        ranker_policy: RankerPolicy | None = None,
        saved_selections_enabled: bool = True,
    ) -> None:
        if chat_id < 1 or not 1 <= top <= 20 or not 0 <= min_score <= 100:
            raise ValueError("Invalid notification settings")
        self.sessions = sessions
        self.profile = profile
        self.chat_id = chat_id
        self.top = top
        self.min_score = min_score
        if fresh_days is not None and fresh_days < 1:
            raise ValueError("Invalid freshness window")
        self.fresh_days = fresh_days
        self.ranker_policy = ranker_policy or RankerPolicy()
        self.saved_selections_enabled = saved_selections_enabled

    @observed("matching")
    async def pending(self) -> list[tuple[Vacancy, MatchResult]]:
        selected, _ = await self._selection()
        return selected

    async def _selection(self) -> tuple[list[tuple[Vacancy, MatchResult]], str | None]:
        started = time.monotonic()
        diagnostics = {}
        async with self.sessions.begin() as session:
            blocked = await NotificationRepository(session).blocked_ids(self.chat_id)
            blocked.update(await FeedbackRepository(session).suppressed_ids(self.chat_id))
            blocked_keys = await NotificationRepository(session).blocked_duplicate_keys(
                self.chat_id
            )
            if blocked:
                blocked_keys.update(
                    await session.scalars(
                        select(VacancyModel.duplicate_key).where(
                            VacancyModel.id.in_(blocked),
                            VacancyModel.duplicate_key.is_not(None),
                        )
                    )
                )
            ranking = await rank_recent(
                VacancyRepository(session),
                self.profile,
                days=self.fresh_days,
                excluded_ids=blocked,
                excluded_keys=blocked_keys,
                ranker_policy=self.ranker_policy,
                cache_owner=self.chat_id if self.saved_selections_enabled else None,
                diagnostics=diagnostics,
            )
            # A diagnostic-only lookup distinguishes "already sent" from no
            # suitable active candidates, without letting history consume Top-K.
            had_blocked = False
            if not ranking and (blocked or blocked_keys):
                with bind_context(stage="blocked_probe"):
                    had_blocked = bool(
                        await rank_recent(
                            VacancyRepository(session), self.profile, days=self.fresh_days, limit=1
                        )
                    )
        unseen, selected, displayed = admit(
            ranking,
            min_score=self.min_score,
            top=self.top,
            blocked=blocked,
            blocked_keys=blocked_keys,
        )
        post_excluded = len(ranking) - len(unseen)
        code = (
            None
            if selected
            else (
                "no_active_candidates"
                if not ranking and not had_blocked
                else "history_feedback_blocked"
                if not unseen
                else "below_threshold"
            )
        )
        emit_event(
            logger,
            "selection.finished",
            result="success" if selected else "skipped",
            code=code,
            **diagnostics,
            skipped_count=post_excluded,
            below_threshold_count=len(unseen) - len(selected),
            selected_count=min(len(selected), self.top),
            duration_ms=(time.monotonic() - started) * 1000,
        )
        if selected:
            return displayed, None
        if not ranking and not had_blocked:
            message = (
                "Новых рекомендаций пока нет: в подборке нет активных вакансий, подходящих "
                "по обязательным условиям профиля. Попробуйте позже или проверьте /profile."
            )
        elif not unseen:
            message = (
                "Новых рекомендаций пока нет. Доступные вакансии уже отправлены "
                "или ожидают подтверждения доставки; повторно их не отправляю."
            )
        else:
            message = (
                f"Новых рекомендаций пока нет: лучшая оценка новых вакансий — "
                f"{max(match.score for _, match in unseen):.1f}%, порог — {self.min_score:g}%. "
                "Попробуйте позже или проверьте навыки, опыт и должности в /profile."
            )
        return [], message

    async def delivery_counts(self) -> dict[str, int]:
        async with self.sessions() as session:
            return await NotificationRepository(session).counts(self.chat_id)

    async def summary(self) -> dict:
        async with self.sessions() as session:
            total, recent, last_seen = (
                await session.execute(
                    select(
                        func.count(),
                        func.count().filter(
                            VacancyModel.published_at >= datetime.now(UTC) - timedelta(days=30)
                        ),
                        func.max(VacancyModel.last_seen_at),
                    )
                )
            ).one()
            sources = dict(
                (
                    await session.execute(
                        select(VacancyModel.source, func.count()).group_by(VacancyModel.source)
                    )
                ).all()
            )
            counts = await NotificationRepository(session).counts(self.chat_id)
            feedback_counts = await FeedbackRepository(session).counts(self.chat_id)
        if last_seen is not None and last_seen.tzinfo is None:
            last_seen = last_seen.replace(tzinfo=UTC)
        return {
            "total": total,
            "recent": recent,
            "last_seen": last_seen,
            "sources": sources,
            "deliveries": counts,
            "feedback": feedback_counts,
        }

    async def _finish(
        self,
        vacancy: Vacancy,
        token: UUID,
        status: DeliveryStatus,
        *,
        message_id: int | None = None,
        error_kind: str | None = None,
    ) -> None:
        try:
            async with self.sessions.begin() as session:
                await NotificationRepository(session).finish(
                    self.chat_id,
                    vacancy.id,
                    token,
                    status,
                    message_id=message_id,
                    error_kind=error_kind,
                )
        except BaseException as exc:
            emit_event(
                logger,
                "delivery.persistence_failed",
                level=logging.ERROR,
                result="uncertain",
                code="delivery_commit_failed",
                error=exc,
            )
            raise

    @observed("delivery")
    async def deliver(self, sender: RecommendationSender) -> DeliveryReport:
        started = time.monotonic()
        selected, empty_message = await self._selection()
        report = DeliveryReport(empty_message=empty_message)
        for rank, (vacancy, match) in enumerate(selected, start=1):
            if report.sent:
                await asyncio.sleep(1.05)
            with (
                operation("delivery"),
                bind_context(
                    vacancy_id=vacancy.id,
                    source_vacancy_id=vacancy.source_id,
                    duplicate_group=vacancy_duplicate_key(vacancy),
                ),
            ):
                outcome = await self._attempt(sender, vacancy, match, rank)
            if outcome == "sent":
                report.sent += 1
            elif outcome in {"failed", "uncertain"}:
                setattr(report, outcome, getattr(report, outcome) + 1)
                break
            elif outcome == "stop":
                break
        emit_event(
            logger,
            "delivery.finished",
            result=("uncertain" if report.uncertain else "failed" if report.failed else "success"),
            sent_count=report.sent,
            failed_count=report.failed,
            uncertain_count=report.uncertain,
            duration_ms=(time.monotonic() - started) * 1000,
        )
        return report

    async def _attempt(self, sender, vacancy, match, rank):
        def skip(code):
            emit_event(logger, "delivery.skipped", result="skipped", code=code)

        async with self.sessions.begin() as session:
            # Same owner lock and short transaction as before; no network under a DB lock.
            user = await session.get(
                TelegramUserModel, self.chat_id, with_for_update=True, populate_existing=True
            )
            from nlp.professional_facts import profile_input_revision
            from scoring.serving_features import profile_revision
            from storage.repositories.user_repository import read_profile

            if user is not None and (
                profile_input_revision(read_profile(user)) != profile_input_revision(self.profile)
                or self.profile.professional_facts is not None
                and profile_revision(read_profile(user)) != profile_revision(self.profile)
            ):
                skip("profile_changed")
                return "stop"
            rows = await CandidateRepository(session).load_selected([vacancy.id])
            if not rows:
                skip("candidate_changed")
                return "skipped"
            current = CandidateRepository.vacancy(rows[0])
            now = datetime.now(UTC)
            from domain.vacancy_analysis import is_fresh

            if (
                vacancy_revision(current) != vacancy_revision(vacancy)
                or not is_fresh(current, now=now, days=self.fresh_days)
                or not retrieval_eligibility(current, self.profile, now).eligible
            ):
                skip("candidate_changed")
                return "skipped"
            suppressed = await FeedbackRepository(session).suppressed_ids(self.chat_id)
            if current.id in suppressed:
                skip("feedback_blocked")
                return "skipped"
            key = vacancy_duplicate_key(current)
            if (
                key
                and suppressed
                and await session.scalar(
                    select(VacancyModel.id)
                    .where(VacancyModel.id.in_(suppressed), VacancyModel.duplicate_key == key)
                    .limit(1)
                )
            ):
                skip("feedback_blocked")
                return "skipped"
            vacancy = current
            if match.retrieval:
                match.retrieval.source_payload_hash = rows[0]["source_payload_hash"]
            ranking_context = shown_context(vacancy, self.profile, match, rank)
            source_hash = match.retrieval.source_payload_hash if match.retrieval else None
            if source_hash is None:
                raw = await session.scalar(
                    select(VacancyModel.raw_payload).where(VacancyModel.id == vacancy.id)
                )
                if raw is None:
                    skip("source_payload_missing")
                    return "skipped"
                source_hash = payload_hash(raw)
            token = await NotificationRepository(session).claim(
                self.chat_id,
                vacancy.id,
                exposure_context={
                    "version": "recommendation-exposure-v2",
                    "profile": self.profile.model_dump(mode="json"),
                    "match": match.model_dump(mode="json"),
                    "rank": rank,
                    "payload_hash": source_hash,
                    "ranker_context": ranking_context,
                },
            )
        if token is None:
            return "skipped"
        started = time.monotonic()
        emit_event(logger, "delivery.attempt_started", result="ready")

        def transport(result, code=None):
            emit_event(
                logger,
                "delivery.transport_finished",
                result=result,
                code=code,
                level=logging.INFO if result == "sent" else logging.WARNING,
                duration_ms=(time.monotonic() - started) * 1000,
            )

        try:
            message_id = await sender.send_recommendation(self.chat_id, vacancy, match)
        except asyncio.CancelledError:
            transport("uncertain", "transport_uncertain")
            await self._finish(vacancy, token, "uncertain", error_kind="CancelledError")
            raise
        except DeliveryFailure as exc:
            status = "uncertain" if exc.uncertain else "failed"
            transport(status, "transport_uncertain" if exc.uncertain else "transport_rejected")
            await self._finish(vacancy, token, status, error_kind=exc.kind)
            return status
        except Exception as exc:
            transport("uncertain", "transport_uncertain")
            await self._finish(vacancy, token, "uncertain", error_kind=type(exc).__name__)
            raise
        transport("sent")
        await self._finish(vacancy, token, "sent", message_id=message_id)
        return "sent"
