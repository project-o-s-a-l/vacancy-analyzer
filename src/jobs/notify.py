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
from domain.exchange_rates import use_rates
from domain.matching import MatchResult
from domain.serving_ranker import RankerPolicy
from domain.vacancy_duplicates import vacancy_duplicate_key
from jobs.catalog_freshness import RECHECK_AFTER, available_for_delivery
from jobs.exchange_rates import rates_for
from jobs.feedback_context import delivery_context
from jobs.rank import rank_recent
from jobs.recommendations import recommend
from jobs.saved_selections import vacancy_revision
from jobs.serving_compute import (
    RecommendationBusy,
    bounded_recommendations,
    compute,
    owner_selection,
)
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
    selection_ms: float = 0
    delivery_ms: float = 0
    pending_message_id: int | None = None


class _ReviewPending(Exception):
    def __init__(self, message_id: int | None):
        self.message_id = message_id
        super().__init__(
            "Оцените эту вакансию кнопкой «👍 Интересно» или «👎 Не подходит», "
            "чтобы получить следующую."
            if message_id is not None
            else "Карточка уже отправляется. Дождитесь её в чате."
        )


class NotificationService:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        profile: CandidateProfile,
        *,
        chat_id: int,
        top: int = 5,
        wait_for_rating: bool = False,
        min_score: float = 50,
        fresh_days: int | None = None,
        check_interval=RECHECK_AFTER,
        ranker_policy: RankerPolicy | None = None,
        saved_selections_enabled: bool = True,
        expected_profile_revision: int | None = None,
        expected_owner_epoch: int | None = None,
    ) -> None:
        if chat_id < 1 or not 1 <= top <= 20 or not 0 <= min_score <= 100:
            raise ValueError("Invalid notification settings")
        self.sessions = sessions
        self.profile = profile
        self.chat_id = chat_id
        self.wait_for_rating = wait_for_rating
        self.top = 1 if wait_for_rating else top
        self.min_score = min_score
        if fresh_days is not None and fresh_days < 1:
            raise ValueError("Invalid freshness window")
        self.fresh_days = fresh_days
        self.check_interval = check_interval
        self.ranker_policy = ranker_policy or RankerPolicy()
        self.saved_selections_enabled = saved_selections_enabled
        self.expected_profile_revision = expected_profile_revision
        self.expected_owner_epoch = expected_owner_epoch

    async def _profile_current(self, session, user):
        from nlp.professional_facts import profile_input_revision
        from nlp.profile_experience import sync_profile_experience
        from scoring.serving_features import profile_revision
        from storage.repositories.selection_repository import SelectionRepository
        from storage.repositories.user_repository import read_profile

        if self.expected_profile_revision is not None and (
            user is None or user.profile_revision != self.expected_profile_revision
        ):
            return False
        if self.expected_owner_epoch is not None and (
            await SelectionRepository(session).owner_epoch(self.chat_id)
            != self.expected_owner_epoch
        ):
            return False
        if user is None:
            return not self.wait_for_rating
        # Recommendations use the same derived experience as the profile editor.
        current = sync_profile_experience(read_profile(user))
        return profile_input_revision(current) == profile_input_revision(self.profile) and (
            self.profile.professional_facts is None
            or profile_revision(current) == profile_revision(self.profile)
        )

    @observed("matching")
    async def pending(self) -> list[tuple[Vacancy, MatchResult]]:
        selected, _ = await self._selection()
        return selected

    async def _selection(self) -> tuple[list[tuple[Vacancy, MatchResult]], str | None]:
        try:
            with use_rates(await rates_for(self.profile)):
                return await self._select()
        except RecommendationBusy as exc:
            return [], str(exc)

    @bounded_recommendations
    async def _select(self) -> tuple[list[tuple[Vacancy, MatchResult]], str | None]:
        started = time.monotonic()
        diagnostics = {}
        async with owner_selection(self.sessions, self.chat_id), self.sessions() as session:
            current_user = await session.get(
                TelegramUserModel, self.chat_id, populate_existing=True
            )
            if not await self._profile_current(session, current_user):
                return [], "Профиль изменился во время подбора. Запустите /latest ещё раз."
            await session.commit()
            recommendation = await recommend(
                VacancyRepository(session),
                self.profile,
                days=self.fresh_days,
                history_owner=self.chat_id,
                release_read=session.commit,
                ranker_policy=self.ranker_policy,
                cache_owner=self.chat_id if self.saved_selections_enabled else None,
                diagnostics=diagnostics,
                min_score=self.min_score,
                top=self.top,
            )
            ranking = recommendation.ranking
            # A diagnostic-only lookup distinguishes "already sent" from no
            # suitable active candidates, without letting history consume Top-K.
            had_blocked = False
            if not ranking:
                history = CandidateRepository.owner_history(self.chat_id)
                has_history = await session.scalar(
                    select(history[0].exists() | history[1].exists())
                )
                diagnostics["history_has_entries"] = int(has_history)
            if not ranking and has_history:
                if diagnostics.get("blocked_probe_known"):
                    had_blocked = bool(diagnostics["blocked_probe_has_candidate"])
                else:
                    with bind_context(stage="blocked_probe"):
                        probe = await rank_recent(
                            VacancyRepository(session),
                            self.profile,
                            days=self.fresh_days,
                            limit=1,
                            release_read=session.commit,
                        )
                    had_blocked = bool(probe)
                    if self.saved_selections_enabled:
                        from jobs.candidate_cache import remember_blocked_probe

                        await remember_blocked_probe(
                            session, self.chat_id, probe, days=self.fresh_days
                        )
            await session.commit()
        unseen, selected, displayed = (
            recommendation.unseen,
            recommendation.admitted,
            recommendation.displayed,
        )
        post_excluded = len(ranking) - len(unseen)
        async with self.sessions.begin() as session:
            user = await session.get(TelegramUserModel, self.chat_id, populate_existing=True)
            if not await self._profile_current(session, user):
                return [], "Профиль изменился во время подбора. Запустите /latest ещё раз."
            overdue = set()
            available = await available_for_delivery(
                session,
                [v.id for v, _ in displayed],
                interval=self.check_interval,
                overdue=overdue,
                history_owner=self.chat_id,
            )
            await VacancyRepository(session).attach_source_links([v for v, _ in displayed])
        changed = any(v.id not in available for v, _ in displayed)
        displayed = [(v, m) for v, m in displayed if v.id in available]
        incomplete = diagnostics.get("pool_budget_exhausted") or diagnostics.get(
            "projection_budget_exhausted"
        )
        code = (
            "candidate_changed"
            if changed
            else "retrieval_budget_exhausted"
            if incomplete
            else None
            if selected
            else (
                "no_active_candidates"
                if not ranking and not had_blocked
                else "history_feedback_blocked"
                if not unseen
                else "below_threshold"
            )
        )
        from jobs.feedback_monitoring import record_selection

        await record_selection(
            self.sessions,
            self.chat_id,
            self.profile,
            revision=self.expected_profile_revision,
            epoch=self.expected_owner_epoch,
            ranking=ranking,
            displayed=displayed,
            overdue_count=len(overdue),
            code=code,
            latency_ms=(time.monotonic() - started) * 1000,
        )
        emit_event(
            logger,
            "selection.finished",
            result="success" if displayed else "skipped",
            code=code,
            **diagnostics,
            freshness_checks_requested=len(overdue),
            skipped_count=post_excluded,
            below_threshold_count=len(unseen) - len(selected),
            selected_count=len(displayed),
            duration_ms=(time.monotonic() - started) * 1000,
        )
        if displayed:
            return displayed, None
        if changed:
            return [], "За время подбора вакансии стали недоступны. Запустите поиск ещё раз."
        if incomplete:
            return [], (
                "Поиск пока не завершён: часть карточек требует обновления или дополнительных "
                "проверок. Это не означает, что подходящих вакансий нет. Попробуйте позже."
            )
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
        elif any(m.ranking and m.ranking.applied for _, m in unseen):
            message = (
                "Новых рекомендаций пока нет: недостаточно подтверждённых сведений "
                "или профессионального соответствия для показа. "
                "Попробуйте позже или уточните навыки, опыт и задачи в /profile."
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

    async def _check_pending_review(self, session: AsyncSession) -> None:
        pending = await NotificationRepository(session).pending_review(self.chat_id)
        if pending is not None:
            raise _ReviewPending(pending.message_id if pending.status == "sent" else None)

    @observed("delivery")
    async def deliver(self, sender: RecommendationSender) -> DeliveryReport:
        try:
            if self.wait_for_rating:
                async with self.sessions.begin() as session:
                    await self._check_pending_review(session)
            with use_rates(await rates_for(self.profile)):
                return await self._deliver(sender)
        except _ReviewPending as exc:
            return DeliveryReport(empty_message=str(exc), pending_message_id=exc.message_id)

    async def _deliver(self, sender: RecommendationSender) -> DeliveryReport:
        started = time.monotonic()
        selected, empty_message = await self._selection()
        delivery_started = time.monotonic()
        report = DeliveryReport(
            empty_message=empty_message, selection_ms=(delivery_started - started) * 1000
        )
        for rank, (vacancy, match) in enumerate(selected, start=1):
            with (
                operation("delivery"),
                bind_context(
                    vacancy_id=vacancy.id,
                    source_vacancy_id=vacancy.source_id,
                    duplicate_group=vacancy_duplicate_key(vacancy),
                ),
            ):
                outcome = await self._attempt(
                    sender, vacancy, match, report.sent + 1, selection_rank=rank
                )
            if outcome == "sent":
                report.sent += 1
            elif outcome in {"failed", "uncertain"}:
                setattr(report, outcome, getattr(report, outcome) + 1)
                break
            elif outcome == "stop":
                break
        report.delivery_ms = (time.monotonic() - delivery_started) * 1000
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

    async def _attempt(self, sender, vacancy, match, rank, *, selection_rank=None):
        def skip(code):
            emit_event(logger, "delivery.skipped", result="skipped", code=code)

        # These inputs are immutable for this selection. Revalidate their revisions
        # under the owner lock before recording a claim.
        checked_at = datetime.now(UTC)
        eligible = await compute(retrieval_eligibility, vacancy, self.profile, checked_at)
        if not eligible.eligible:
            skip("candidate_changed")
            return "skipped"
        ranking_context = await compute(shown_context, vacancy, self.profile, match, rank)
        selected_revision = vacancy_revision(vacancy)
        async with self.sessions.begin() as session:
            user = await session.get(
                TelegramUserModel, self.chat_id, with_for_update=True, populate_existing=True
            )
            if not await self._profile_current(session, user):
                skip("profile_changed")
                return "stop"
            if self.wait_for_rating:
                # Recheck under the owner lock: a second sender may have won during ranking.
                await self._check_pending_review(session)
            rows = await CandidateRepository(session).load_selected([vacancy.id])
            if not rows:
                skip("candidate_changed")
                return "skipped"
            current = CandidateRepository.vacancy(rows[0])
            now = datetime.now(UTC)
            if current.id not in await available_for_delivery(
                session, [current.id], now=now, interval=self.check_interval
            ):
                skip("candidate_changed")
                return "skipped"
            from domain.vacancy_analysis import is_fresh

            if (
                vacancy_revision(current) != selected_revision
                or checked_at.date() != now.date()
                or not is_fresh(current, now=now, days=self.fresh_days)
            ):
                skip("candidate_changed")
                return "skipped"
            if await session.scalar(
                select(VacancyModel.id).where(
                    VacancyModel.id == current.id, CandidateRepository.owner_blocked(self.chat_id)
                )
            ):
                skip("feedback_blocked")
                return "skipped"
            vacancy = current
            await VacancyRepository(session).attach_source_links([vacancy])
            if match.retrieval:
                match.retrieval.source_payload_hash = rows[0]["source_payload_hash"]
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
                awaiting_rating=self.wait_for_rating,
                exposure_context={
                    "version": "recommendation-exposure-v3",
                    "profile_write_revision": self.expected_profile_revision,
                    "profile_epoch": self.expected_owner_epoch,
                    "profile": self.profile.model_dump(mode="json"),
                    "match": match.model_dump(mode="json"),
                    "rank": rank,
                    "selection_rank": selection_rank if selection_rank is not None else rank,
                    "payload_hash": source_hash,
                    "ranker_context": ranking_context,
                    "feedback_evidence": delivery_context(
                        vacancy, self.profile, match, min_score=self.min_score, top=self.top
                    ),
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
