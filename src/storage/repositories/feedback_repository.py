from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from domain import Vacancy
from domain.feedback import ACTIONS, REASONS, FeedbackAction, FeedbackState
from observability.transactions import after_commit
from storage.models import (
    ExposureModel,
    FeedbackEventModel,
    FeedbackModel,
    NotificationModel,
    VacancyModel,
)
from storage.repositories.vacancy_repository import _to_domain


def _state(vacancy_id: UUID, row: FeedbackModel | None) -> FeedbackState:
    return (
        FeedbackState(vacancy_id, row.interest, row.saved, row.applied, row.reason, row.version)
        if row
        else FeedbackState(vacancy_id)
    )


class FeedbackRepository:
    """Caller locks the registered user and owns the transaction."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def _exposure(self, user_id: int, vacancy_id: UUID) -> NotificationModel:
        exposure = await self.session.get(NotificationModel, (user_id, vacancy_id))
        if exposure is None or (
            exposure.status not in {"sent", "sending", "uncertain"}
            and exposure.history_reset_at is None
        ):
            raise ValueError("Эта вакансия не была показана вам ботом. Откройте /history.")
        return exposure

    async def state(self, user_id: int, vacancy_id: UUID) -> FeedbackState:
        await self._exposure(user_id, vacancy_id)
        return _state(vacancy_id, await self.session.get(FeedbackModel, (user_id, vacancy_id)))

    async def confirmed_snapshot(
        self, user_id: int, vacancy_id: UUID, message_id: int | None
    ) -> ExposureModel:
        statement = select(ExposureModel).where(
            ExposureModel.chat_id == user_id,
            ExposureModel.vacancy_id == vacancy_id,
        )
        if message_id is not None:
            statement = statement.where(ExposureModel.message_id == message_id)
        snapshot = await self.session.scalar(
            statement.order_by(ExposureModel.confirmed_at.desc()).limit(1)
        )
        if snapshot is None:
            raise ValueError("Показ этой карточки не подтверждён. Откройте /history заново.")
        return snapshot

    async def record(
        self,
        user_id: int,
        vacancy_id: UUID,
        action: FeedbackAction,
        *,
        event_key: str,
        reason: str | None = None,
        expected_version: int | None = None,
        action_context: dict | None = None,
        exposure_message_id: int | None = None,
    ) -> FeedbackState:
        if action not in ACTIONS or not 1 <= len(event_key) <= 128:
            raise ValueError("Некорректное действие")
        if reason is not None and (action != "reason" or reason not in REASONS):
            raise ValueError("Некорректная причина")
        notification = await self._exposure(user_id, vacancy_id)
        # Non-UI callers explicitly act on the latest confirmed snapshot, even if a
        # later redelivery failed. UI callbacks must resolve their exact message.
        snapshot = await self.confirmed_snapshot(user_id, vacancy_id, exposure_message_id)
        action_context = deepcopy(action_context or {})
        action_context["exposure_resolution"] = "confirmed_snapshot"
        action_context["snapshot_id"] = str(snapshot.id)
        previous = await self.session.get(FeedbackEventModel, (user_id, event_key))
        row = await self.session.get(FeedbackModel, (user_id, vacancy_id), populate_existing=True)
        if previous is not None:
            if (previous.vacancy_id, previous.action, previous.reason) != (
                vacancy_id,
                action,
                reason,
            ):
                raise ValueError("Повтор действия содержит другие данные")
            if (previous.action_context or {}).get("snapshot_id") != str(snapshot.id):
                raise ValueError("Повтор действия содержит другие данные показа")
            after_commit(
                self.session,
                "feedback.reused",
                vacancy_id=vacancy_id,
                feedback_action=action,
                result="skipped",
            )
            return _state(vacancy_id, row)
        if action == "reason" and (
            row is None or row.interest != "dislike" or row.version != expected_version
        ):
            raise ValueError("Оценка уже изменилась. Нажмите «Не подходит» заново.")
        if action == "reason":
            parent = await self.session.scalar(
                select(FeedbackEventModel)
                .where(
                    FeedbackEventModel.user_id == user_id,
                    FeedbackEventModel.vacancy_id == vacancy_id,
                    FeedbackEventModel.action.in_(["like", "dislike", "reason"]),
                )
                .order_by(
                    FeedbackEventModel.action_context["state_after"]["version"]
                    .as_integer()
                    .desc()
                    .nullslast(),
                    FeedbackEventModel.created_at.desc(),
                    FeedbackEventModel.event_key.desc(),
                )
                .limit(1)
            )
            if parent is None or (parent.action_context or {}).get("snapshot_id") != str(
                snapshot.id
            ):
                raise ValueError(
                    "Причина относится к другой карточке. Нажмите «Не подходит» заново."
                )
            action_context["reason_parent_event"] = parent.event_key
        before = _state(vacancy_id, row)
        now = datetime.now(UTC)
        if row is None:
            row = FeedbackModel(
                user_id=user_id,
                vacancy_id=vacancy_id,
                saved=False,
                applied=False,
                version=0,
                updated_at=now,
            )
            self.session.add(row)
        if action in {"like", "dislike"}:
            row.interest, row.reason = action, None
        elif action in {"save", "unsave"}:
            row.saved = action == "save"
        elif action in {"apply", "unapply"}:
            row.applied = action == "apply"
        else:
            row.reason = reason
        row.version += 1
        row.updated_at = now
        action_context["state_before"] = {
            "interest": before.interest,
            "saved": before.saved,
            "applied": before.applied,
            "reason": before.reason,
            "version": before.version,
        }
        action_context["state_after"] = {
            "interest": row.interest,
            "saved": row.saved,
            "applied": row.applied,
            "reason": row.reason,
            "version": row.version,
        }
        action_context["state_changed"] = any(
            action_context["state_before"][key] != action_context["state_after"][key]
            for key in ("interest", "saved", "applied", "reason")
        )
        self.session.add(
            FeedbackEventModel(
                user_id=user_id,
                event_key=event_key,
                vacancy_id=vacancy_id,
                action=action,
                reason=reason,
                snapshot_id=snapshot.id,
                action_context=deepcopy(action_context),
                created_at=now,
            )
        )
        advance = False
        if (
            action in {"like", "dislike"}
            and exposure_message_id is not None
            and notification.awaiting_rating
            and notification.status == "sent"
        ):
            advance = notification.claim_token == snapshot.id
            # A copy opened after this delivery can resolve a deleted Telegram card.
            # Browsing the library never starts the next delivery on its own.
            in_library = (snapshot.context or {}).get("surface") == "history"
            if advance or (in_library and snapshot.confirmed_at >= notification.delivered_at):
                notification.awaiting_rating = False
        await self.session.flush()
        after_commit(
            self.session,
            "feedback.changed",
            vacancy_id=vacancy_id,
            feedback_action=action,
            result="success",
        )
        return replace(_state(vacancy_id, row), advance_feed=advance)

    async def counts(self, user_id: int) -> dict[str, int]:
        row = (
            await self.session.execute(
                select(
                    func.count().filter(FeedbackModel.interest == "like"),
                    func.count().filter(FeedbackModel.interest == "dislike"),
                    func.count().filter(FeedbackModel.saved.is_(True)),
                    func.count().filter(FeedbackModel.applied.is_(True)),
                ).where(FeedbackModel.user_id == user_id)
            )
        ).one()
        return dict(zip(("like", "dislike", "saved", "applied"), row, strict=True))

    async def page(
        self, user_id: int, kind: str, page: int, size: int = 5
    ) -> list[tuple[Vacancy, FeedbackState]]:
        if kind not in {"saved", "applied", "history"} or not 0 <= page <= 100_000:
            raise ValueError("Некорректная страница")
        if kind == "history":
            statement = (
                select(VacancyModel)
                .join(NotificationModel)
                .where(
                    NotificationModel.chat_id == user_id,
                    NotificationModel.status == "sent",
                    NotificationModel.history_reset_at.is_(None),
                )
                .order_by(NotificationModel.delivered_at.desc(), VacancyModel.id)
            )
        else:
            criterion = FeedbackModel.saved if kind == "saved" else FeedbackModel.applied
            statement = (
                select(VacancyModel)
                .join(FeedbackModel)
                .where(FeedbackModel.user_id == user_id, criterion.is_(True))
                .order_by(FeedbackModel.updated_at.desc(), VacancyModel.id)
            )
        rows = list(await self.session.scalars(statement.offset(page * size).limit(size + 1)))
        if not rows:
            return []
        states = {
            row.vacancy_id: row
            for row in await self.session.scalars(
                select(FeedbackModel).where(
                    FeedbackModel.user_id == user_id,
                    FeedbackModel.vacancy_id.in_([row.id for row in rows]),
                )
            )
        }
        return [(_to_domain(row), _state(row.id, states.get(row.id))) for row in rows]
