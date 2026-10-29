from copy import deepcopy
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, or_, select
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
        exposure = await self._exposure(user_id, vacancy_id)
        message_id = exposure_message_id if exposure_message_id is not None else exposure.message_id
        statement = select(ExposureModel).where(
            ExposureModel.chat_id == user_id,
            ExposureModel.vacancy_id == vacancy_id,
        )
        if exposure_message_id is not None:
            statement = statement.where(ExposureModel.message_id == message_id)
        # Non-UI callers explicitly act on the latest confirmed snapshot, even if a
        # later redelivery failed. UI callbacks must resolve their exact message.
        snapshot = await self.session.scalar(
            statement.order_by(ExposureModel.confirmed_at.desc()).limit(1)
        )
        if snapshot is None:
            raise ValueError("Показ этой карточки не подтверждён. Откройте /history заново.")
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
        self.session.add(
            FeedbackEventModel(
                user_id=user_id,
                event_key=event_key,
                vacancy_id=vacancy_id,
                action=action,
                reason=reason,
                exposure_context=deepcopy(snapshot.context),
                action_context=deepcopy(action_context),
                created_at=now,
            )
        )
        await self.session.flush()
        after_commit(
            self.session,
            "feedback.changed",
            vacancy_id=vacancy_id,
            feedback_action=action,
            result="success",
        )
        return _state(vacancy_id, row)

    async def suppressed_ids(self, user_id: int) -> set[UUID]:
        return set(
            await self.session.scalars(
                select(FeedbackModel.vacancy_id).where(
                    FeedbackModel.user_id == user_id,
                    or_(FeedbackModel.interest == "dislike", FeedbackModel.applied.is_(True)),
                )
            )
        )

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
