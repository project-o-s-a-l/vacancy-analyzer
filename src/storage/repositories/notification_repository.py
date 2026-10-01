import hashlib
from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from sqlalchemy import and_, func, or_, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from observability.transactions import after_commit
from storage.models import NotificationModel, VacancyModel

DeliveryStatus = Literal["sent", "failed", "uncertain"]


class NotificationRepository:
    """Caller commits claims before contacting Telegram; ambiguous attempts stay blocked."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        dialect = session.get_bind().dialect.name
        if dialect not in {"postgresql", "sqlite"}:
            raise ValueError(f"Unsupported database dialect: {dialect}")
        self.insert = pg_insert if dialect == "postgresql" else sqlite_insert

    async def claim(
        self, chat_id: int, vacancy_id: UUID, *, exposure_context: dict | None = None
    ) -> UUID | None:
        key = await self.session.scalar(
            select(VacancyModel.duplicate_key).where(VacancyModel.id == vacancy_id)
        )
        if key is not None:
            if self.session.get_bind().dialect.name == "postgresql":
                # Serialize competing claims for DIFFERENT source IDs in the
                # same group. The caller commits before sending to Telegram.
                lock_key = int.from_bytes(
                    hashlib.sha256(f"notification:{chat_id}:{key}".encode()).digest()[:8],
                    signed=True,
                )
                await self.session.execute(
                    text("SELECT pg_advisory_xact_lock(:key)"), {"key": lock_key}
                )
            if (
                await self.session.scalar(
                    select(NotificationModel.vacancy_id)
                    .where(
                        NotificationModel.chat_id == chat_id,
                        NotificationModel.duplicate_key == key,
                        self._blocked(),
                    )
                    .limit(1)
                )
                is not None
            ):
                after_commit(
                    self.session,
                    "delivery.skipped",
                    result="skipped",
                    code="delivery_blocked",
                    vacancy_id=vacancy_id,
                    duplicate_group=key,
                )
                return None
        token = uuid4()
        statement = self.insert(NotificationModel).values(
            chat_id=chat_id,
            vacancy_id=vacancy_id,
            claim_token=token,
            status="sending",
            attempted_at=datetime.now(UTC),
            attempts=1,
            exposure_context=exposure_context,
            duplicate_key=key,
        )
        statement = statement.on_conflict_do_update(
            index_elements=["chat_id", "vacancy_id"],
            set_={
                "claim_token": token,
                "status": "sending",
                "attempted_at": statement.excluded.attempted_at,
                "attempts": NotificationModel.attempts + 1,
                "error_kind": None,
                "exposure_context": statement.excluded.exposure_context,
                "duplicate_key": statement.excluded.duplicate_key,
            },
            where=or_(
                NotificationModel.status == "failed",
                and_(
                    NotificationModel.status == "sent",
                    NotificationModel.history_reset_at.is_not(None),
                ),
            ),
        ).returning(NotificationModel.claim_token)
        claimed = (await self.session.execute(statement)).scalar_one_or_none()
        after_commit(
            self.session,
            "delivery.claimed" if claimed else "delivery.skipped",
            vacancy_id=vacancy_id,
            duplicate_group=key,
            result="ready" if claimed else "skipped",
            **({} if claimed else {"code": "delivery_blocked"}),
        )
        return claimed

    async def finish(
        self,
        chat_id: int,
        vacancy_id: UUID,
        token: UUID,
        status: DeliveryStatus,
        *,
        message_id: int | None = None,
        error_kind: str | None = None,
    ) -> None:
        if status == "sent" and message_id is None:
            raise ValueError("A successful delivery requires message_id")
        values = {
            "status": status,
            "message_id": message_id,
            "error_kind": error_kind,
            "delivered_at": datetime.now(UTC) if status == "sent" else None,
        }
        if status == "sent":
            values["history_reset_at"] = None
        statement = (
            update(NotificationModel)
            .where(
                NotificationModel.chat_id == chat_id,
                NotificationModel.vacancy_id == vacancy_id,
                NotificationModel.claim_token == token,
                NotificationModel.status == "sending",
            )
            .values(**values)
            .returning(NotificationModel.duplicate_key)
        )
        row = (await self.session.execute(statement)).one_or_none()
        if row is None:
            raise ValueError("Delivery claim no longer belongs to this attempt")
        after_commit(
            self.session,
            "delivery.persisted",
            vacancy_id=vacancy_id,
            duplicate_group=row[0],
            result=status,
        )

    async def blocked_ids(self, chat_id: int) -> set[UUID]:
        rows = await self.session.scalars(
            select(NotificationModel.vacancy_id).where(
                NotificationModel.chat_id == chat_id,
                self._blocked(),
            )
        )
        return set(rows)

    @staticmethod
    def _blocked():
        return or_(
            NotificationModel.status.in_(["sending", "uncertain"]),
            and_(NotificationModel.status == "sent", NotificationModel.history_reset_at.is_(None)),
        )

    async def blocked_duplicate_keys(self, chat_id: int) -> set[str]:
        return set(
            await self.session.scalars(
                select(NotificationModel.duplicate_key).where(
                    NotificationModel.chat_id == chat_id,
                    NotificationModel.duplicate_key.is_not(None),
                    self._blocked(),
                )
            )
        )

    @staticmethod
    def _resettable(chat_id: int, before: datetime) -> tuple:
        if before.tzinfo is None or before.utcoffset() is None:
            raise ValueError("History reset cutoff must contain a timezone")
        return (
            NotificationModel.chat_id == chat_id,
            NotificationModel.status == "sent",
            NotificationModel.history_reset_at.is_(None),
            func.coalesce(NotificationModel.delivered_at, NotificationModel.attempted_at)
            <= before.astimezone(UTC),
        )

    async def history_count(self, chat_id: int, *, before: datetime) -> int:
        return await self.session.scalar(
            select(func.count())
            .select_from(NotificationModel)
            .where(*self._resettable(chat_id, before))
        )

    async def reset_history(self, chat_id: int, *, before: datetime) -> int:
        # Keep the durable row: only confirmed deliveries may become eligible again.
        rows = await self.session.scalars(
            update(NotificationModel)
            .where(*self._resettable(chat_id, before))
            .values(history_reset_at=datetime.now(UTC))
            .returning(NotificationModel.vacancy_id)
        )
        return len(list(rows))

    async def counts(self, chat_id: int) -> dict[str, int]:
        rows = await self.session.execute(
            select(NotificationModel.status, func.count())
            .where(NotificationModel.chat_id == chat_id)
            .group_by(NotificationModel.status)
        )
        return {status: count for status, count in rows}
