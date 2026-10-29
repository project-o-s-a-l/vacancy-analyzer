"""Caller owns the transaction. A racing replacement is safe: every input is verified."""

import json

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from storage.models import SavedSelectionModel, ServingRevisionModel
from storage.repositories.vacancy_repository import _aware

MAX_POOL = 128
MAX_BYTES = 4 * 1024 * 1024


class SelectionRepository:
    def __init__(self, session):
        self.session = session

    async def load(self, owner_id, revision, now):
        row = (
            await self.session.execute(
                select(
                    SavedSelectionModel.request_revision,
                    SavedSelectionModel.expires_at,
                    SavedSelectionModel.updated_at,
                    SavedSelectionModel.entries,
                ).where(SavedSelectionModel.owner_id == owner_id)
            )
        ).one_or_none()
        if (
            row is None
            or row.request_revision != revision
            or _aware(row.expires_at) <= now
            or _aware(row.updated_at) > now
            or not isinstance(row.entries, list)
            or len(row.entries) > MAX_POOL
        ):
            return []
        return row.entries

    async def owner_epoch(self, owner_id, *, lock=False):
        scope = f"profile:{owner_id}"
        if lock:
            insert = (
                pg_insert if self.session.get_bind().dialect.name == "postgresql" else sqlite_insert
            )
            await self.session.execute(
                insert(ServingRevisionModel)
                .values(scope=scope, revision=0)
                .on_conflict_do_nothing(index_elements=["scope"])
            )
        statement = select(ServingRevisionModel.revision).where(ServingRevisionModel.scope == scope)
        if lock:
            statement = statement.with_for_update()
        return (await self.session.scalar(statement)) or 0

    async def save(self, owner_id, revision, now, expires, context, entries, *, owner_epoch=None):
        if owner_id < 1:
            raise ValueError("Invalid selection owner")
        if owner_epoch is not None and await self.owner_epoch(owner_id, lock=True) != owner_epoch:
            return  # Deletion/profile mutation won; never recreate its private cache.
        # Never persist resume text, full vacancies or an unbounded candidate catalogue.
        bounded, size = [], 0
        for item in entries[:MAX_POOL]:
            size += len(json.dumps(item, ensure_ascii=False).encode("utf-8"))
            if size > MAX_BYTES:
                break
            bounded.append(item)
        insert = (
            pg_insert if self.session.get_bind().dialect.name == "postgresql" else sqlite_insert
        )
        statement = insert(SavedSelectionModel).values(
            owner_id=owner_id,
            request_revision=revision,
            updated_at=now,
            expires_at=expires,
            context=context,
            entries=bounded,
        )
        await self.session.execute(
            statement.on_conflict_do_update(
                index_elements=["owner_id"],
                set_={
                    name: getattr(statement.excluded, name)
                    for name in (
                        "request_revision",
                        "updated_at",
                        "expires_at",
                        "context",
                        "entries",
                    )
                },
            )
        )
