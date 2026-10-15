"""Fact history shares its caller's transaction, never commits independently."""

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from domain.professional import ProfessionalFacts
from storage.models import FactRevisionModel


class FactRepository:
    def __init__(self, session):
        self.session = session
        self.insert = (
            pg_insert if session.get_bind().dialect.name == "postgresql" else sqlite_insert
        )

    async def record(self, subject_kind, subject_id, facts):
        statement = (
            self.insert(FactRevisionModel)
            .values(
                subject_kind=subject_kind,
                subject_id=str(subject_id),
                revision=facts.revision,
                input_revision=facts.input_sha256,
                schema_version=facts.schema_version,
                payload=facts.model_dump(mode="json"),
                recorded_at=datetime.now(UTC),
            )
            .on_conflict_do_nothing(index_elements=["subject_kind", "subject_id", "revision"])
        )
        await self.session.execute(statement)

    async def history(self, subject_kind, subject_id):
        rows = await self.session.scalars(
            select(FactRevisionModel)
            .where(
                FactRevisionModel.subject_kind == subject_kind,
                FactRevisionModel.subject_id == str(subject_id),
            )
            .order_by(FactRevisionModel.recorded_at, FactRevisionModel.revision)
        )
        return [ProfessionalFacts.model_validate(row.payload) for row in rows]
