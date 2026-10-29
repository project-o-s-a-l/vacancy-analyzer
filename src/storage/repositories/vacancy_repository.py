from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import case, literal_column, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from domain import Vacancy
from domain.contracts import facts_contract
from domain.professional import ProfessionalFacts
from domain.retrieval import RETRIEVAL_VERSION
from domain.vacancy_analysis import VacancyAnalysis, VacancyCategory, analysis_input_hash
from domain.vacancy_duplicates import vacancy_duplicate_key
from nlp.professional_facts import (
    vacancy_fact_input_revision,
    vacancy_facts,
    vacancy_facts_compatible,
)
from observability.transactions import after_commit
from scoring.retrieval import document_terms, search_document
from storage.models import VacancyModel, VacancySnapshot
from storage.payloads import payload_hash
from storage.repositories.fact_repository import FactRepository


def _aware(value: datetime) -> datetime:
    # SQLite drops timezones; values written by this application are always UTC.
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _to_domain(row: VacancyModel) -> Vacancy:
    values = {name: getattr(row, name) for name in Vacancy.model_fields}
    for name in ("published_at", "collected_at"):
        values[name] = _aware(values[name])
    if values["last_checked_at"] is not None:
        values["last_checked_at"] = _aware(values["last_checked_at"])
    if values["valid_until"] is not None:
        values["valid_until"] = _aware(values["valid_until"])
    vacancy = Vacancy.model_validate(values)
    if vacancy.professional_facts is None or not vacancy_facts_compatible(
        vacancy.professional_facts, vacancy
    ):
        vacancy.professional_facts = vacancy_facts(vacancy, legacy=True)
    return vacancy


class VacancyRepository:
    """Caller owns the transaction; save never commits independently."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.last_save = None
        dialect = session.get_bind().dialect.name
        if dialect not in {"postgresql", "sqlite"}:
            raise ValueError(f"Unsupported database dialect: {dialect}")
        self.insert = pg_insert if dialect == "postgresql" else sqlite_insert

    async def save(
        self, vacancy: Vacancy, *, seen_in_search: bool = True, verified: bool = True
    ) -> UUID:
        values = vacancy.model_dump(mode="python", exclude={"professional_facts"})
        values["url"] = str(vacancy.url)
        values["company_url"] = str(vacancy.company_url) if vacancy.company_url else None
        if vacancy.analysis is not None:
            if vacancy.analysis.input_hash != analysis_input_hash(vacancy):
                raise ValueError("Analysis does not match the vacancy content")
            values["analysis"] = vacancy.analysis.model_dump(mode="json")
        observed = vacancy.collected_at
        values["last_seen_at"] = observed
        values["last_checked_at"] = observed if verified else None
        values["last_check_attempt_at"] = observed if verified else None
        values["next_check_at"] = None
        values["check_requested_at"] = None
        values["availability"] = (
            "closed"
            if vacancy.archived
            else (
                "expired" if vacancy.valid_until and vacancy.valid_until <= observed else "active"
            )
        )
        values["closed_at"] = observed if vacancy.archived else None
        values["check_error"] = None
        values["content_revision"] = analysis_input_hash(vacancy)
        # Publication age is not collection age. The first source date is immutable.
        significant = vacancy.model_dump(
            mode="json",
            exclude={
                "id",
                "source",
                "source_id",
                "collected_at",
                "last_checked_at",
                "published_at",
                "description",
                "description_html",
                "raw_payload",
                "analysis",
                "professional_facts",
            },
        )
        # The text hash already represents cleaned description content; markup
        # or tracking-only changes are retained as raw history, not new meaning.
        significant["text_revision"] = values["content_revision"]
        values["catalog_revision"] = payload_hash(significant)
        values["raw_payload_hash"] = payload_hash(vacancy.raw_payload)
        values["duplicate_key"] = vacancy_duplicate_key(vacancy)
        previous = (
            await self.session.execute(
                select(
                    VacancyModel.id,
                    VacancyModel.catalog_revision,
                    VacancyModel.raw_payload_hash,
                    VacancyModel.archived,
                    VacancyModel.search_version,
                    VacancyModel.search_revision,
                    VacancyModel.content_revision,
                    VacancyModel.revision_number,
                )
                .where(
                    VacancyModel.source == vacancy.source,
                    VacancyModel.source_id == vacancy.source_id,
                )
                .with_for_update()
            )
        ).one_or_none()
        if (
            previous
            and previous.catalog_revision == values["catalog_revision"]
            and previous.raw_payload_hash == values["raw_payload_hash"]
            and previous.archived == vacancy.archived
            and previous.search_version == RETRIEVAL_VERSION
            and previous.search_revision == values["content_revision"]
            and vacancy.analysis is None
        ):
            # Identical source: touch observation metadata without rewriting HTML,
            # normalised fields, analysis or snapshot payloads.
            metadata = {
                name: values[name]
                for name in (
                    "last_checked_at",
                    "availability",
                    "check_error",
                    "last_check_attempt_at",
                    "next_check_at",
                    "check_requested_at",
                )
            }
            if not verified:
                # Offline replay of identical content cannot certify or invalidate a live check.
                metadata = {}
            if seen_in_search:
                metadata["last_seen_at"] = observed
            metadata["closed_at"] = case(
                (VacancyModel.archived.is_(True), VacancyModel.closed_at), else_=None
            )
            await self.session.execute(
                update(VacancyModel).where(VacancyModel.id == previous.id).values(**metadata)
            )
            self._saved(
                vacancy, values, previous.id, "unchanged", "reused", previous.revision_number
            )
            await self.refresh_facts(previous.id)
            return previous.id
        if (
            previous is None
            or previous.search_version != RETRIEVAL_VERSION
            or previous.search_revision != values["content_revision"]
            or previous.content_revision != values["content_revision"]
        ):
            document = search_document(
                vacancy.model_dump(include={"title", "skills", "description", "description_html"})
            )
            values.update(
                search_version=RETRIEVAL_VERSION,
                search_revision=values["content_revision"],
                search_document=document,
                search_terms=document_terms(document),
            )
        statement = self.insert(VacancyModel).values(**values)
        # Stable UUID and first collection timestamp survive repeated collection.
        updates = {
            name: getattr(statement.excluded, name)
            for name in values
            if name not in {"id", "source", "source_id", "collected_at", "published_at"}
        }
        if not seen_in_search:
            updates.pop("last_seen_at")
        updates["revision_number"] = case(
            (VacancyModel.catalog_revision.is_(None), VacancyModel.revision_number),
            (
                VacancyModel.catalog_revision == statement.excluded.catalog_revision,
                VacancyModel.revision_number,
            ),
            else_=VacancyModel.revision_number + 1,
        )
        updates["closed_at"] = case(
            (
                statement.excluded.archived.is_(True),
                case((VacancyModel.closed_at.is_not(None), VacancyModel.closed_at), else_=observed),
            ),
            else_=None,
        )
        if vacancy.analysis is None:
            # Preserve a concurrent worker's valid result atomically, not using
            # the earlier SELECT snapshot. Changed input invalidates old analysis.
            updates["analysis"] = case(
                (
                    VacancyModel.analysis["input_hash"].as_string()
                    == statement.excluded.content_revision,
                    VacancyModel.analysis,
                ),
                else_=None,
            )
        statement = statement.on_conflict_do_update(
            index_elements=["source", "source_id"], set_=updates
        )
        columns = [VacancyModel.id, VacancyModel.revision_number]
        if self.insert is pg_insert:
            columns.append(literal_column("xmax = 0").label("inserted"))
        saved_row = (await self.session.execute(statement.returning(*columns))).one()
        vacancy_id = saved_row.id
        snapshot = self.insert(VacancySnapshot).values(
            id=uuid4(),
            vacancy_id=vacancy_id,
            payload_hash=values["raw_payload_hash"],
            raw_payload=vacancy.raw_payload,
            observed_at=vacancy.collected_at,
        )
        snapshot_id = (
            await self.session.execute(
                snapshot.on_conflict_do_nothing(
                    index_elements=["vacancy_id", "payload_hash"]
                ).returning(VacancySnapshot.id)
            )
        ).scalar_one_or_none()
        inserted = saved_row.inserted if self.insert is pg_insert else previous is None
        change = (
            "new"
            if inserted
            else (
                "unchanged"
                if (previous and previous.catalog_revision == values["catalog_revision"])
                or (previous is None and saved_row.revision_number == 1)
                else "changed"
            )
        )
        self._saved(
            vacancy,
            values,
            vacancy_id,
            change,
            "new" if snapshot_id else "reused",
            saved_row.revision_number,
        )
        await self.refresh_facts(vacancy_id)
        return vacancy_id

    async def refresh_facts(self, vacancy_id: UUID, *, force=False, supplied=None) -> bool:
        """Facts, immutable history and lexical projection commit atomically with the source."""
        row = await self.session.get(
            VacancyModel, vacancy_id, with_for_update=True, populate_existing=True
        )
        if row is None:
            return False
        vacancy = _to_domain(row)
        row.content_revision = analysis_input_hash(vacancy)
        expected = vacancy_fact_input_revision(vacancy)
        await self.archive_source(vacancy)
        if supplied is not None:
            supplied = ProfessionalFacts.model_validate(supplied.model_dump(mode="json"))
            if supplied.subject != "vacancy" or supplied.input_sha256 != expected:
                return False
            if supplied.source_revision != expected:
                return False
            if supplied.document_id != f"vacancy:{vacancy.source}:{vacancy.source_id}":
                return False
            from processing.cleaner import clean_html

            supplied.verify_text(
                supplied.document_id,
                expected,
                {
                    "title": vacancy.title,
                    "description": clean_html(vacancy.description or vacancy.description_html),
                },
            )
            facts = supplied
        elif (
            row.professional_facts
            and row.facts_input_revision == expected
            and not force
            and row.professional_facts["extractor_contract"] == facts_contract()
            and vacancy_facts_compatible(
                ProfessionalFacts.model_validate(row.professional_facts), vacancy
            )
        ):
            facts = vacancy.professional_facts
            from scoring.fact_projection import projection_contract

            if (
                row.projection_contract == projection_contract()
                and row.search_version == RETRIEVAL_VERSION
                and row.search_revision == analysis_input_hash(vacancy)
                and row.search_document is not None
                and row.search_terms is not None
                and row.facts_revision == facts.revision
                and row.projection_facts_revision == facts.revision
            ):
                return True
        else:
            facts = vacancy_facts(vacancy)
        if row.professional_facts:
            await FactRepository(self.session).record(
                "vacancy", vacancy_id, ProfessionalFacts.model_validate(row.professional_facts)
            )
        await FactRepository(self.session).record("vacancy", vacancy_id, facts)
        row.professional_facts = facts.model_dump(mode="json")
        row.facts_input_revision = expected
        row.facts_revision = facts.revision
        row.facts_status = (
            "partial"
            if facts.input_scope == "partial"
            else "complete"
            if facts.model_id
            else "rules"
        )
        from scoring.fact_projection import fact_projection, projection_revision

        document = fact_projection(vacancy, facts)
        row.search_document = document
        row.search_terms = document_terms(document)
        row.search_version = RETRIEVAL_VERSION
        row.search_revision = analysis_input_hash(vacancy)
        row.projection_revision = projection_revision(vacancy, facts)
        from scoring.fact_projection import projection_contract

        row.projection_contract = projection_contract()
        row.projection_facts_revision = facts.revision
        await self.session.flush()
        return True

    async def set_facts(self, vacancy_id, facts, *, expected_facts_revision):
        """extraction CAS: bind both source revision and the facts being replaced."""
        row = await self.session.get(
            VacancyModel, vacancy_id, with_for_update=True, populate_existing=True
        )
        if row is None:
            return False
        current = (
            ProfessionalFacts.model_validate(row.professional_facts).revision
            if row.professional_facts
            else None
        )
        if current != expected_facts_revision:
            return False
        return await self.refresh_facts(vacancy_id, supplied=facts)

    def _saved(self, vacancy, values, key, change, snapshot, revision=None):
        self.last_save = {"change_kind": change}
        after_commit(
            self.session,
            "catalog.saved",
            vacancy_id=key,
            source_vacancy_id=vacancy.source_id,
            change_kind=change,
            snapshot_state=snapshot,
            availability=values["availability"],
            content_revision=values["content_revision"],
            raw_revision=values["raw_payload_hash"],
            duplicate_group=values["duplicate_key"],
            revision_number=revision,
            result="success",
        )

    async def get_by_id(self, vacancy_id: UUID) -> Vacancy | None:
        row = await self.session.get(VacancyModel, vacancy_id, populate_existing=True)
        return _to_domain(row) if row is not None else None

    async def get_by_source(self, source: str, source_id: str) -> Vacancy | None:
        row = await self.session.scalar(
            select(VacancyModel)
            .where(VacancyModel.source == source, VacancyModel.source_id == source_id)
            .execution_options(populate_existing=True)
        )
        return _to_domain(row) if row is not None else None

    async def set_analysis(
        self, vacancy_id: UUID, analysis: VacancyAnalysis, *, expected_facts_revision=None
    ) -> bool:
        row = await self.session.get(
            VacancyModel, vacancy_id, with_for_update=True, populate_existing=True
        )
        if row is None or analysis.input_hash != analysis_input_hash(_to_domain(row)):
            return False
        if expected_facts_revision is not None and row.facts_revision != expected_facts_revision:
            return False
        row.analysis = analysis.model_dump(mode="json")
        await self.session.flush()
        await self.refresh_facts(vacancy_id, force=True)
        return True

    async def archive_source(self, vacancy):
        from domain.professional import digest
        from storage.models.vacancy import VacancyFactSource

        payload = vacancy.model_dump(
            mode="json",
            exclude={"professional_facts", "analysis", "raw_payload", "collected_at", "id"},
        )
        raw_hash = payload_hash(vacancy.raw_payload)
        await self.session.execute(
            self.insert(VacancyFactSource)
            .values(
                vacancy_id=vacancy.id,
                revision=digest({"source": payload, "raw": raw_hash}),
                input_revision=vacancy_fact_input_revision(vacancy),
                content_revision=analysis_input_hash(vacancy),
                raw_payload_hash=raw_hash,
                source_payload=payload,
                recorded_at=vacancy.collected_at,
            )
            .on_conflict_do_nothing(index_elements=["vacancy_id", "revision"])
        )

    async def source_history(self, vacancy_id, *, input_revision=None):
        from storage.models.vacancy import VacancyFactSource

        statement = select(VacancyFactSource).where(VacancyFactSource.vacancy_id == vacancy_id)
        if input_revision is not None:
            statement = statement.where(VacancyFactSource.input_revision == input_revision)
        return list(await self.session.scalars(statement.order_by(VacancyFactSource.recorded_at)))

    async def projection_state(self, vacancy_id):
        """API: distinguish searchable rules from enrichment and stale/missing fields."""
        from scoring.fact_projection import projection_contract, projection_revision

        row = await self.session.get(VacancyModel, vacancy_id, populate_existing=True)
        if row is None:
            return None
        vacancy = _to_domain(row)
        ready = bool(
            row.search_document is not None
            and row.search_terms is not None
            and row.search_revision == analysis_input_hash(vacancy)
            and row.facts_input_revision == vacancy_fact_input_revision(vacancy)
            and row.projection_contract == projection_contract()
            and row.facts_revision == vacancy.professional_facts.revision
            and row.projection_facts_revision == row.facts_revision
            and row.projection_revision == projection_revision(vacancy, vacancy.professional_facts)
        )
        return dict(
            status="ready" if ready else "stale" if row.search_document else "missing",
            facts_status=row.facts_status,
            facts_revision=row.facts_revision,
            projection_revision=row.projection_revision,
            source_revision=row.content_revision,
            document=row.search_document if ready else None,
        )

    async def list_recent(
        self,
        *,
        since: datetime | None = None,
        limit: int = 500,
        category: VacancyCategory | None = None,
    ) -> list[Vacancy]:
        if limit < 1:
            raise ValueError("limit must be positive")
        if since is not None and (since.tzinfo is None or since.utcoffset() is None):
            raise ValueError("since must contain a timezone")
        statement = select(VacancyModel)
        if since is not None:
            statement = statement.where(VacancyModel.published_at >= since.astimezone(UTC))
        if category is not None:
            statement = statement.where(
                VacancyModel.analysis["extraction"]["category"].as_string() == category.value
            )
        rows = await self.session.scalars(
            statement.order_by(VacancyModel.published_at.desc(), VacancyModel.source_id)
            .limit(limit)
            .execution_options(populate_existing=True)
        )
        return [_to_domain(row) for row in rows]

    async def history(self, vacancy_id: UUID) -> list[dict]:
        rows = await self.session.scalars(
            select(VacancySnapshot)
            .where(VacancySnapshot.vacancy_id == vacancy_id)
            .order_by(VacancySnapshot.observed_at, VacancySnapshot.payload_hash)
        )
        return [row.raw_payload for row in rows]

    async def matching_sample(
        self, *, since: datetime, until: datetime, limit: int = 500
    ) -> tuple[list[Vacancy], bool]:
        if (
            limit < 1
            or since.tzinfo is None
            or until.tzinfo is None
            or since.utcoffset() is None
            or until.utcoffset() is None
            or since > until
        ):
            raise ValueError("Invalid matching window or limit")
        rows = await self.session.scalars(
            select(VacancyModel)
            .where(
                VacancyModel.published_at >= since.astimezone(UTC),
                VacancyModel.published_at <= until.astimezone(UTC),
                VacancyModel.archived.is_(False),
                or_(VacancyModel.valid_until.is_(None), VacancyModel.valid_until > until),
            )
            .order_by(VacancyModel.published_at.desc(), VacancyModel.source, VacancyModel.source_id)
            .limit(limit + 1)
            .execution_options(populate_existing=True)
        )
        records = list(rows)
        return [_to_domain(row) for row in records[:limit]], len(records) > limit

    async def skill_gap_sample(
        self, *, since: datetime, until: datetime, limit: int = 2000
    ) -> tuple[list[Vacancy], bool]:
        if limit < 1 or since.tzinfo is None or until.tzinfo is None or since > until:
            raise ValueError("Invalid analysis window or limit")
        since, until = since.astimezone(UTC), until.astimezone(UTC)
        from storage.repositories.candidate_repository import CandidateRepository

        # needs the same facts and conditions as ranking, never raw source payloads.
        rows = (
            (
                await self.session.execute(
                    select(*CandidateRepository.projection(analysis=True))
                    .where(
                        VacancyModel.published_at >= since.astimezone(UTC),
                        VacancyModel.published_at <= until.astimezone(UTC),
                        VacancyModel.archived.is_(False),
                        or_(VacancyModel.valid_until.is_(None), VacancyModel.valid_until > until),
                    )
                    .order_by(
                        VacancyModel.published_at.desc(),
                        VacancyModel.source,
                        VacancyModel.source_id,
                    )
                    .limit(limit + 1)
                )
            )
            .mappings()
            .all()
        )
        return [CandidateRepository.vacancy(row) for row in rows[:limit]], len(rows) > limit
