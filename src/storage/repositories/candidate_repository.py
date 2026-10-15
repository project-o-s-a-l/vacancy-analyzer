"""Indexed lexical matching and bounded projections; never fetch source HTML payloads."""

from datetime import UTC

from sqlalchemy import Text, and_, cast, exists, func, not_, or_, select, text, true
from sqlalchemy.dialects.postgresql import ARRAY, array

from domain import Vacancy
from scoring.fact_projection import projection_contract
from storage.models import VacancyModel as V
from storage.models.vacancy import SEARCH_REPAIR_PREDICATE
from storage.repositories.vacancy_repository import _aware


class CandidateRepository:
    def __init__(self, session):
        self.session = session

    def _overlap(self, terms):
        if self.session.get_bind().dialect.name == "postgresql":
            return V.search_terms.op("?|")(cast(array(sorted(terms)), ARRAY(Text)))
        words = func.json_each(V.search_terms).table_valued("key", "value")
        return exists(select(1).select_from(words).where(words.c.value.in_(sorted(terms))))

    @staticmethod
    def projection(*, analysis=False):
        names = set(Vacancy.model_fields) - {"raw_payload"}
        if not analysis:
            names.remove("analysis")
            names.remove("professional_facts")
        return [
            *(getattr(V, name) for name in sorted(names)),
            V.search_document.label("lexical_document"),
            V.search_version.label("lexical_version"),
            V.search_revision.label("lexical_revision"),
            V.content_revision.label("source_revision"),
            V.raw_payload_hash.label("source_payload_hash"),
            V.duplicate_key.label("duplicate_group"),
            V.projection_contract.label("lexical_contract"),
            V.facts_revision.label("facts_revision"),
            V.projection_facts_revision.label("projection_facts_revision"),
        ]

    @staticmethod
    def vacancy(row):
        values = {name: row[name] for name in Vacancy.model_fields if name in row}
        for name in ("published_at", "collected_at", "valid_until"):
            if values.get(name) is not None:
                values[name] = _aware(values[name])
        vacancy = Vacancy.model_validate(values | {"raw_payload": {}})
        if "professional_facts" in row:
            from nlp.professional_facts import vacancy_facts, vacancy_facts_compatible

            if vacancy.professional_facts is None or not vacancy_facts_compatible(
                vacancy.professional_facts, vacancy
            ):
                vacancy.professional_facts = vacancy_facts(vacancy, legacy=True)
        return vacancy

    @staticmethod
    def _state_predicate(now, since, category):
        criteria = [
            V.archived.is_(False),
            V.published_at <= now,
            or_(V.valid_until.is_(None), V.valid_until > now),
        ]
        if since is not None:
            criteria.append(V.published_at >= since.astimezone(UTC))
        if category is not None:
            criteria.append(V.analysis["extraction"]["category"].as_string() == category.value)
        return and_(*criteria)

    @staticmethod
    def _required_predicate(profile, include_ineligible):
        criteria = [true()]
        prefs = profile.preferences
        if not include_ineligible:
            if prefs.remote in {"required", "office"}:
                criteria.append(V.remote.is_(prefs.remote == "required"))
            if prefs.salary_required and prefs.salary_min and prefs.salary_min > 0:
                currencies = (
                    [prefs.currency, "RUR"] if prefs.currency == "RUB" else [prefs.currency]
                )
                criteria.extend(
                    [
                        func.upper(V.currency).in_(currencies),
                        V.salary_period == "MONTH",
                        V.salary_gross.is_(prefs.salary_gross),
                        or_(V.salary_to >= prefs.salary_min, V.salary_from >= prefs.salary_min),
                    ]
                )
        return and_(*criteria)

    @staticmethod
    def _unblocked_predicate(excluded_ids, excluded_keys):
        criteria = [true()]
        if excluded_ids:
            criteria.append(V.id.not_in(excluded_ids))
        if excluded_keys:
            criteria.append(or_(V.duplicate_key.is_(None), V.duplicate_key.not_in(excluded_keys)))
        return and_(*criteria)

    async def diagnostic_counts(
        self, profile, *, now, since, category, include_ineligible, excluded_ids, excluded_keys
    ):
        """One scalar aggregate, no descriptions, payloads, or user state retained."""
        active = self._state_predicate(now, since, category)
        required = self._required_predicate(profile, include_ineligible)
        unblocked = self._unblocked_predicate(excluded_ids, excluded_keys)
        counts = (
            await self.session.execute(
                select(
                    func.count(),
                    func.count().filter(active),
                    func.count().filter(active, required),
                    func.count().filter(active, required, unblocked),
                ).select_from(V)
            )
        ).one()
        total, active_count, required_count, available = counts
        return dict(
            catalog_rows=total,
            state_excluded=total - active_count,
            required_sql_excluded=active_count - required_count,
            history_feedback_excluded=required_count - available,
            available_rows=available,
        )

    async def batches(
        self,
        profile,
        query,
        *,
        now,
        since,
        category,
        include_ineligible,
        excluded_ids,
        excluded_keys,
        positive,
        batch_size,
    ):
        statement = select(*self.projection()).where(
            self._state_predicate(now, since, category),
            self._required_predicate(profile, include_ineligible),
            self._unblocked_predicate(excluded_ids, excluded_keys),
        )
        unknown = or_(
            text("(" + SEARCH_REPAIR_PREDICATE + ")"),
            V.projection_contract.is_(None),
            V.projection_contract != projection_contract(),
            V.facts_revision.is_(None),
            V.projection_facts_revision.is_(None),
            V.projection_facts_revision != V.facts_revision,
        )
        known = not_(unknown)
        if positive is None:
            statement = statement.where(unknown)
        elif query.terms:
            overlap = self._overlap(query.terms)
            statement = statement.where(known, overlap if positive else not_(overlap))
        elif positive:
            return
        else:
            statement = statement.where(known)
        statement = statement.order_by(V.published_at.desc(), V.source_id, V.id)
        stream = await self.session.stream(statement.execution_options(yield_per=batch_size))
        try:
            async for batch in stream.mappings().partitions(batch_size):
                yield batch
        finally:
            await stream.close()

    async def load_selected(self, ids):
        if not ids:
            return []
        return list(
            (
                await self.session.execute(
                    select(*self.projection(analysis=True)).where(V.id.in_(ids))
                )
            )
            .mappings()
            .all()
        )
