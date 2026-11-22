"""Indexed lexical matching and bounded projections; never fetch source HTML payloads."""

from datetime import UTC

from sqlalchemy import (
    BigInteger,
    Text,
    Uuid,
    and_,
    any_,
    case,
    cast,
    column,
    exists,
    func,
    literal,
    not_,
    or_,
    select,
    text,
    true,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

from domain import Vacancy
from domain.exchange_rates import current_rates
from scoring.fact_projection import projection_contract
from storage.models import VacancyModel as V
from storage.models.vacancy import SEARCH_REPAIR_PREDICATE
from storage.repositories.vacancy_repository import _aware


class CandidateRepository:
    def __init__(self, session):
        self.session = session

    def _overlap(self, terms):
        if self.session.get_bind().dialect.name == "postgresql":
            return V.search_terms.op("?|")(literal(sorted(terms), type_=ARRAY(Text)))
        words = func.json_each(V.search_terms).table_valued("key", "value")
        return exists(select(1).select_from(words).where(words.c.value.in_(sorted(terms))))

    def matching_terms(self, query):
        original = self._overlap(query.terms)
        guards = query.alias_requirements()
        if not guards:
            return original
        title = V.search_document["title"]
        if self.session.get_bind().dialect.name == "postgresql":
            choices = func.jsonb_array_elements(
                literal([sorted(guard) for guard in guards], type_=JSONB)
            ).table_valued(column("value", JSONB))
            matches = exists(select(1).select_from(choices).where(title.op("@>")(choices.c.value)))
        else:
            words = func.json_each(title).table_valued("value")
            matches = or_(
                *(
                    select(func.count())
                    .select_from(words)
                    .where(words.c.value.in_(sorted(guard)))
                    .correlate(V)
                    .scalar_subquery()
                    == len(guard)
                    for guard in guards
                )
            )
        # The combined lookup still uses the existing GIN. Verify aliases against the title.
        return and_(
            self._overlap(query.terms.union(*guards)),
            or_(original, matches),
        )

    @staticmethod
    def rank_query(query):
        payload = literal(query.payload(), type_=JSONB)
        if not any(form.required for form in query.forms):
            return payload
        forms = func.jsonb_array_elements(
            literal(
                [{"terms": sorted(f.terms), "required": sorted(f.required)} for f in query.forms],
                type_=JSONB,
            )
        ).table_valued(column("value", JSONB))
        # An alias contributes only if every substituted word occurs in this title.
        # Original intents have an empty guard and always contribute ordinary partial coverage.
        valid = (
            select(func.jsonb_agg(forms.c.value["terms"]))
            .select_from(forms)
            .where(V.search_document["title"].op("@>")(forms.c.value["required"]))
            .correlate(V)
            .scalar_subquery()
        )
        return func.jsonb_set(payload, literal(["titles"], type_=ARRAY(Text)), valid)

    @staticmethod
    def projection(*, analysis=False):
        names = set(Vacancy.model_fields) - {"raw_payload"}
        if not analysis:
            names.remove("analysis")
            # residual filters require ready facts before Top-K truncation.
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
            V.projection_revision,
        ]

    @staticmethod
    def vacancy(row):
        values = {name: row[name] for name in Vacancy.model_fields if name in row}
        for name in ("published_at", "collected_at", "valid_until", "last_checked_at"):
            if values.get(name) is not None:
                values[name] = _aware(values[name])
        vacancy = Vacancy.model_validate(values | {"raw_payload": {}})
        vacancy._catalog_group = row.get("duplicate_group")
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
            V.availability.in_(("active", "unknown")),
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
                minimums = current_rates().minimum_by_currency(prefs.salary_min, prefs.currency)
                minimum = case(
                    {code: literal(value, type_=BigInteger) for code, value in minimums.items()},
                    value=func.upper(V.currency),
                )
                criteria.extend(
                    [
                        V.salary_period == "MONTH",
                        V.salary_gross.is_(prefs.salary_gross),
                        func.coalesce(V.salary_to, V.salary_from) >= minimum,
                    ]
                )
        return and_(*criteria)

    def _unblocked_predicate(self, excluded_ids, excluded_keys):
        criteria = [true()]
        pg = self.session.get_bind().dialect.name == "postgresql"
        if excluded_ids:
            criteria.append(
                not_(V.id == any_(literal(list(excluded_ids), type_=ARRAY(Uuid))))
                if pg
                else V.id.not_in(excluded_ids)
            )
        if excluded_keys:
            blocked = V.duplicate_key == any_(literal(list(excluded_keys), type_=ARRAY(Text)))
            if not pg:
                blocked = V.duplicate_key.in_(excluded_keys)
            criteria.append(or_(V.duplicate_key.is_(None), not_(blocked)))
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

    async def duplicate_keys_for_ids(self, ids):
        if not ids:
            return set()
        predicate = (
            V.id == any_(literal(list(ids), type_=ARRAY(Uuid)))
            if self.session.get_bind().dialect.name == "postgresql"
            else V.id.in_(ids)
        )
        return set(
            await self.session.scalars(
                select(V.duplicate_key).where(predicate, V.duplicate_key.is_not(None))
            )
        )

    @staticmethod
    def unknown():
        return or_(
            text("(" + SEARCH_REPAIR_PREDICATE + ")"),
            V.projection_contract.is_(None),
            V.projection_contract != projection_contract(),
            V.facts_revision.is_(None),
            V.projection_facts_revision.is_(None),
            V.projection_facts_revision != V.facts_revision,
        )

    def predicates(self, profile, options):
        predicates = [
            self._state_predicate(options["now"], options["since"], options["category"]),
            self._required_predicate(profile, options["include_ineligible"]),
            self._unblocked_predicate(options["excluded_ids"], options["excluded_keys"]),
        ]
        if (
            not options["include_ineligible"]
            and profile.preferences.locations_required
            and profile.preferences.locations
        ):
            from scoring.eligibility import location_key

            # Missing/stale projections go to bounded repair, never treated as a failed location.
            predicates.append(
                or_(
                    self.unknown(),
                    V.remote.is_(True),
                    V.search_document["location_key"]
                    .as_string()
                    .in_([location_key(v) for v in profile.preferences.locations]),
                )
            )
        return predicates

    async def stale_ids(self, profile, *, limit, **options):
        return list(
            (
                await self.session.execute(
                    select(V.id)
                    .where(*self.predicates(profile, options), self.unknown())
                    .order_by(V.id)
                    .limit(limit)
                )
            ).mappings()
        )

    def shortlist_statement(self, profile, query, parameters, *, limit, positive, **options):
        score = (
            func.va_lexical_rank_v2(
                V.search_document,
                self.rank_query(query),
                literal(parameters.model_dump(), type_=JSONB),
            )
            if positive
            else literal(0.0)
        )
        group = func.coalesce(V.duplicate_key, cast(V.id, Text))
        order = [score.desc(), V.published_at.desc(), V.source_id, V.id]
        statement = select(
            V.id,
            V.source_id,
            V.published_at,
            V.duplicate_key.label("duplicate_group"),
            V.content_revision.label("source_revision"),
            V.facts_revision,
            V.projection_revision,
            score.label("lexical_score"),
            func.row_number().over(partition_by=group, order_by=order).label("copy_number"),
            func.count().over(partition_by=group).label("copy_count"),
        ).where(*self.predicates(profile, options), not_(self.unknown()))
        if query.terms:
            statement = statement.where(
                self.matching_terms(query) if positive else not_(self.matching_terms(query))
            )
        elif positive:
            statement = statement.where(False)
        ranked = statement.subquery()
        return (
            select(ranked)
            .where(ranked.c.copy_number == 1)
            .order_by(
                ranked.c.lexical_score.desc(),
                ranked.c.published_at.desc(),
                ranked.c.source_id,
                ranked.c.id,
            )
            .limit(limit)
        )

    async def shortlist(self, profile, query, parameters, *, limit, positive, **options):
        if self.session.get_bind().dialect.name == "postgresql" and parameters.backend == "sql":
            return list(
                (
                    await self.session.execute(
                        self.shortlist_statement(
                            profile, query, parameters, limit=limit, positive=positive, **options
                        )
                    )
                ).mappings()
            )
        # Portable rollback / SQLite: score compact ready documents, never descriptions/facts.
        statement = select(
            V.id,
            V.source_id,
            V.published_at,
            V.duplicate_key.label("duplicate_group"),
            V.content_revision.label("source_revision"),
            V.facts_revision,
            V.projection_revision,
            V.search_document.label("lexical_document"),
        ).where(*self.predicates(profile, options), not_(self.unknown()))
        if query.terms:
            statement = statement.where(
                self.matching_terms(query) if positive else not_(self.matching_terms(query))
            )
        elif positive:
            return []
        from scoring.retrieval import lexical_evidence

        groups = {}
        stream = await self.session.stream(statement.execution_options(yield_per=128))
        try:
            async for row in stream.mappings():
                row = dict(row)
                row["lexical_score"] = lexical_evidence(
                    row.pop("lexical_document"), query, parameters
                ).lexical_score
                key = row["duplicate_group"] or row["id"]
                old = groups.get(key)
                row["copy_count"] = (old["copy_count"] if old else 0) + 1
                if old is None or compact_order(row) < compact_order(old):
                    groups[key] = row
                else:
                    old["copy_count"] = row["copy_count"]
        finally:
            await stream.close()
        return sorted(groups.values(), key=compact_order)[:limit]

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


def compact_order(row):
    return (
        -row["lexical_score"],
        -_aware(row["published_at"]).timestamp(),
        row["source_id"],
        str(row["id"]),
    )
