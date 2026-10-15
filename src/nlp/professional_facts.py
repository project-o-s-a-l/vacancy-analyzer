"""Adapters used by persistence and evaluation; no candidate-specific vacancy extraction."""

from datetime import UTC, datetime

from domain.contracts import facts_contract
from domain.professional import (
    Claim,
    Evidence,
    ProfessionalFacts,
    QualificationFact,
    RoleFact,
    SkillFact,
    SourceReference,
    ValueFact,
    digest,
)

ADAPTER_VERSION = "professional-adapters-v1"


def profile_payload(profile):
    data = profile.model_dump(mode="json", exclude={"professional_facts"})
    # The previous application rejects null level. Omission preserves its wire contract;
    # the authoritative fact sidecar still records missing/unknown explicitly.
    for skill in data["skills"].values():
        if skill.get("level") is None:
            skill.pop("level", None)
    return data


def profile_input_revision(profile):
    return digest(profile_payload(profile))


def vacancy_fact_input_revision(vacancy):
    from processing.cleaner import clean_html

    values = vacancy.model_dump(
        mode="json",
        exclude={
            "id",
            "raw_payload",
            "collected_at",
            "analysis",
            "professional_facts",
            "description_html",
            "description",
        },
    )
    values["description"] = clean_html(vacancy.description or vacancy.description_html)
    return digest(values)


def vacancy_facts_compatible(facts, vacancy):
    from nlp.analysis_identity import pipeline_hash

    return not facts.compatibility(
        input_revision=vacancy_fact_input_revision(vacancy), extractor_contract=facts_contract()
    ) and (
        facts.model_id is None
        or facts.analysis_pipeline == pipeline_hash("vacancy_analysis", facts.model_id)
    )


def _identity(kind, name):
    return kind + ":" + digest(name)[:24]


def _assert(value, origin, evidence=()):
    return dict(status="present", value=value, origin=origin, evidence=list(evidence))


def candidate_facts(profile, owner_id, *, previous=None, confirmed_fields=(), reference_time=None):
    """Legacy values remain intact; only the explicitly edited field becomes confirmed."""
    data = profile_payload(profile)
    revision = digest(data)
    source = SourceReference(id=f"profile:{owner_id}", revision=revision, kind="profile")
    confirmed_fields = set(confirmed_fields)

    def claim(value, field):
        confirmed = field.split(".")[0] in confirmed_fields
        return _assert(
            value,
            "user_confirmed" if confirmed else "legacy_unknown",
            [Evidence(source_id=source.id, source_revision=revision, field=field)],
        )

    facts = []
    groups = {}
    for name, skill in profile.skills.items():
        key = _identity("skill", name)
        facts.append(
            SkillFact(
                id=key,
                name=name,
                assertion=claim(True, "skills." + name),
                level=Claim(
                    status="unknown" if skill.level else "missing", legacy_value=skill.level
                ),
                years=Claim(
                    status="unknown" if skill.years is not None else "missing",
                    legacy_value=skill.years,
                ),
            )
        )
        groups[key] = "skills"
    if profile.experience_years is not None:
        facts.append(
            ValueFact(
                kind="experience_total",
                id="experience_total",
                name="years",
                assertion=claim(profile.experience_years, "experience"),
            )
        )
        groups["experience_total"] = "experience"
    for item in profile.qualifications:
        key = _identity("qualification", item.name)
        assertion = claim(True, "qualifications." + item.name)
        if item.status == "absent":
            assertion.update(status="explicit_absent", value=None)
        facts.append(
            QualificationFact(
                id=key,
                name=item.name,
                assertion=assertion,
                valid_until=Claim(**claim(item.valid_until, "qualifications." + item.name))
                if item.valid_until
                else Claim(),
            )
        )
        groups[key] = "qualifications"
    for field, value in data["preferences"].items():
        key = "preference:" + field
        edit_field = {
            "salary_min": "salary",
            "currency": "salary",
            "salary_gross": "salary",
            "salary_required": "constraints",
            "locations_required": "constraints",
        }.get(field, field)
        assertion = claim(value, edit_field) if value is not None else dict(status="missing")
        facts.append(ValueFact(kind="preference", id=key, name=field, assertion=assertion))
        groups[key] = edit_field
    sources = [source]
    if previous:
        known = {f.id: f for f in previous.facts}
        facts = [
            known[f.id]
            if f.id in known
            and groups[f.id] not in confirmed_fields
            and known[f.id].assertion.value == f.assertion.value
            else f
            for f in facts
        ]
        for fact in facts:
            old = known.get(fact.id)
            if isinstance(fact, SkillFact) and isinstance(old, SkillFact):
                # Editing membership never reviews or discards the richer level/tenure facts.
                fact.level, fact.years, fact.last_used = old.level, old.years, old.last_used
        # Rich facts/episodes survive ordinary preference edits.
        linked = {
            key for e in previous.episodes for key in [*e.role_ids, *e.skill_ids, *e.task_ids]
        }
        facts += [
            f
            for f in previous.facts
            if f.id not in groups
            and (
                f.id in linked
                or f.kind not in {"skill", "qualification", "preference", "experience_total"}
            )
        ]
        sources += previous.sources
    now = datetime.now(UTC)
    return ProfessionalFacts(
        document_id=f"candidate:{owner_id}",
        subject="candidate",
        owner_id=owner_id,
        source_revision=revision,
        input_sha256=revision,
        extractor="profile-adapter",
        extractor_version=ADAPTER_VERSION,
        extractor_contract=facts_contract(),
        extracted_at=now,
        reference_time=reference_time or (previous.reference_time if previous else now),
        input_scope="full" if confirmed_fields else "legacy",
        sources=list({(s.id, s.revision): s for s in sources}.values()),
        facts=facts,
        episodes=previous.episodes if previous else [],
        model_id=previous.model_id if previous else None,
        model_digest=previous.model_digest if previous else None,
        analysis_pipeline=previous.analysis_pipeline if previous else None,
        overlap_policy=previous.overlap_policy if previous else "unaggregated",
        uncertainty_reasons=previous.uncertainty_reasons if previous else [],
    )


def resume_facts(text, extraction, owner_id, *, version):
    if extraction.reference_time is not None:
        from nlp.resume_facts import document_facts

        return document_facts(text, extraction, owner_id, version=version)
    revision = digest(text)
    source = SourceReference(id=f"resume:{owner_id}", revision=revision, kind="resume")
    output = SourceReference(
        id=f"extraction:{owner_id}",
        revision=digest(extraction.model_dump(mode="json")),
        kind="extraction",
    )

    def evidence(value, field):
        start = text.casefold().find(str(value).casefold())
        if start >= 0:
            return [
                Evidence(
                    source_id=source.id,
                    source_revision=source.revision,
                    field="text",
                    start=start,
                    end=start + len(str(value)),
                    quote=text[start : start + len(str(value))],
                )
            ]
        # Canonical names can differ from the quote. Do not invent offsets.
        return [Evidence(source_id=output.id, source_revision=output.revision, field=field)]

    facts = [
        SkillFact(
            id=_identity("skill", s),
            name=s,
            assertion=_assert(True, "machine_proposed", evidence(s, f"skills.{i}")),
        )
        for i, s in enumerate(extraction.skills)
    ]
    facts += [
        RoleFact(
            id=_identity("role", s),
            name=s,
            context="desired",
            assertion=_assert(True, "machine_proposed", evidence(s, f"positions.{i}")),
        )
        for i, s in enumerate(extraction.positions)
    ]
    if extraction.experience_years is not None:
        facts.append(
            ValueFact(
                kind="experience_total",
                id="experience_total",
                name="years",
                assertion=_assert(
                    extraction.experience_years,
                    "machine_proposed",
                    evidence(extraction.experience_years, "experience_years"),
                ),
            )
        )
    now = datetime.now(UTC)
    result = ProfessionalFacts(
        document_id=f"resume:{owner_id}",
        subject="candidate",
        owner_id=owner_id,
        source_revision=revision,
        input_sha256=revision,
        extractor="resume-runtime",
        extractor_version=version,
        extractor_contract=facts_contract(),
        extracted_at=now,
        reference_time=now,
        input_scope="full",
        sources=[source, output],
        facts=facts,
    )
    result.verify_text(source.id, source.revision, {"text": text})
    return result


def confirm_resume_facts(profile, draft, owner_id):
    previous = profile.professional_facts
    proposed = draft.professional_facts
    if proposed is None:
        proposed = resume_facts(draft.text, draft.extraction, owner_id, version=draft.version)
    if proposed.owner_id != owner_id or proposed.input_sha256 != digest(draft.text):
        raise ValueError("Resume facts do not belong to this source/owner")
    confirmed = set()
    if draft.extraction.skills:
        confirmed.add("skills")
    if draft.extraction.positions:
        confirmed.add("positions")
    if draft.extraction.experience_years is not None:
        confirmed.add("experience")
    base = candidate_facts(
        profile,
        owner_id,
        previous=previous,
        confirmed_fields=confirmed,
        reference_time=proposed.reference_time,
    )
    # New resume replaces the career snapshot. Prior tasks/episodes remain in history.
    values = {
        f.id: f
        for f in base.facts
        if f.kind not in {"role", "task"}
        and not (f.kind == "experience_total" and f.id != "experience_total")
        and not (f.kind == "skill" and f.name not in profile.skills)
    }
    for fact in proposed.facts:
        value = fact.model_dump(mode="json")
        previous_fact = values.get(fact.id)
        if (
            fact.kind == "qualification"
            and previous_fact is not None
            and previous_fact.assertion.status == "explicit_absent"
        ):
            continue  # Imported presence cannot silently reverse an explicit profile absence.
        if (
            previous_fact is not None
            and previous_fact.assertion.origin == "user_confirmed"
            and not (fact.id == "experience_total" and "experience" in draft.edited_fields)
        ):
            # Imported text never silently overwrites a reviewed value, including a
            # manually corrected total. The draft continues to show the disagreement.
            if previous_fact.assertion.value != fact.assertion.value:
                continue
        if isinstance(fact, SkillFact) and isinstance(previous_fact, SkillFact):
            for field in ("level", "years", "last_used"):
                if (
                    getattr(fact, field).status == "missing"
                    or getattr(previous_fact, field).origin == "user_confirmed"
                ):
                    value[field] = getattr(previous_fact, field).model_dump(mode="json")
        if isinstance(fact, SkillFact):
            for field in ("level", "years", "last_used"):
                if value[field]["status"] == "present":
                    value[field]["origin"] = "user_confirmed"
        # Confirmation applies to the displayed assertions, never to missing levels.
        if value["assertion"]["status"] in {"present", "explicit_absent"}:
            value["assertion"]["origin"] = "user_confirmed"
        values[fact.id] = type(fact).model_validate(value)
    data = base.model_dump(mode="json")
    data.update(
        facts=[f.model_dump(mode="json") for f in values.values()],
        input_scope=proposed.input_scope,
        input_ranges=proposed.input_ranges,
        overlap_policy=proposed.overlap_policy,
        uncertainty_reasons=proposed.uncertainty_reasons,
        model_id=proposed.model_id,
        model_digest=proposed.model_digest,
        analysis_pipeline=proposed.analysis_pipeline,
        episodes=[e.model_dump(mode="json") for e in proposed.episodes],
        sources=list(
            {
                (s.id, s.revision): s.model_dump(mode="json")
                for s in [*base.sources, *proposed.sources]
            }.values()
        ),
    )
    return ProfessionalFacts.model_validate(data)


def vacancy_facts(vacancy, *, legacy=False):
    from nlp.vacancy_facts import extract_vacancy_facts

    return extract_vacancy_facts(vacancy, legacy=legacy)
