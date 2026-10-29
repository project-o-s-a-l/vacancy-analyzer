"""User-independent facts: source + full rules pass + grounded model observations."""

import re
from datetime import UTC, datetime

from domain.contracts import facts_contract
from domain.professional import (
    Alternative,
    Evidence,
    LexicalObservation,
    ProfessionalFacts,
    QualificationFact,
    Requirement,
    RoleFact,
    SkillFact,
    SourceReference,
    TaskFact,
    ValueFact,
)
from domain.requirements import RequirementEvidence, SkillRequirement
from nlp.analysis_identity import current_vacancy_analysis
from nlp.experience import experience_bounds, experience_from_quote
from nlp.experience_requirement import _EMPLOYER, EXPERIENCE_MINIMUM
from nlp.extraction import canonical_skill, extract_skills, skill_key
from nlp.professional_facts import _assert, _identity, vacancy_fact_input_revision
from nlp.qualifications import compare_qualifications, is_qualification
from nlp.requirements import _HEADER_PATTERNS, _importance, analyze_requirements
from nlp.vacancy_document import professional_quote, professional_text, sections, source_quote
from processing.cleaner import clean_html


def reconciled_requirements(vacancy):
    """All rules evidence survives; model vocabulary is scanned through the FULL source."""
    text = vacancy.description or vacancy.description_html
    analysis = current_vacancy_analysis(vacancy)
    if analysis and (analysis.input_scope != "full" or analysis.requirements_source != "llm"):
        analysis = None
    extractions = (analysis.chunks or [analysis.extraction]) if analysis else []
    modeled = [
        item.model_copy(update={"evidence": source_quote(text, item.evidence)})
        for extraction in extractions
        for item in extraction.skills
        if professional_quote(text, item.evidence) and not is_qualification(item.name)
    ]
    _, skill_text = compare_qualifications(
        professional_text(text),
        [],
        today=vacancy.published_at.date(),
        explicit=vacancy.skills,
    )
    result = analyze_requirements(
        skill_text,
        [name for name in vacancy.skills if not is_qualification(name)],
        vocabulary=[item.name for item in modeled],
    )
    by_key = {skill_key(item.skill): item for item in result.skills}
    for item in modeled:
        key = skill_key(item.name)
        if key not in by_key:
            found = SkillRequirement(skill=canonical_skill(item.name), importance=item.importance)
            result.skills.append(found)
            by_key[key] = found
        found = by_key[key]
        observation = RequirementEvidence(
            origin="description",
            importance=item.importance,
            text=item.evidence,
            section="model_observation",
        )
        if observation not in found.evidence:
            found.evidence.append(observation)
        statuses = {e.importance for e in found.evidence} & {
            "required",
            "preferred",
            "not_required",
        }
        if len(statuses) > 1:
            found.importance = "unspecified"
            result.warnings.append("Для одного навыка найдены противоречивые требования")
    # Rules validate OR over the full source with the enriched vocabulary. Model-only
    # normalized alternatives are accepted only by the existing exact OR validator.
    from nlp.vacancy_requirements import analyzed_requirements

    if analysis:
        for extraction in extractions:
            adapted = analyzed_requirements(analysis.model_copy(update={"extraction": extraction}))
            for group in adapted.alternatives:
                if not all(professional_quote(text, e.text) for e in group.evidence):
                    continue
                if group.id not in {g.id for g in result.alternatives}:
                    result.alternatives.append(group)
                for name in group.options:
                    key = skill_key(name)
                    if key not in by_key:
                        found = SkillRequirement(
                            skill=canonical_skill(name),
                            importance="unspecified",
                            evidence=group.evidence,
                            counted_separately=False,
                        )
                        result.skills.append(found)
                        by_key[key] = found
    result.warnings = sorted(set(result.warnings))
    return result


def extract_vacancy_facts(vacancy, *, legacy=False):
    revision = vacancy_fact_input_revision(vacancy)
    source_id = f"vacancy:{vacancy.source}:{vacancy.source_id}"
    text = clean_html(vacancy.description or vacancy.description_html)
    origin = "legacy_unknown" if legacy else "machine_proposed"
    analysis = current_vacancy_analysis(vacancy)

    def evidence(field, quote=None):
        body = text if field == "description" else getattr(vacancy, field, "")
        start = body.find(quote) if quote and isinstance(body, str) else -1
        if start < 0 and quote and isinstance(body, str):
            anchored = re.search(r"\s+".join(re.escape(word) for word in quote.split()), body)
            if anchored:
                quote, start = anchored[0], anchored.start()
        return Evidence(
            source_id=source_id,
            source_revision=revision,
            field=field,
            **(dict(start=start, end=start + len(quote), quote=quote) if start >= 0 else {}),
        )

    def importance(value, ev, observations=(), claim_origin=origin):
        known = sorted(set(observations) & {"required", "preferred", "not_required"})
        if len(known) > 1:
            return dict(status="conflict", alternatives=known, origin=claim_origin, evidence=ev)
        if value == "unspecified":
            return dict(status="unknown", origin=claim_origin, evidence=ev)
        return _assert(value, claim_origin, ev)

    def evs(items):
        return [
            evidence("description" if e.origin == "description" else "skills", e.text).model_copy(
                update={"section": e.section, "interpretation": e.importance}
            )
            for e in items
        ]

    role = RoleFact(
        id="role:primary",
        name=vacancy.title[:1000],
        context="required",
        assertion=_assert(True, "source", [evidence("title", vacancy.title)]),
    )
    facts, reqs, alternatives = [role], [], []
    from nlp.alternatives import OR
    from nlp.text import word_tokens

    lexical = []
    for part in sections(text):
        if part.kind in {"nonprofessional", "conditions", "tasks"}:
            continue
        default = "required" if part.kind == "requirements" else "unspecified"
        for pattern, status in _HEADER_PATTERNS:
            if pattern.match(part.heading):
                default = status
                break
        for line in re.split(r"[;\n]+|(?<=[.!?])\s+", part.text):
            if line.startswith(part.heading + ":"):
                line = line[len(part.heading) + 1 :].strip()
            if not line:
                continue
            importance_value, ambiguous = _importance(line, default)
            _, skill_line = compare_qualifications(line, [], today=vacancy.published_at.date())
            lexical.append(
                LexicalObservation(
                    tokens=list(word_tokens(skill_line)),
                    importance=importance_value,
                    ambiguous=ambiguous or bool(OR.search(line)),
                    evidence=evidence("description", line),
                )
            )
    requirements = reconciled_requirements(vacancy)
    for item in requirements.skills:
        key = _identity("skill", item.skill)
        ev = evs(item.evidence)
        facts.append(SkillFact(id=key, name=item.skill, assertion=_assert(True, origin, ev)))
        if item.counted_separately:
            standalone = [e for e in item.evidence if not e.alternative_id]
            reqs.append(
                Requirement(
                    id="req:" + key,
                    fact_id=key,
                    scope="skill",
                    scope_id=key,
                    importance=importance(
                        item.importance, evs(standalone), [e.importance for e in standalone]
                    ),
                    evidence=evs(standalone),
                )
            )
    for group in requirements.alternatives:
        keys = []
        for name in group.options:
            key = _identity("skill", canonical_skill(name))
            ev = evs(group.evidence)
            if key not in {f.id for f in facts}:
                facts.append(
                    SkillFact(
                        id=key, name=canonical_skill(name), assertion=_assert(True, origin, ev)
                    )
                )
            req_id = "req:or:" + group.id + ":" + key
            reqs.append(
                Requirement(
                    id=req_id,
                    fact_id=key,
                    scope="skill",
                    scope_id=key,
                    importance=importance(
                        group.importance, ev, [e.importance for e in group.evidence]
                    ),
                    evidence=ev,
                )
            )
            keys.append(req_id)
        alternatives.append(Alternative(id="or:" + group.id, requirement_ids=keys))

    qualifications, _ = compare_qualifications(
        text,
        [],
        today=vacancy.published_at.date(),
        explicit=vacancy.skills,
    )
    # Only explicit named levels, not an inferred default or ambiguous adjectives.
    for fact in facts:
        if fact.kind != "skill":
            continue
        pattern = re.compile(
            re.escape(fact.name) + r"\s*[:—-]?\s*(?:уровень\s*[:—-]?\s*)?"
            r"(beginner|intermediate|advanced)\b",
            re.I,
        )
        observations = list(pattern.finditer(text))
        levels = sorted({m[1].lower() for m in observations})
        ev = [evidence("description", m[0]) for m in observations]
        if levels:
            fact.level = type(fact.level)(
                **(
                    _assert(levels[0], origin, ev)
                    if len(levels) == 1
                    else dict(status="conflict", alternatives=levels, origin=origin, evidence=ev)
                )
            )

    document_groups = {}
    for item in qualifications:
        key, ev = _identity("qualification", item.name), evs(item.evidence)
        groups = {e.alternative_id for e in item.evidence if e.alternative_id}
        assertion = (
            _assert(True, origin, ev)
            if item.status != "uncertain" or groups
            else dict(status="unknown", origin=origin, evidence=ev)
        )
        facts.append(QualificationFact(id=key, name=item.name, assertion=assertion))
        for group_id in sorted(
            groups | ({""} if any(not e.alternative_id for e in item.evidence) else set())
        ):
            observations = [e for e in item.evidence if (e.alternative_id or "") == group_id]
            req_id = "req:" + ("or:" + group_id + ":" if group_id else "") + key
            reqs.append(
                Requirement(
                    id=req_id,
                    fact_id=key,
                    importance=importance(
                        item.importance, evs(observations), [e.importance for e in observations]
                    ),
                    evidence=evs(observations),
                )
            )
            if group_id:
                document_groups.setdefault(group_id, []).append(req_id)
    for group_id, keys in document_groups.items():
        if len(keys) >= 2:
            alternatives.append(Alternative(id="or:document:" + group_id, requirement_ids=keys))

    # Parse every relevant line; company age and benefits never become candidate tenure.
    experience = {}
    tasks = set()
    for part in sections(text):
        if part.kind == "tasks":
            for line in re.split(r"(?<=[.!?;])\s+", part.text):
                if line.startswith(part.heading + ":"):
                    line = line[len(part.heading) + 1 :].strip()
                line = line.rstrip(".; ")
                if line.strip(" :") != part.heading:
                    if line:
                        tasks.add(line)
        if part.kind in {"conditions", "nonprofessional", "tasks"}:
            continue
        for line in re.split(r"[;]+|(?<=[.!?])\s+", part.text):
            if _EMPLOYER.search(line) or not re.search(r"опыт|стаж|experience", line, re.I):
                continue
            minimum, relevant = experience_from_quote(line)
            if not relevant:
                continue
            named = extract_skills(line, vocabulary=[f.name for f in facts if f.kind == "skill"])
            targets = [_identity("skill", canonical_skill(n)) for n in named] or [role.id]
            default = "required" if part.kind == "requirements" else "unspecified"
            for pattern, importance_value in _HEADER_PATTERNS:
                if pattern.match(part.heading):
                    default = importance_value
                    break
            status, _ = _importance(line, default)
            for target in targets:
                if target not in {f.id for f in facts}:
                    continue
                entry = experience.setdefault((target, status), {"values": set(), "evidence": []})
                entry["values"].update(experience_bounds(line))
                entry["evidence"].append(evidence("description", line))
    hh = EXPERIENCE_MINIMUM.get(vacancy.experience or "")
    if hh is not None:
        reqs.append(
            Requirement(
                id="req:source:experience",
                fact_id=role.id,
                importance=importance(
                    "required" if hh else "not_required",
                    [evidence("experience")],
                    claim_origin="source",
                ),
                years_min=_assert(hh, "source", [evidence("experience")]),
                evidence=[evidence("experience")],
            )
        )
    if not experience:
        experience[(role.id, "unspecified")] = {"values": set(), "evidence": []}
    for index, ((target, status), entry) in enumerate(experience.items()):
        values, ev = entry["values"], entry["evidence"]
        # HH ranges are coarse. A value inside the source range is compatible.
        if target == role.id and hh is not None and status not in {"preferred", "context"}:
            upper = {"no_experience": 0, "1-3": 3, "3-6": 6}.get(vacancy.experience, 60)
            if values and any(v < hh or v > upper for v in values):
                values.add(hh)
                ev.append(evidence("experience"))
        years = (
            dict(status="conflict", alternatives=sorted(values), origin=origin, evidence=ev)
            if len(values) > 1
            else _assert(next(iter(values)), origin, ev)
            if values
            else dict(status="unknown", origin=origin, evidence=ev)
        )
        scope = "role" if target == role.id else "skill"
        if any(OR.search(e.quote or "") for e in ev):
            grouped_ids = {key for group in alternatives for key in group.requirement_ids}
            matched_or = [
                r
                for r in reqs
                if r.id in grouped_ids
                and r.fact_id == target
                and any(
                    a.quote and b.quote and (a.quote in b.quote or b.quote in a.quote)
                    for a in r.evidence
                    for b in ev
                )
            ]
            if matched_or:
                for requirement in matched_or:
                    requirement.years_min = type(requirement.years_min)(**years)
                continue
            # Unsupported alternative (e.g. employment OR academic project) is not AND.
            status = "unspecified"
            requirements.warnings.append("Не определена область альтернативы требования опыта")
        reqs.append(
            Requirement(
                id=f"req:experience:{index}",
                fact_id=target,
                scope=scope,
                scope_id=target,
                importance=importance(
                    "unspecified" if years.get("status") == "conflict" else status, ev
                ),
                years_min=years,
                evidence=ev,
            )
        )

    # A positive scoped duration already requires presence; don't count the
    # same standalone condition twice. Independent OR/AND clauses retain IDs.
    duration_subjects = {
        (r.fact_id, r.importance.value)
        for r in reqs
        if r.id.startswith("req:experience:") and r.years_min.value is not None
    }
    reqs = [
        r
        for r in reqs
        if not (
            r.id.startswith("req:skill:") and (r.fact_id, r.importance.value) in duration_subjects
        )
    ]

    if hh == 0 and any(
        r.scope == "role" and r.importance.value == "required" and r.years_min.status == "unknown"
        for r in reqs
    ):
        for req in reqs:
            if req.id == "req:source:experience":
                req.years_min = type(req.years_min)(status="unknown", evidence=req.evidence)
                req.importance = type(req.importance)(status="unknown")

    role_conflicts = [r for r in reqs if r.scope == "role" and r.years_min.status == "conflict"]
    if role_conflicts:
        for req in reqs:
            if req.id == "req:source:experience":
                req.years_min = role_conflicts[0].years_min.model_copy(deep=True)
                req.importance = type(req.importance)(status="unknown")

    if analysis:
        for extraction in analysis.chunks or [analysis.extraction]:
            tasks.update(
                source_quote(text, q)
                for q in extraction.responsibilities
                if professional_quote(text, q)
            )
            if extraction.specialization and (
                extraction.role_evidence in vacancy.title
                or professional_quote(text, extraction.role_evidence)
            ):
                key = _identity("role", extraction.specialization)
                if key not in {f.id for f in facts}:
                    field = "title" if extraction.role_evidence in vacancy.title else "description"
                    facts.append(
                        RoleFact(
                            id=key,
                            name=extraction.specialization,
                            context="required",
                            assertion=_assert(
                                True, origin, [evidence(field, extraction.role_evidence)]
                            ),
                        )
                    )
    for task in sorted(tasks):
        # Fact names are bounded, evidence retains the complete original statement.
        facts.append(
            TaskFact(
                id=_identity("task", task),
                name=task[:1000],
                assertion=_assert(True, origin, [evidence("description", task)]),
            )
        )
    for field in (
        "remote",
        "location",
        "salary_from",
        "salary_to",
        "currency",
        "salary_gross",
        "salary_period",
        "employment",
        "schedule",
        "work_format",
        "working_hours",
        "experience",
        "archived",
        "published_at",
        "valid_until",
    ):
        value = vacancy.model_dump(mode="json")[field]
        facts.append(
            ValueFact(
                kind="condition",
                id="condition:" + field,
                name=field,
                assertion=_assert(value, "source", [evidence(field)])
                if value is not None
                else dict(status="missing"),
            )
        )
    for part in sections(text):
        if part.kind == "conditions":
            facts.append(
                ValueFact(
                    kind="condition",
                    id=_identity("condition:text", part.text),
                    name=part.text[:1000],
                    assertion=_assert(part.text, "source", [evidence("description", part.text)]),
                )
            )
    # Repeated source lines must not create duplicate IDs.
    facts = list({f.id: f for f in facts}.values())
    result = ProfessionalFacts(
        document_id=source_id,
        subject="vacancy",
        source_revision=revision,
        input_sha256=revision,
        extractor="vacancy-runtime",
        extractor_version="vacancy-facts-v2",
        extractor_contract=facts_contract(),
        model_id=analysis.model if analysis else None,
        model_digest=analysis.model_digest if analysis else None,
        analysis_pipeline=analysis.pipeline_hash if analysis else None,
        extracted_at=datetime.now(UTC),
        reference_time=vacancy.published_at,
        input_scope="legacy"
        if legacy
        else "partial"
        if analysis and analysis.processing_status == "partial"
        else "full",
        input_ranges=[(0, len(text))] if text else [],
        uncertainty_reasons=(
            requirements.warnings + (analysis.failure_codes if analysis else ["rules_only"])
        ),
        sources=[SourceReference(id=source_id, revision=revision, kind="vacancy")],
        facts=facts,
        requirements=reqs,
        lexical_observations=lexical,
        alternatives=alternatives,
    )
    result.verify_text(source_id, revision, {"title": vacancy.title, "description": text})
    return result
