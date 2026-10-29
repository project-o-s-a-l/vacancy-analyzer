"""Adapt validated LLM extraction to the existing matching contract."""

import hashlib
import json

from domain.requirements import (
    AlternativeRequirement,
    RequirementAnalysis,
    RequirementEvidence,
    SkillRequirement,
)
from domain.vacancy_analysis import VacancyAnalysis
from nlp.extraction import canonical_skill, skill_key
from nlp.qualifications import is_qualification


def analyzed_requirements(analysis: VacancyAnalysis) -> RequirementAnalysis:
    skills: dict[tuple[str, ...], SkillRequirement] = {}
    for item in analysis.extraction.skills:
        if is_qualification(item.name):
            continue
        key = skill_key(item.name)
        evidence = RequirementEvidence(
            origin="description", importance=item.importance, text=item.evidence
        )
        if key not in skills:
            skills[key] = SkillRequirement(
                skill=canonical_skill(item.name), importance=item.importance, evidence=[evidence]
            )
        else:
            previous = skills[key]
            previous.evidence.append(evidence)
            if previous.importance != item.importance:
                previous.importance = "unspecified"
    alternatives = []
    for item in analysis.extraction.alternatives:
        options = list(dict.fromkeys(canonical_skill(name) for name in item.options))
        if len(options) < 2 or any(is_qualification(name) for name in options):
            continue
        identity = json.dumps(sorted(skill_key(name) for name in options))
        alternatives.append(
            AlternativeRequirement(
                id=hashlib.sha256(identity.encode()).hexdigest()[:16],
                options=options,
                importance=item.importance,
                evidence=[
                    RequirementEvidence(
                        origin="description", importance=item.importance, text=item.evidence
                    )
                ],
            )
        )
        for name in options:
            if skill_key(name) in skills:
                skills[skill_key(name)].counted_separately = False
    return RequirementAnalysis(skills=list(skills.values()), alternatives=alternatives)


def fact_requirements(facts) -> RequirementAnalysis:
    """Read shared extraction without parsing text or invoking a model per candidate."""
    names = {f.id: f.name for f in facts.facts if f.kind == "skill"}
    requirements = {r.id: r for r in facts.requirements}
    grouped = {key for group in facts.alternatives for key in group.requirement_ids}

    def status(req):
        return req.importance.value if req.importance.status == "present" else "unspecified"

    def evidence(req, group=None):
        return [
            RequirementEvidence(
                origin="description" if e.field == "description" else "tag",
                importance=e.interpretation or status(req),
                text=e.quote or names.get(req.fact_id, ""),
                section=e.section,
                alternative_id=group,
            )
            for e in req.evidence
        ]

    skills = []
    for key, name in names.items():
        items = [r for r in facts.requirements if r.fact_id == key]
        standalone = [r for r in items if r.id not in grouped]
        req = next(iter(standalone or items), None)
        skills.append(
            SkillRequirement(
                skill=name,
                importance=status(req) if req else "unspecified",
                counted_separately=bool(standalone),
                evidence=evidence(req) if req else [],
            )
        )
    alternatives = []
    for group in facts.alternatives:
        items = [requirements[key] for key in group.requirement_ids]
        if any(item.fact_id not in names for item in items):
            continue
        alternatives.append(
            AlternativeRequirement(
                id=group.id.removeprefix("or:"),
                options=[names[r.fact_id] for r in items],
                importance=status(items[0]),
                evidence=evidence(items[0], group.id.removeprefix("or:")),
            )
        )
    return RequirementAnalysis(
        skills=skills,
        alternatives=alternatives,
        warnings=["Для одного навыка найдены противоречивые требования"]
        if any(r.importance.status == "conflict" for r in facts.requirements)
        else [],
    )


def phrase_requirements(facts, vocabulary=()):
    """Bounded lookup in the public phrase index, never parse source per pair."""
    from nlp.text import contains_phrase

    known = {skill_key(f.name) for f in facts.facts if f.kind == "skill"}
    output = []
    extra = sorted(
        {name for name in vocabulary if skill_key(name) not in known and not is_qualification(name)}
    )[:128]
    for name in extra:
        key = skill_key(name)
        observations = [
            o for o in facts.lexical_observations if contains_phrase(tuple(o.tokens), key)
        ]
        if not observations:
            continue
        statuses = {o.importance for o in observations} - {"context", "unspecified"}
        if statuses and statuses <= {"not_required"}:
            continue
        importance = next(iter(statuses)) if len(statuses) == 1 else "unspecified"
        if any(o.ambiguous for o in observations):
            importance = "unspecified"
        output.append(
            SkillRequirement(
                skill=canonical_skill(name),
                importance=importance,
                evidence=[
                    RequirementEvidence(
                        origin="description", importance=o.importance, text=o.evidence.quote or name
                    )
                    for o in observations
                ],
            )
        )
    return output


def matching_requirements(vacancy, vocabulary=()):
    from nlp.professional_facts import vacancy_facts, vacancy_facts_compatible

    facts = vacancy.professional_facts
    if facts is None or not vacancy_facts_compatible(facts, vacancy):
        facts = vacancy_facts(vacancy)
        vacancy.professional_facts = facts
    result = fact_requirements(facts)
    result.skills.extend(phrase_requirements(facts, vocabulary))
    return result
