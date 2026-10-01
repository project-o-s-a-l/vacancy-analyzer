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
