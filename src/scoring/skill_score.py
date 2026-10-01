from collections.abc import Iterable
from dataclasses import dataclass

from domain.requirements import AlternativeMatch, AlternativeRequirement
from nlp.extraction import canonical_skill, skill_key


@dataclass(frozen=True)
class SkillMatch:
    score: float | None
    matched: tuple[str, ...]
    missing: tuple[str, ...]
    alternatives: tuple[AlternativeMatch, ...] = ()


def match_skills(
    candidate: Iterable[str],
    required: Iterable[str],
    optional: Iterable[str] = (),
    *,
    unspecified: Iterable[str] = (),
    alternatives: Iterable[AlternativeRequirement] = (),
) -> SkillMatch:
    known = {skill_key(value) for value in candidate if value.strip()}
    required_names = {
        skill_key(value): canonical_skill(value) for value in required if value.strip()
    }
    optional_names = {
        skill_key(value): canonical_skill(value) for value in optional if value.strip()
    }
    unspecified_names = {
        skill_key(value): canonical_skill(value) for value in unspecified if value.strip()
    }
    required_set = required_names.keys()
    unspecified_set = unspecified_names.keys() - required_set
    optional_set = optional_names.keys() - required_set - unspecified_set
    weights = {
        **dict.fromkeys(optional_set, 1),
        **dict.fromkeys(unspecified_set, 2),
        **dict.fromkeys(required_set, 3),
    }
    total = sum(weights.values())
    names = {**optional_names, **unspecified_names, **required_names}
    matched_keys = known & weights.keys()
    matched_weight = sum(weights[key] for key in matched_keys)
    groups = {}
    priority = {"required": 3, "unspecified": 2, "preferred": 1, "not_required": 0, "context": 0}
    for group in alternatives:
        identity = frozenset(skill_key(value) for value in group.options)
        if (
            identity not in groups
            or priority[group.importance] > priority[groups[identity].importance]
        ):
            groups[identity] = group
    evaluated = []
    group_matched = set()
    for identity, group in groups.items():
        value = priority[group.importance]
        # A stronger standalone condition implies this OR; do not weight it twice.
        counted = value > 0 and not any(weights.get(key, 0) >= value for key in identity)
        matched_options = sorted(
            {canonical_skill(v) for v in group.options if skill_key(v) in known}
        )
        satisfied = bool(matched_options)
        evaluated.append(
            AlternativeMatch(
                **group.model_dump(),
                satisfied=satisfied,
                matched_options=matched_options,
                counted_in_score=counted,
            )
        )
        if counted:
            total += value
            if satisfied:
                matched_weight += value
                group_matched.update(matched_options)
    matched = sorted({names[key] for key in matched_keys} | group_matched)
    missing = sorted(names[key] for key in weights.keys() - known)
    score = matched_weight / total if total else None
    return SkillMatch(score, tuple(matched), tuple(missing), tuple(evaluated))
