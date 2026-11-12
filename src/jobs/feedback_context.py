"""Frozen delivery metadata. Free text stays private; exported slices use bounded codes."""

from datetime import UTC, datetime

from domain.vacancy_duplicates import vacancy_duplicate_key
from scoring.admission_policy import policy_contract
from scoring.serving_features import digest, retrieval_contract

# Bounded diagnostic vocabulary, not a role classifier or training taxonomy.
ROLE_SLICES = {
    "python": "python developer",
    "java": "java developer",
    "designer": "дизайнер",
    "accountant": "бухгалтер",
    "driver": "водитель",
    "sales": "менеджер по продажам",
    "nurse": "медицинская сестра",
    "electrician": "электромонтёр",
}
SKILL_SLICES = {
    "python",
    "sql",
    "docker",
    "java",
    "spring",
    "git",
    "figma",
    "photoshop",
    "typography",
    "1с",
    "excel",
    "ндс",
    "навигация",
    "путевые листы",
    "логистика",
    "crm",
    "переговоры",
    "инъекции",
    "стерилизация",
    "медицинская документация",
    "электробезопасность",
    "схемы",
    "диагностика",
}


def slices(profile):
    # Allowlist prevents contacts disguised as skill/role names from entering exports.
    skills = {s.casefold() for s in profile.skills}
    positions = " ".join(profile.preferences.positions).casefold()
    roles = [code for code, title in ROLE_SLICES.items() if title in positions]
    known = SKILL_SLICES
    facts = profile.professional_facts
    return {
        "roles": roles or ["other_or_unknown"],
        "skills": sorted(skills & known),
        "other_skill_count": len(skills - known),
        "facts_present": facts is not None,
        "tasks_present": bool(facts and any(f.kind == "task" for f in facts.facts)),
    }


def delivery_context(vacancy, profile, match, *, min_score, top, surface="recommendation"):
    retrieval = match.retrieval.model_dump(mode="json") if match.retrieval else None
    return {
        "version": "feedback-evidence-v1",
        "surface": surface,
        "captured_at": datetime.now(UTC).isoformat(),
        "vacancy_group": digest(vacancy_duplicate_key(vacancy) or str(vacancy.id)),
        "retrieval_contract": retrieval_contract(),
        "retrieval": retrieval,
        "policy_contract": policy_contract(),
        "model_id": match.ranking.model_id if match.ranking and match.ranking.applied else "rules",
        "admission_policy": match.admission,
        "min_score": min_score,
        "top": top,
        "propensity": None,
        "slices": slices(profile),
        "published_at": vacancy.published_at.isoformat(),
        "last_checked_at": vacancy.last_checked_at.isoformat() if vacancy.last_checked_at else None,
    }
