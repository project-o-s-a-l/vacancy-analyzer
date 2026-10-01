"""Small, explainable search expansion, independent of delivery and matching."""

from dataclasses import dataclass

from collectors.base import SearchQuery
from domain import CandidateProfile

PLAN_VERSION = "hh-plan-v1"


@dataclass(frozen=True)
class PlannedQuery:
    original: str
    text: str
    reason: str
    origin: str


def plan_profile(profile: CandidateProfile) -> list[PlannedQuery]:
    if not profile.skills:
        return []
    origin = "position" if profile.preferences.positions else "skill"
    plan = []
    for value in profile.preferences.positions or profile.skills:
        title = " ".join(value.split())
        if not title:
            continue
        plan.append(PlannedQuery(title, title, "original", origin))
        # Retain both the occupation and its subject. Never expand Unity teachers
        # to Unity developers, or arbitrary skills to an unrelated occupation.
        for prefix in ("преподаватель разработки на ", "преподаватель программирования на "):
            if title.casefold().startswith(prefix) and title[len(prefix) :].strip():
                plan.append(
                    PlannedQuery(
                        title,
                        "Преподаватель " + title[len(prefix) :],
                        "short_teacher_title",
                        origin,
                    )
                )
                break
    return plan


def plan_queries(plan: list[PlannedQuery], template: SearchQuery) -> list[SearchQuery]:
    return [template.model_copy(update={"text": item.text, "search_field": "all"}) for item in plan]
