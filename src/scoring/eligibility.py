"""One admissibility policy for every caller, regardless of ranking weights."""

import re
from datetime import UTC, datetime

from domain import CandidateProfile, Vacancy
from domain.eligibility import EligibilityEvidence, EligibilityIssue, EligibilityResult
from domain.exchange_rates import current_rates
from nlp.text import normalize_text
from scoring.comparison import compare_facts, ready_facts, total_experience

_SENIOR = re.compile(
    r"\b(?:senior(?!\s+care\b)|lead(?![-\s]+gen(?:eration|erator)?\b)|principal|"
    r"сеньор\w*|сениор\w*|ведущ\w*(?!\s+(?:мероприяти\w*|праздник\w*|стрим\w*))|"
    r"старш\w*|руководител\w*)\b",
    re.I,
)


def condition(vacancy, field):
    facts = vacancy.professional_facts or ready_facts(vacancy)
    return next((f.assertion for f in facts.facts if f.id == "condition:" + field), None)


def condition_value(vacancy, field):
    claim = condition(vacancy, field)
    return claim.value if claim is not None else getattr(vacancy, field)


def salary_bounds(vacancy: Vacancy, profile: CandidateProfile):
    """Comparable monthly bounds in the candidate's currency, before display rounding."""
    prefs = profile.preferences
    if (
        condition_value(vacancy, "salary_gross") != prefs.salary_gross
        or condition_value(vacancy, "salary_period") != "MONTH"
    ):
        return None, None
    currency = condition_value(vacancy, "currency") or ""
    rates = current_rates()
    return (
        rates.convert(condition_value(vacancy, "salary_from"), currency, prefs.currency),
        rates.convert(condition_value(vacancy, "salary_to"), currency, prefs.currency),
    )


def location_matches(vacancy: Vacancy, profile: CandidateProfile) -> bool | None:
    if condition_value(vacancy, "remote") is True:
        return True
    keys = vacancy_location_keys(vacancy)
    if not keys:
        return None
    return bool(keys.intersection(location_key(value) for value in profile.preferences.locations))


def vacancy_location_keys(vacancy):
    locations = vacancy.locations
    if not locations and vacancy.location:
        locations = (
            vacancy.location.split(",") if vacancy.source == "habr_career" else [vacancy.location]
        )
    return {location_key(value) for value in locations or () if value.strip()}


def location_key(value):
    return re.sub(r"^(?:г\.?\s+|город\s+)", "", " ".join(normalize_text(value).split()))


def check_eligibility(
    vacancy: Vacancy,
    profile: CandidateProfile,
    *,
    now: datetime | None = None,
    comparison=None,
) -> EligibilityResult:
    now = now or datetime.now(UTC)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must contain a timezone")
    comparison = comparison or compare_facts(vacancy, profile, now=now)
    result = EligibilityResult()

    def issue(
        code, field, message, *, exclude=False, evidence=(), state="unknown", mandatory=False
    ):
        if not evidence and field in {"remote", "location", "salary", "publication"}:
            values = {
                "remote": str(vacancy.remote),
                "location": vacancy.location or "не указано",
                "salary": f"{vacancy.salary_from}–{vacancy.salary_to} {vacancy.currency}, "
                f"gross={vacancy.salary_gross}, {vacancy.salary_period}",
                "publication": f"published={vacancy.published_at.isoformat()}, "
                f"archived={vacancy.archived}, valid_until={vacancy.valid_until}",
            }
            evidence = [EligibilityEvidence(source=vacancy.source, text=values[field])]
        fields = (
            ("salary_from", "salary_to", "currency", "salary_gross", "salary_period")
            if field == "salary"
            else (field,)
        )
        claims = [condition(vacancy, name) for name in fields]
        if any(claim is not None and claim.status == "conflict" for claim in claims):
            state = "conflict"
        item = EligibilityIssue(
            code=code,
            field=field,
            message=message,
            evidence=list(evidence),
            state=state,
            mandatory=mandatory or exclude,
        )
        (result.exclusions if exclude else result.uncertainties).append(item)

    if vacancy.archived:
        issue("archived", "publication", "Вакансия в архиве", exclude=True)
    if vacancy.valid_until and vacancy.valid_until <= now:
        issue("expired", "publication", "Срок вакансии истёк", exclude=True)
    if vacancy.published_at > now:
        issue(
            "future_publication", "publication", "Дата публикации находится в будущем", exclude=True
        )
    years = total_experience(profile).value
    experience_items = [
        i
        for i in comparison.items
        if i.years_required is not None or i.id.startswith("req:experience:")
    ]
    result.experience.minimum_years = max(
        (i.years_required for i in experience_items if i.years_required is not None), default=None
    )
    result.experience.conflict = any(i.state == "conflict" for i in experience_items)
    result.experience.required = (
        True if any(i.importance == "required" for i in experience_items) else None
    )
    result.experience.evidence = [
        EligibilityEvidence(
            source="description" if e.field == "description" else vacancy.source,
            text=e.quote or i.name,
        )
        for i in experience_items
        for e in i.evidence
    ]
    for item in comparison.items:
        if item.state == "match":
            continue
        field = (
            "qualification"
            if item.kind == "qualification"
            else (
                "experience"
                if item.years_required is not None or item.id.startswith("req:experience:")
                else "skill"
                if item.kind == "skill"
                else "experience"
            )
        )
        state_label = {
            "unknown": "не хватает сведений",
            "conflict": "конфликт источников",
            "contradiction": "подтверждено несоответствие",
        }[item.state]
        issue(
            field + "_" + item.state,
            field,
            f"{item.name}: {state_label}. {item.detail}",
            exclude=item.importance == "required" and item.state == "contradiction",
            state=item.state,
            mandatory=item.importance == "required",
            evidence=[
                EligibilityEvidence(
                    source="description" if e.field == "description" else vacancy.source,
                    text=e.quote or item.name,
                )
                for e in item.evidence
            ],
        )
    if years is None:
        issue("candidate_experience_unknown", "experience", "Общий стаж кандидата не указан")
    if years == 0 and _SENIOR.search(vacancy.title):
        issue(
            "seniority_mismatch",
            "seniority",
            "Старшая позиция при явно указанном нулевом стаже",
            exclude=True,
            evidence=[EligibilityEvidence(source="title", text=vacancy.title)],
        )

    prefs = profile.preferences
    if prefs.remote in {"required", "office"}:
        expected = prefs.remote == "required"
        remote = condition_value(vacancy, "remote")
        if remote is None or remote != expected:
            issue(
                "work_format_required",
                "remote",
                "Обязательный формат работы не подтверждён",
                exclude=True,
                state="unknown" if remote is None else "contradiction",
            )
    elif prefs.remote == "preferred" and vacancy.remote is not True:
        issue("remote_preferred", "remote", "Предпочтение удалённой работы не подтверждено")
    if prefs.locations:
        match = location_matches(vacancy, profile)
        if match is not True:
            issue(
                "location_mismatch" if match is False else "location_unknown",
                "location",
                "Город не подходит" if match is False else "Место работы не указано",
                exclude=prefs.locations_required,
                state="unknown" if match is None else "contradiction",
            )
    if prefs.salary_min is not None and prefs.salary_min > 0:
        lower, upper = salary_bounds(vacancy, profile)
        if lower is None and upper is None:
            issue(
                "salary_incomparable",
                "salary",
                "Зарплата неизвестна или несопоставима по валюте, периоду и налогам",
                exclude=prefs.salary_required,
            )
        elif upper is not None and upper < prefs.salary_min:
            issue(
                "salary_below_minimum",
                "salary",
                "Верхняя граница зарплаты ниже вашего минимума",
                exclude=prefs.salary_required,
                state="contradiction",
            )
        elif lower is None or lower < prefs.salary_min:
            issue(
                "salary_not_guaranteed",
                "salary",
                "Зарплатный минимум не гарантирован нижней границей",
                exclude=prefs.salary_required and upper is None,
            )
    return result
