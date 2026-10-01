"""One admissibility policy for every caller, regardless of ranking weights."""

import re
from datetime import UTC, datetime

from domain import CandidateProfile, Vacancy
from domain.eligibility import EligibilityEvidence, EligibilityIssue, EligibilityResult
from domain.qualifications import QualificationMatch
from nlp.experience_requirement import experience_requirement
from nlp.qualifications import compare_qualifications
from nlp.text import normalize_text

EXPERIENCE_TOLERANCE_YEARS = 1.0
_SENIOR = re.compile(
    r"\b(?:senior(?!\s+care\b)|lead(?![-\s]+gen(?:eration|erator)?\b)|principal|"
    r"сеньор\w*|сениор\w*|ведущ\w*(?!\s+(?:мероприяти\w*|праздник\w*|стрим\w*))|"
    r"старш\w*|руководител\w*)\b",
    re.I,
)


def salary_comparable(vacancy: Vacancy, profile: CandidateProfile) -> bool:
    prefs = profile.preferences
    currency = (vacancy.currency or "").upper()
    if currency == "RUR":
        currency = "RUB"
    return (
        currency == prefs.currency
        and vacancy.salary_gross == prefs.salary_gross
        and vacancy.salary_period == "MONTH"
    )


def location_matches(vacancy: Vacancy, profile: CandidateProfile) -> bool | None:
    if vacancy.remote is True:
        return True
    if not vacancy.location:
        return None

    def key(value):
        return re.sub(r"^(?:г\.?\s+|город\s+)", "", " ".join(normalize_text(value).split()))

    return key(vacancy.location) in {key(value) for value in profile.preferences.locations}


def check_eligibility(
    vacancy: Vacancy,
    profile: CandidateProfile,
    *,
    now: datetime | None = None,
    qualifications: list[QualificationMatch] | None = None,
) -> EligibilityResult:
    now = now or datetime.now(UTC)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must contain a timezone")
    result = EligibilityResult(experience=experience_requirement(vacancy))

    def issue(code, field, message, *, exclude=False, evidence=()):
        if not evidence and field in {"remote", "location", "salary", "publication"}:
            values = {
                "remote": str(vacancy.remote),
                "location": vacancy.location or "не указано",
                "salary": f"{vacancy.salary_from}–{vacancy.salary_to} {vacancy.currency}, "
                f"gross={vacancy.salary_gross}, {vacancy.salary_period}",
                "publication": f"published={vacancy.published_at.isoformat()}, "
                f"archived={vacancy.archived}, valid_until={vacancy.valid_until}",
            }
            evidence = [EligibilityEvidence(source="hh", text=values[field])]
        item = EligibilityIssue(code=code, field=field, message=message, evidence=list(evidence))
        (result.exclusions if exclude else result.uncertainties).append(item)

    if vacancy.archived:
        issue("archived", "publication", "Вакансия в архиве", exclude=True)
    if vacancy.valid_until and vacancy.valid_until <= now:
        issue("expired", "publication", "Срок вакансии истёк", exclude=True)
    if vacancy.published_at > now:
        issue(
            "future_publication", "publication", "Дата публикации находится в будущем", exclude=True
        )
    required = result.experience
    years = profile.experience_years
    if required.conflict:
        issue(
            "experience_conflict",
            "experience",
            "Сведения о требовании опыта расходятся; проверьте описание",
            evidence=required.evidence,
        )
    if years is None:
        issue("candidate_experience_unknown", "experience", "Общий стаж кандидата не указан")
    elif years == 0 and required.required is True:
        issue(
            "experience_required",
            "experience",
            "В вакансии явно требуется опыт; у вас указано 0 лет",
            exclude=True,
            evidence=required.evidence,
        )
    elif years > 0 and required.minimum_years is not None and years < required.minimum_years:
        gap = required.minimum_years - years
        issue(
            "experience_gap",
            "experience",
            f"Требуется опыт от {required.minimum_years:g} лет; у вас {years:g}",
            exclude=gap > EXPERIENCE_TOLERANCE_YEARS,
            evidence=required.evidence,
        )
    if required.minimum_years is None:
        issue(
            "experience_minimum_unknown" if required.required else "job_experience_unknown",
            "experience",
            "Опыт требуется, но число лет не определено"
            if required.required
            else "Требование к общему стажу не определено",
            evidence=required.evidence,
        )
    if years == 0 and _SENIOR.search(vacancy.title):
        issue(
            "seniority_mismatch",
            "seniority",
            "Старшая или руководящая позиция не подходит для нулевого стажа",
            exclude=True,
            evidence=[EligibilityEvidence(source="title", text=vacancy.title)],
        )

    prefs = profile.preferences
    if prefs.remote in {"required", "office"}:
        expected = prefs.remote == "required"
        if vacancy.remote is None or vacancy.remote != expected:
            issue(
                "work_format_required",
                "remote",
                "Обязательный формат работы не подтверждён",
                exclude=True,
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
            )
    if prefs.salary_min is not None and prefs.salary_min > 0:
        comparable = salary_comparable(vacancy, profile)
        if not comparable:
            issue(
                "salary_incomparable",
                "salary",
                "Зарплата неизвестна или несопоставима по валюте, периоду и налогам",
                exclude=prefs.salary_required,
            )
        elif vacancy.salary_to is not None and vacancy.salary_to < prefs.salary_min:
            issue(
                "salary_below_minimum",
                "salary",
                "Верхняя граница зарплаты ниже вашего минимума",
                exclude=prefs.salary_required,
            )
        elif vacancy.salary_from is None or vacancy.salary_from < prefs.salary_min:
            unknown = vacancy.salary_to is None
            issue(
                "salary_not_guaranteed",
                "salary",
                "Зарплатный минимум не гарантирован нижней границей",
                exclude=prefs.salary_required and unknown,
            )
    if qualifications is None:
        qualifications, _ = compare_qualifications(
            vacancy.description or vacancy.description_html,
            profile.qualifications,
            today=now.astimezone(UTC).date(),
            explicit=vacancy.skills,
        )
    for document in qualifications:
        if document.importance in {"context", "not_required"}:
            continue
        if document.status != "declared_present":
            issue(
                "qualification_" + document.status,
                "qualification",
                "Проверьте документ: " + document.name,
                exclude=document.importance == "required"
                and document.status in {"declared_absent", "expired"},
                evidence=[
                    EligibilityEvidence(
                        source="description" if e.origin == "description" else "hh", text=e.text
                    )
                    for e in document.evidence
                ],
            )
    return result
