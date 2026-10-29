"""Transparent heuristic scores. Missing evidence gets no invented positive match."""

from datetime import UTC, datetime
from time import perf_counter

from domain import CandidateProfile, Vacancy
from domain.matching import MatchResult
from nlp.analysis_identity import current_vacancy_analysis as current_analysis
from nlp.extraction import skill_key
from nlp.qualifications import QUALIFICATIONS_VERSION
from nlp.requirements import RULES_VERSION
from nlp.titles import title_tokens
from nlp.vacancy_requirements import matching_requirements
from scoring.comparison import candidate_skill_names, compare_facts, qualification_matches
from scoring.eligibility import (
    check_eligibility,
    condition_value,
    location_matches,
    salary_comparable,
)
from scoring.skill_score import match_skills

WEIGHTS = {
    "skills": 0.40,
    "experience": 0.20,
    "title": 0.15,
    "salary": 0.10,
    "preferences": 0.10,
    "freshness": 0.05,
}


def score_vacancy(
    vacancy: Vacancy, profile: CandidateProfile, *, now: datetime | None = None
) -> MatchResult:
    now = now or datetime.now(UTC)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must contain a timezone")
    warnings: list[str] = []
    reasons: list[str] = []
    comparison = compare_facts(vacancy, profile, now=now)
    qualifications = qualification_matches(comparison)
    eligibility = check_eligibility(vacancy, profile, now=now, comparison=comparison)
    warnings.extend(item.message for item in [*eligibility.exclusions, *eligibility.uncertainties])
    analysis = current_analysis(vacancy)
    use_model_requirements = bool(
        analysis and analysis.input_scope == "full" and analysis.requirements_source == "llm"
    )
    requirements = matching_requirements(vacancy, candidate_skill_names(profile))
    skills = match_skills(
        candidate_skill_names(profile),
        requirements.names("required"),
        requirements.names("preferred"),
        unspecified=requirements.names("unspecified"),
        alternatives=requirements.alternatives,
    )
    for document in qualifications:
        if document.importance in {"not_required", "context"}:
            continue
        if document.status == "declared_present":
            reasons.append("Документ указан вами, без проверки: " + document.name)
        else:
            warnings.append("Уточните документ или допуск: " + document.name)
    if requirements.skills and not vacancy.skills and not use_model_requirements:
        warnings.append(
            "Навыки найдены по упоминаниям в тексте; полный список требований не подтверждён"
        )
    if skills.matched:
        reasons.append("Совпадают навыки: " + ", ".join(skills.matched))
    missing_keys = {skill_key(value) for value in skills.missing}
    missing_by_importance = {
        status: [value for value in requirements.names(status) if skill_key(value) in missing_keys]
        for status in ("required", "preferred", "unspecified")
    }
    for status, label in (
        ("required", "Обязательные навыки не найдены в профиле"),
        ("preferred", "Желательные навыки не найдены в профиле"),
        ("unspecified", "Навыки не найдены в профиле, важность не указана"),
    ):
        if missing_by_importance[status]:
            warnings.append(label + ": " + ", ".join(missing_by_importance[status]))
    warnings.extend(requirements.warnings)
    for group in skills.alternatives:
        compared = next((item for item in comparison.items if item.id == "or:" + group.id), None)
        if compared is not None:
            group.satisfied = compared.state == "match"
        if not group.counted_in_score:
            continue
        label = {
            "required": "обязательно",
            "preferred": "желательно",
            "unspecified": "важность не указана",
        }[group.importance]
        options = " / ".join(group.options)
        if group.satisfied:
            reasons.append(
                f"Выполнено условие «один из» ({label}): " + ", ".join(group.matched_options)
            )
        else:
            warnings.append(f"Условие «один из» не подтверждено ({label}): {options}")
    if skills.score is None:
        warnings.append("Навыки вакансии не определены")

    experience = comparison.features["experience_fit"]
    for item in comparison.items:
        if item.state == "match" and item.years_required is not None:
            reasons.append(f"{item.name}: {item.detail}; требование от {item.years_required:g} лет")
    # Required/preferred weights are the existing 3/1/2 policy; scopes and OR
    # now use the same evidence states as admission. No learned contribution is claimed.
    skill_items = [
        i
        for i in comparison.items
        if i.kind == "skill"
        and not (i.id.startswith("req:experience:") and i.years_required is None)
    ]
    weights = {"required": 3, "preferred": 1, "unspecified": 2}
    denominator = sum(weights.get(i.importance, 0) for i in skill_items)
    skill_score = (
        (sum(weights.get(i.importance, 0) * (i.ratio or 0) for i in skill_items) / denominator)
        if denominator
        else None
    )

    preferences = profile.preferences
    title = None
    if preferences.positions:
        vacancy_tokens = title_tokens(vacancy.title)
        title = (
            max(
                len(vacancy_tokens & tokens) / len(tokens) if tokens else 0.0
                for position in preferences.positions
                if (tokens := title_tokens(position))
            )
            if any(title_tokens(position) for position in preferences.positions)
            else 0.0
        )

    context = [
        comparison.features[key]
        for key in ("role_match", "task_match")
        if comparison.features[key] is not None
    ]
    if context:
        title = sum([*context, *([title] if title is not None else [])]) / (
            len(context) + (title is not None)
        )
        reasons.append("Сопоставлены роли и задачи по нормализованным фактам")

    salary = None
    if preferences.salary_min is not None:
        comparable = salary_comparable(vacancy, profile)
        upper = condition_value(vacancy, "salary_to")
        if upper is None:
            upper = condition_value(vacancy, "salary_from")
        if comparable and upper is not None:
            salary = min(1.0, upper / preferences.salary_min) if preferences.salary_min else 1.0
            if vacancy.salary_from is None or vacancy.salary_from < preferences.salary_min:
                warnings.append("Нижняя граница зарплаты не подтверждает желаемый минимум")
        else:
            warnings.append("Зарплата неизвестна или несопоставима по валюте/периоду/gross")

    preference_parts: list[float] = []
    remote = condition_value(vacancy, "remote")
    if preferences.remote != "any":
        if remote is None:
            warnings.append("Формат работы не указан")
        else:
            remote_match = remote if preferences.remote != "office" else not remote
            preference_parts.append(float(remote_match))
    if preferences.locations and remote is not True:
        location_match = location_matches(vacancy, profile)
        if location_match is not None:
            preference_parts.append(float(location_match))
        else:
            warnings.append("Местоположение не указано")

    age = max(0.0, (now - vacancy.published_at).total_seconds() / 86400)
    components = {
        "skills": skill_score,
        "experience": experience,
        "title": title,
        "salary": salary,
        "preferences": sum(preference_parts) / len(preference_parts) if preference_parts else None,
        "freshness": max(0.0, 1 - age / 30),
    }
    coverage = sum(WEIGHTS[key] for key, value in components.items() if value is not None)
    # Fixed denominator: incomplete evidence must not turn a fresh empty vacancy into 100%.
    total = sum(WEIGHTS[key] * value for key, value in components.items() if value is not None)
    return MatchResult(
        vacancy_id=str(vacancy.id),
        score=round(100 * total, 2),
        components=components,
        evidence_coverage=round(coverage, 6),
        matched_skills=list(skills.matched),
        missing_skills=list(skills.missing),
        reasons=reasons,
        warnings=list(dict.fromkeys(warnings)),
        eligible=eligibility.eligible,
        eligibility=eligibility,
        comparison=comparison,
        requirements=requirements.skills,
        missing_required=missing_by_importance["required"],
        missing_preferred=missing_by_importance["preferred"],
        missing_unspecified=missing_by_importance["unspecified"],
        requirements_version=analysis.version if use_model_requirements else RULES_VERSION,
        alternatives=list(skills.alternatives),
        qualifications=qualifications,
        qualifications_version=QUALIFICATIONS_VERSION,
    )


def rank_vacancies(
    vacancies: list[Vacancy],
    profile: CandidateProfile,
    *,
    now: datetime | None = None,
    include_ineligible: bool = False,
    trace: dict | None = None,
) -> list[tuple[Vacancy, MatchResult]]:
    reference_time = now or datetime.now(UTC)
    started = perf_counter()
    results = [
        (vacancy, score_vacancy(vacancy, profile, now=reference_time)) for vacancy in vacancies
    ]
    if trace is not None:
        trace["rule_scoring_seconds"] = perf_counter() - started
    started = perf_counter()
    eligible = [
        (vacancy, result) for vacancy, result in results if result.eligible or include_ineligible
    ]
    if trace is not None:
        trace["eligibility_filter_seconds"] = perf_counter() - started
    started = perf_counter()
    ranked = sorted(
        eligible,
        key=lambda pair: (-pair[1].score, -pair[0].published_at.timestamp(), pair[0].source_id),
    )
    if trace is not None:
        trace["rule_sort_seconds"] = perf_counter() - started
    return ranked
