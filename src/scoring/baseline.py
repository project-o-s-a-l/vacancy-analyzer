"""Transparent heuristic scores. Missing evidence gets no invented positive match."""

from datetime import UTC, datetime

from domain import CandidateProfile, Vacancy
from domain.matching import MatchResult
from nlp.analysis_identity import current_vacancy_analysis as current_analysis
from nlp.extraction import skill_key
from nlp.qualifications import QUALIFICATIONS_VERSION, compare_qualifications, is_qualification
from nlp.requirements import RULES_VERSION, analyze_requirements
from nlp.titles import title_tokens
from nlp.vacancy_requirements import analyzed_requirements
from processing.cleaner import clean_html
from scoring.eligibility import check_eligibility, location_matches, salary_comparable
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
    text = vacancy.description
    if vacancy.description_html and (
        text is None or clean_html(vacancy.description_html) == clean_html(text)
    ):
        text = vacancy.description_html
    qualifications, skill_text = compare_qualifications(
        text, profile.qualifications, today=now.astimezone(UTC).date(), explicit=vacancy.skills
    )
    eligibility = check_eligibility(vacancy, profile, now=now, qualifications=qualifications)
    warnings.extend(item.message for item in [*eligibility.exclusions, *eligibility.uncertainties])
    analysis = current_analysis(vacancy)
    use_model_requirements = bool(
        analysis and analysis.input_scope == "full" and analysis.requirements_source == "llm"
    )
    requirements = (
        analyzed_requirements(analysis)
        if use_model_requirements
        else analyze_requirements(
            skill_text,
            [value for value in vacancy.skills if not is_qualification(value)],
            vocabulary=[value for value in profile.skills if not is_qualification(value)],
        )
    )
    skills = match_skills(
        profile.skills,
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
            warnings.append(f"Один из навыков не найден в профиле ({label}): {options}")
    if skills.score is None:
        warnings.append("Навыки вакансии не определены")

    minimum = eligibility.experience.minimum_years
    experience = None
    if (
        minimum is not None
        and profile.experience_years is not None
        and not (minimum == 0 and eligibility.experience.conflict)
    ):
        experience = min(1.0, profile.experience_years / minimum) if minimum else 1.0
        if experience == 1:
            reasons.append("Достаточный опыт по нижней границе требований")
        else:
            warnings.append(f"Требуется опыт от {minimum} лет")

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

    salary = None
    if preferences.salary_min is not None:
        comparable = salary_comparable(vacancy, profile)
        upper = vacancy.salary_to if vacancy.salary_to is not None else vacancy.salary_from
        if comparable and upper is not None:
            salary = min(1.0, upper / preferences.salary_min) if preferences.salary_min else 1.0
            if vacancy.salary_from is None or vacancy.salary_from < preferences.salary_min:
                warnings.append("Нижняя граница зарплаты не подтверждает желаемый минимум")
        else:
            warnings.append("Зарплата неизвестна или несопоставима по валюте/периоду/gross")

    preference_parts: list[float] = []
    if preferences.remote != "any":
        if vacancy.remote is None:
            warnings.append("Формат работы не указан")
        else:
            remote_match = vacancy.remote if preferences.remote != "office" else not vacancy.remote
            preference_parts.append(float(remote_match))
    if preferences.locations and vacancy.remote is not True:
        if vacancy.location:
            preference_parts.append(float(location_matches(vacancy, profile)))
        else:
            warnings.append("Местоположение не указано")

    age = max(0.0, (now - vacancy.published_at).total_seconds() / 86400)
    components = {
        "skills": skills.score,
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
) -> list[tuple[Vacancy, MatchResult]]:
    reference_time = now or datetime.now(UTC)
    results = [
        (vacancy, score_vacancy(vacancy, profile, now=reference_time)) for vacancy in vacancies
    ]
    return sorted(
        [(vacancy, result) for vacancy, result in results if result.eligible or include_ineligible],
        key=lambda pair: (-pair[1].score, -pair[0].published_at.timestamp(), pair[0].source_id),
    )
