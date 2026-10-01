from collections import Counter
from statistics import mean

from domain import CandidateProfile
from domain.skill_gap import SampleScope, SkillGapReport, SkillGapVacancy, SkillOpportunity
from nlp.experience_requirement import EXPERIENCE_MINIMUM
from nlp.extraction import canonical_skill, skill_key
from nlp.titles import title_tokens


def in_direction(vacancy: SkillGapVacancy, positions: list[str]) -> bool:
    """Any requested position must match all its normalized title words."""
    words = title_tokens(vacancy.title)
    return not positions or any(
        bool(tokens := title_tokens(position)) and tokens <= words for position in positions
    )


def _skill_sets(vacancies: list[SkillGapVacancy]):
    names = {}
    sets = []
    for vacancy in vacancies:
        keys = set()
        for value in vacancy.skills:
            if value.strip():
                key = skill_key(value)
                label = canonical_skill(value)
                # Stable display labels independent of database row order.
                names[key] = min(names.get(key, label), label)
                keys.add(key)
        sets.append(keys)
    return names, sets


def skill_gap_report(
    vacancies: list[SkillGapVacancy], scope: SampleScope, profile: CandidateProfile
) -> SkillGapReport:
    selected = [
        v for v in vacancies if profile.preferences.remote != "required" or v.remote is True
    ]
    names, sets = _skill_sets(selected)
    known = {skill_key(value) for value in profile.skills}
    recorded = [keys for keys in sets if keys]
    missing = Counter(key for keys in recorded for key in keys - known)
    opportunities = [
        SkillOpportunity(
            skill=names[key],
            vacancies=count,
            mean_coverage_gain_pp=round(
                100 * sum(1 / len(keys) for keys in recorded if key in keys) / len(recorded), 2
            ),
            additional_fully_covered=sum(keys - known == {key} for keys in recorded),
        )
        for key, count in missing.items()
    ]
    minimums = [EXPERIENCE_MINIMUM.get(v.experience or "") for v in selected]
    return SkillGapReport(
        scope=scope.model_copy(update={"vacancies": len(selected), "with_skills": len(recorded)}),
        excluded_by_remote=len(vacancies) - len(selected),
        mean_skill_coverage=mean(len(keys & known) / len(keys) for keys in recorded)
        if recorded
        else None,
        fully_covered=sum(keys <= known for keys in recorded),
        insufficient_experience=sum(
            value is not None and value > profile.experience_years for value in minimums
        )
        if profile.experience_years is not None
        else 0,
        unknown_experience=sum(value is None for value in minimums)
        if profile.experience_years is not None
        else len(selected),
        opportunities=sorted(
            opportunities,
            key=lambda item: (-item.vacancies, -item.mean_coverage_gain_pp, item.skill),
        )[:20],
    )


def _scope_lines(scope: SampleScope) -> list[str]:
    lines = [
        "Направление: " + (", ".join(scope.positions) if scope.positions else "весь каталог"),
        f"Публикации: {scope.since:%d.%m.%Y} — {scope.until:%d.%m.%Y} (UTC)",
        f"В выборке: {scope.vacancies}; с указанными навыками: {scope.with_skills}.",
    ]
    if scope.truncated:
        lines.append(f"Выборка ограничена: проверены {scope.examined} последних вакансий каталога.")
    return lines


def render_skill_gap(report: SkillGapReport) -> str:
    lines = ["Какие требования вакансий не отражены в профиле", *_scope_lines(report.scope)]
    if report.excluded_by_remote:
        lines.append(f"Исключено без подтверждённой удалённой работы: {report.excluded_by_remote}.")
    if report.mean_skill_coverage is None:
        lines.append("Для расчёта недостаточно вакансий с явно указанными навыками.")
    else:
        lines += [
            f"Среднее покрытие известных навыков: {report.mean_skill_coverage:.1%}.",
            f"Все указанные навыки есть в профиле: {report.fully_covered} вакансий.",
            "Если добавить подтверждённый навык в профиль:",
        ]
        lines += [
            f"• {v.skill[:100]}: в {v.vacancies} вак.; "
            f"+{v.mean_coverage_gain_pp:g} п.п. покрытия; "
            f"ещё {v.additional_fully_covered} вак. со всеми известными навыками."
            for v in report.opportunities[:8]
        ] or ["Пробелов среди указанных навыков не найдено."]
    lines += [
        f"Недостаточный опыт по нижней границе: {report.insufficient_experience}; "
        f"сравнение опыта невозможно: {report.unknown_experience}.",
        report.scope.note,
        report.note,
    ]
    return "\n".join(lines)
