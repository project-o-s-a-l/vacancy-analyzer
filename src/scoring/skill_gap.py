from statistics import mean

from domain import CandidateProfile, Vacancy
from domain.candidate import Skill
from domain.skill_gap import SampleScope, SkillGapReport, SkillOpportunity
from nlp.extraction import canonical_skill, skill_key
from scoring.comparison import compare_facts
from scoring.eligibility import check_eligibility
from scoring.request_context import prepared_profile
from scoring.retrieval import tokens
from scoring.serving_cache import cache_scope
from scoring.title_aliases import title_variants


def direction_terms(positions):
    return tuple(
        {
            form.terms
            for position in positions
            for form in title_variants(position, tokens)
            if form.terms
        }
    )


def in_direction(vacancy: Vacancy, positions: list[str]) -> bool:
    words = tokens(vacancy.title)
    return not positions or any(form <= words for form in direction_terms(positions))


def _skill_items(comparison):
    return [item for item in comparison.items if item.kind == "skill"]


def _coverage(items):
    return mean(item.state == "match" for item in items) if items else None


def skill_gap_report(
    vacancies: list[Vacancy], scope: SampleScope, profile: CandidateProfile
) -> SkillGapReport:
    with cache_scope(None), prepared_profile(profile):
        comparisons = {v.id: compare_facts(v, profile, now=scope.until) for v in vacancies}
        eligibility = {
            v.id: check_eligibility(v, profile, now=scope.until, comparison=comparisons[v.id])
            for v in vacancies
        }
    selected = [v for v in vacancies if eligibility[v.id].eligible]
    items = {v.id: _skill_items(comparisons[v.id]) for v in selected}
    recorded = [v for v in selected if items[v.id]]
    names, appearances = {}, {}
    for vacancy in recorded:
        for fact in vacancy.professional_facts.facts:
            if fact.kind == "skill" and fact.assertion.value is True:
                key = skill_key(fact.name)
                names[key] = canonical_skill(fact.name)
                appearances.setdefault(key, {})[vacancy.id] = vacancy
    opportunities = []
    for key, relevant in appearances.items():
        # Counterfactual changes membership only. It cannot invent years/level or
        # satisfy a conjunction of duration and presence by adding a name.
        hypothetical = profile.model_copy(deep=True)
        hypothetical.skills[names[key]] = Skill()
        if hypothetical.professional_facts:
            for fact in hypothetical.professional_facts.facts:
                if fact.kind == "skill" and skill_key(fact.name) == key:
                    fact.assertion = type(fact.assertion)(status="present", value=True)
        gains, newly_covered = [], 0
        with cache_scope(None), prepared_profile(hypothetical):
            for vacancy in relevant.values():
                before = _coverage(items[vacancy.id])
                after = _coverage(
                    _skill_items(compare_facts(vacancy, hypothetical, now=scope.until))
                )
                gain = (after or 0) - (before or 0)
                if gain > 0:
                    gains.append(gain)
                    newly_covered += after == 1 and before != 1
        if gains:
            opportunities.append(
                SkillOpportunity(
                    skill=names[key],
                    vacancies=len(gains),
                    mean_coverage_gain_pp=round(100 * sum(gains) / len(recorded), 2),
                    additional_fully_covered=newly_covered,
                )
            )
    # Experience counts cover the examined direction, including exclusions, so a
    # hard deficit does not disappear from the analysis merely because it was filtered.
    experience = [
        [
            i
            for i in comparisons[v.id].items
            if i.years_required is not None or i.id.startswith("req:experience:")
        ]
        for v in vacancies
    ]
    compared = [i for v in selected for i in comparisons[v.id].items]
    return SkillGapReport(
        scope=scope.model_copy(update={"vacancies": len(selected), "with_skills": len(recorded)}),
        excluded_by_remote=sum(
            any(i.field == "remote" for i in e.exclusions) for e in eligibility.values()
        ),
        excluded_by_constraints=sum(not e.eligible for e in eligibility.values()),
        requirement_coverage=_coverage(compared),
        uncertain_requirements=sum(i.state in {"unknown", "conflict"} for i in compared),
        confirmed_deficits=sum(i.state == "contradiction" for i in compared),
        mean_skill_coverage=mean(_coverage(items[v.id]) for v in recorded) if recorded else None,
        fully_covered=sum(_coverage(items[v.id]) == 1 for v in recorded),
        insufficient_experience=sum(
            any(i.state == "contradiction" for i in group) for group in experience
        ),
        unknown_experience=sum(
            not group
            or all(i.years_required is None for i in group)
            or any(
                i.state in {"unknown", "conflict"} and i.years_required is not None for i in group
            )
            for group in experience
        ),
        opportunities=sorted(
            opportunities, key=lambda i: (-i.vacancies, -i.mean_coverage_gain_pp, i.skill)
        )[:20],
    )


def _scope_lines(scope: SampleScope) -> list[str]:
    lines = [
        "Направление: " + (", ".join(scope.positions) if scope.positions else "весь каталог"),
        f"Публикации: {scope.since:%d.%m.%Y} — {scope.until:%d.%m.%Y} (UTC)",
        f"В выборке: {scope.vacancies}; с указанными навыками: {scope.with_skills}.",
    ]
    if scope.truncated:
        lines.append(f"Выборка ограничена. Учтены последние {scope.examined} вакансий направления.")
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
            f"Все требования к навыкам подтверждены: {report.fully_covered} вакансий.",
            "Если добавить подтверждённый навык в профиль:",
        ]
        lines += [
            f"• {v.skill[:100]}: в {v.vacancies} вак.; "
            f"+{v.mean_coverage_gain_pp:g} п.п. покрытия; "
            f"ещё {v.additional_fully_covered} вак. с покрытием требований к навыкам."
            for v in report.opportunities[:8]
        ] or ["Пробелов среди указанных навыков не найдено."]
    lines += [
        f"Исключено общими ограничениями: {report.excluded_by_constraints}. "
        f"Неопределённых требований: {report.uncertain_requirements}; "
        f"подтверждённых дефицитов: {report.confirmed_deficits}.",
        f"Недостаточный опыт по нижней границе: {report.insufficient_experience}; "
        f"сравнение опыта невозможно: {report.unknown_experience}.",
        report.scope.note,
        report.note,
    ]
    return "\n".join(lines)
