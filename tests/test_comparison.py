"""Behavioral gates: evidence states, scoped tenure, SQL/CLI/evaluator parity."""

from datetime import date, timedelta
from uuid import uuid4

import pytest
from test_saved_selections import selection_sessions  # noqa: F401

from bot.cards import render_card
from domain import CandidateProfile
from domain.comparison import COMPARISON_FEATURES
from domain.professional import (
    Claim,
    ExperienceEpisode,
    Requirement,
    RoleFact,
    TaskFact,
)
from jobs.admission import admit
from jobs.cli import render
from jobs.rank import rank_recent
from jobs.retrieval_evaluation import MemoryCandidateSource
from jobs.skill_gap import analyze_skill_gap
from nlp.professional_facts import candidate_facts, vacancy_facts
from scoring import score_vacancy
from scoring.comparison import compare_facts
from scoring.linear_ranker import FEATURE_NAMES, feature_vector
from scoring.retrieval import retrieval_eligibility
from scoring.serving_features import pair_features, shown_context
from storage.repositories import VacancyRepository


def claim(value):
    return Claim(status="present", value=value)


def job(vacancy, now, text="Требования: Python обязателен.", **changes):
    v = vacancy.model_copy(
        update=dict(
            id=uuid4(),
            source_id=uuid4().hex,
            title="Разработчик",
            description=text,
            description_html=None,
            analysis=None,
            professional_facts=None,
            skills=[],
            experience=None,
            published_at=now,
            remote=True,
        )
        | changes
    )
    v.professional_facts = vacancy_facts(v)
    return v


def person(now, years=None, *, level=None, total=10):
    p = CandidateProfile(skills={"Python": {}}, experience_years=total)
    p.professional_facts = candidate_facts(p, 42, reference_time=now)
    skill = next(f for f in p.professional_facts.facts if f.kind == "skill")
    if years is not None:
        skill.years = type(skill.years)(status="present", value=years)
    if level is not None:
        skill.level = type(skill.level)(status="present", value=level)
    return p


def skill_requirement(v, years=3, importance="required"):
    f = next(f for f in v.professional_facts.facts if f.kind == "skill")
    r = next(r for r in v.professional_facts.requirements if r.fact_id == f.id)
    r.years_min = claim(years)
    r.importance = claim(importance)
    return f, r


def test_same_total_and_names_different_scoped_years(vacancy, now):
    v = job(vacancy, now)
    skill_requirement(v)
    junior, experienced = person(now, 1), person(now, 4)
    low, high = [score_vacancy(v, p, now=now) for p in (junior, experienced)]
    assert junior.experience_years == experienced.experience_years
    assert not low.eligible and high.eligible
    assert low.score < high.score
    assert high.comparison.features["skill_years_fit"] == 1
    assert low.comparison.features["required_deficit"] > 0
    assert not admit([(v, low)], min_score=0)[2]
    assert "подтверждено несоответствие" in render([(v, low)], as_json=False)
    assert "Не соответствует" in render_card(v, low)
    assert pair_features(v, junior, low) != pair_features(v, experienced, high)
    features = pair_features(v, experienced, high)
    assert len(feature_vector(features)) == len(FEATURE_NAMES)
    assert set(features.comparison_features) == set(COMPARISON_FEATURES)
    assert (
        shown_context(v, experienced, high, 1)["comparison"]["version"]
        == "professional-comparison-v1"
    )


def test_unknown_is_not_zero_and_conflict_not_best_value(vacancy, now):
    v = job(vacancy, now)
    skill_requirement(v)
    p = person(now, total=20)
    match = score_vacancy(v, p, now=now)
    assert match.eligible and match.components["experience"] is None
    assert match.comparison.features["required_coverage"] == 0
    skill = next(f for f in p.professional_facts.facts if f.kind == "skill")
    skill.years = type(skill.years)(status="conflict", alternatives=[1, 8])
    result = compare_facts(v, p, now=now)
    assert result.features["conflict_fraction"] > 0
    assert result.features["skill_years_fit"] is None
    absent = CandidateProfile(experience_years=20)
    assert score_vacancy(v, absent, now=now).eligible
    skill.assertion = type(skill.assertion)(status="explicit_absent")
    assert not score_vacancy(v, p, now=now).eligible


def test_level_and_recency_only_affect_relevant_requirements(vacancy, now):
    v = job(vacancy, now)
    low, high = person(now, 1, level="beginner"), person(now, 4, level="advanced")
    a, b = [score_vacancy(v, p, now=now) for p in (low, high)]
    assert a.score == b.score and a.comparison.features == b.comparison.features
    skill, _ = skill_requirement(v, years=0, importance="preferred")
    skill.level = type(skill.level)(status="present", value="advanced")
    skill.last_used = type(skill.last_used)(status="present", value=date(2025, 1, 1))
    for p, last in ((low, date(2020, 1, 1)), (high, date(2026, 1, 1))):
        f = next(f for f in p.professional_facts.facts if f.kind == "skill")
        f.last_used = type(f.last_used)(status="present", value=last)
    a, b = [score_vacancy(v, p, now=now) for p in (low, high)]
    assert a.eligible and b.eligible and a.score < b.score
    assert a.comparison.features["skill_recency_fit"] == 0
    assert b.comparison.features["skill_level_fit"] == 1


def add_role(p, role="Разработчик", *, project=False, duplicate=False, unknown=False):
    f = RoleFact(id="held", name=role, context="held", assertion=claim(True))
    t = TaskFact(id="task", name="Разрабатывать отчёты", assertion=claim(True))
    p.professional_facts.facts.extend([f, t])
    episode = ExperienceEpisode(
        id="job1",
        kind="project" if project else "employment",
        context="academic" if project else "work",
        started_precision="month",
        ended_precision="month",
        started=claim(date(2022, 1, 1)),
        ended=Claim(status="unknown") if unknown else claim(date(2025, 12, 31)),
        role_ids=[f.id],
        task_ids=[t.id],
    )
    p.professional_facts.episodes.append(episode)
    if duplicate:
        p.professional_facts.episodes.append(episode.model_copy(update={"id": "job2"}))


def test_career_change_tasks_calendar_overlap_and_projects(vacancy, now):
    v = job(vacancy, now, "Требования: Опыт работы от 3 лет.\nОбязанности:\nРазрабатывать отчёты.")
    p, changed, project = person(now), person(now), person(now)
    add_role(p, duplicate=True)
    add_role(changed, role="Бухгалтер")
    add_role(project, project=True)
    results = [score_vacancy(v, x, now=now) for x in (p, changed, project)]
    assert results[0].score > results[1].score
    assert results[0].comparison.features["task_match"] == 1
    assert results[0].comparison.features["role_exposure"] == pytest.approx(0.8, abs=0.001)
    assert results[2].comparison.features["role_exposure"] is None
    assert results[2].comparison.features["project_match"] == 1
    assert results[2].components["experience"] is None
    # Role association is not proof of four years performing every linked task.
    assert results[0].comparison.features["required_coverage"] == 0
    p.professional_facts.episodes[0].ended = Claim(
        status="conflict", alternatives=[date(2023, 1, 1), date(2025, 1, 1)]
    )
    p.professional_facts.episodes.pop()
    assert compare_facts(v, p, now=now).features["role_exposure"] is None


def test_alternative_duration_and_independent_and_requirement(vacancy, now):
    v = job(vacancy, now, "Требования: Python или SQL обязательны.")
    facts = v.professional_facts
    group = facts.alternatives[0]
    for r in facts.requirements:
        if r.id in group.requirement_ids:
            r.years_min = claim(3)
    p = person(now, 4)
    result = compare_facts(v, p, now=now)
    assert result.features["alternative_coverage"] == 1
    assert result.features["skill_years_fit"] == 1
    assert not result.exclusions
    sql = next(f for f in facts.facts if f.kind == "skill" and f.name == "sql")
    facts.requirements.append(
        Requirement(
            id="independent",
            fact_id=sql.id,
            scope="skill",
            scope_id=sql.id,
            importance=claim("required"),
        )
    )
    from domain.professional import SkillFact

    p.professional_facts.facts.append(
        SkillFact(id="sql", name="SQL", assertion=Claim(status="explicit_absent"))
    )
    assert compare_facts(v, p, now=now).exclusions[0].id == "independent"


def test_documents_or_expiry_and_conflicting_sources(vacancy, now):
    from domain.qualifications import Qualification

    v = job(vacancy, now, "Требования: медкнижка или сертификат AWS обязательны.")
    p = CandidateProfile(qualifications=[Qualification(name="медкнижка", valid_until=now.date())])
    result = score_vacancy(v, p, now=now)
    assert result.eligible and result.comparison.features["alternative_coverage"] == 1
    p.qualifications.append(Qualification(name="сертификат AWS", status="absent"))
    assert not score_vacancy(v, p, now=now + timedelta(days=1)).eligible
    p.qualifications.append(Qualification(name="медицинская книжка", status="absent"))
    result = score_vacancy(v, p, now=now)
    assert result.eligible and result.comparison.features["conflict_fraction"] > 0


def test_ready_facts_no_extraction_and_changed_conditions_repair(vacancy, now, monkeypatch):
    v = job(vacancy, now, "Требования: Python и реставрация струнных инструментов.")
    p = person(now, 4)
    from domain.candidate import Skill

    p.skills["реставрация струнных инструментов"] = Skill()

    def forbidden(*a, **kw):
        raise AssertionError("Extraction on a ready pair")

    with monkeypatch.context() as patch:
        patch.setattr("nlp.professional_facts.vacancy_facts", forbidden)
        patch.setattr("nlp.requirements.analyze_requirements", forbidden)
        patch.setattr("nlp.qualifications.compare_qualifications", forbidden)
        patch.setattr("nlp.experience_requirement.experience_requirement", forbidden)
        assert "реставрация струнных инструментов" in score_vacancy(v, p, now=now).matched_skills
        retrieval_eligibility(v, p, now)
    old = v.professional_facts.revision
    v.remote = False
    p.preferences.remote = "required"
    assert not score_vacancy(v, p, now=now).eligible
    assert v.professional_facts.revision != old


@pytest.mark.parametrize("title", ["Бухгалтер", "Водитель", "Медсестра", "Повар"])
def test_profession_independent_role_exposure(vacancy, now, title):
    v = job(vacancy, now, "Требования: Опыт работы от 2 лет.", title=title)
    p = person(now)
    add_role(p, role=title)
    assert compare_facts(v, p, now=now).features["role_exposure"] > 0


async def test_sql_memory_cli_gap_parity_and_pool_refill(selection_sessions, vacancy, now):  # noqa: F811
    p = person(now, 1)
    p.preferences.positions = ["Разработчик"]
    bad = job(vacancy, now, "Требования: Python обязателен. Опыт с Python от 3 лет.")
    good = job(vacancy, now, "Требования: Python обязателен.")
    async with selection_sessions.begin() as session:
        repo = VacancyRepository(session)
        for v in (bad, good):
            await repo.save(v)
        sql = await rank_recent(repo, p, now=now, limit=1)
        offline = await rank_recent(
            repo,
            p,
            now=now,
            limit=1,
            candidate_source=MemoryCandidateSource([bad, good]),
        )
        assert [v.id for v, _ in sql] == [good.id] == [v.id for v, _ in offline]
        assert sql[0][1].comparison == offline[0][1].comparison
        gap = await analyze_skill_gap(repo, profile=p, now=now)
        assert gap.excluded_by_constraints == 1 and gap.scope.vacancies == 1
        assert gap.mean_skill_coverage == 1
        assert "professional-comparison-v1" in render(sql, as_json=True)


def test_extracted_alternative_duration_is_or_not_two_mandatory_tenures(vacancy, now):
    v = job(vacancy, now, "Требования: Опыт с Python или SQL от 3 лет.")
    p = person(now, 4)
    assert v.professional_facts.alternatives
    result = score_vacancy(v, p, now=now)
    assert result.eligible and result.comparison.features["alternative_coverage"] == 1
    assert result.comparison.features["skill_years_fit"] == 1
    assert not any(i.kind == "skill" and i.state == "unknown" for i in result.comparison.items)


def test_unsupported_experience_alternative_never_becomes_strict_and(vacancy, now):
    v = job(vacancy, now, "Требования: Опыт работы от 3 лет или учебные проекты.")
    assert score_vacancy(v, person(now, total=0), now=now).eligible


def test_explicit_level_extracted_and_conflicting_dates_preserved(vacancy, now):
    v = job(vacancy, now, "Требования: Python уровень advanced.")
    assert not score_vacancy(v, person(now, level="beginner"), now=now).eligible
    assert score_vacancy(v, person(now, level="advanced"), now=now).eligible


def test_condition_conflict_and_strict_unknown_are_not_confirmed_contradictions(vacancy, now):
    v = job(vacancy, now)
    remote = next(f for f in v.professional_facts.facts if f.id == "condition:remote")
    remote.assertion = type(remote.assertion)(status="conflict", alternatives=[True, False])
    p = person(now)
    p.preferences.remote = "required"
    result = score_vacancy(v, p, now=now)
    assert not result.eligible
    assert any(i.state == "conflict" and i.mandatory for i in result.eligibility.exclusions)
    assert result.components["preferences"] is None


async def test_skill_gap_respects_or_and_does_not_invent_tenure(session, vacancy, now):
    p = person(now, 1)
    p.preferences.positions = ["Разработчик"]
    v = job(
        vacancy, now, "Требования: Python или SQL обязательны.\nЖелательно: опыт с Python от 3 лет."
    )
    await VacancyRepository(session).save(v)
    report = await analyze_skill_gap(VacancyRepository(session), profile=p, now=now)
    assert report.scope.vacancies == 1
    assert report.mean_skill_coverage == 0.5
    assert not report.opportunities  # Adding SQL cannot improve the already satisfied OR.


def test_episode_scope_cannot_borrow_global_skill_tenure(vacancy, now):
    v = job(vacancy, now)
    _, requirement = skill_requirement(v)
    v.professional_facts.episodes.append(ExperienceEpisode(id="specific-project", kind="project"))
    requirement.scope, requirement.scope_id = "episode", "specific-project"
    result = compare_facts(v, person(now, 20), now=now)
    assert result.features["skill_years_fit"] is None
    assert result.items[0].state == "unknown"
