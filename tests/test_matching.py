from copy import deepcopy
from datetime import timedelta

import pytest

from domain import CandidateProfile, Vacancy
from nlp.extraction import canonical_skill, extract_skills
from processing.cleaner import clean_html
from scoring.baseline import rank_vacancies, score_vacancy
from scoring.skill_score import match_skills


def test_html_cleaning_preserves_word_and_section_boundaries():
    assert clean_html("<p>Python &amp; SQL</p><p>  Docker&nbsp; </p>") == "Python & SQL\nDocker"
    assert clean_html("Fast<b>API</b><script>Java</script>") == "FastAPI"


def test_aliases_boundaries_and_unknown_explicit_skills():
    result = extract_skills(
        "Python3, postgres, k8s, sklearn, C++, C# и .NET. JavaScript и MySQL.",
        ["  Internal Framework  ", "PostgreSQL"],
    )
    assert {"python", "postgresql", "kubernetes", "scikit-learn", "c++", "c#", ".net"} <= set(
        result
    )
    assert "internal framework" in result
    assert "java" not in result
    assert "sql" not in result
    assert result.count("postgresql") == 1
    assert canonical_skill("  PSQL  ") == "postgresql"


def test_required_skills_have_more_weight_and_are_not_double_counted():
    result = match_skills(["python"], ["python", "postgres"], ["docker", "Python"])
    assert result.score == pytest.approx(3 / 7)
    assert result.matched == ("python",)
    assert result.missing == ("docker", "postgresql")
    assert match_skills([], []).score is None


def test_relevant_job_ranks_above_unrelated_job(sample_data, profile, now):
    vacancies = [Vacancy.model_validate(payload) for payload in sample_data["vacancies"]]
    ranking = rank_vacancies(list(reversed(vacancies)), profile, now=now, include_ineligible=True)
    assert [vacancy.source_id for vacancy, _ in ranking] == ["demo-1", "demo-2", "demo-3"]
    assert ranking[0][1].score > 95
    assert ranking[-1][1].score < 35
    assert not ranking[-1][1].eligible
    assert [v.source_id for v, _ in rank_vacancies(vacancies, profile, now=now)] == [
        "demo-1",
        "demo-2",
    ]


def test_missing_information_cannot_produce_full_match(vacancy, now):
    data = deepcopy(vacancy.model_dump())
    data.update(skills=[], description=None, experience=None, published_at=now)
    result = score_vacancy(Vacancy(**data), CandidateProfile(), now=now)
    assert result.score == 5
    assert result.evidence_coverage == 0.05
    assert result.components["skills"] is None


def test_unset_preferences_do_not_cost_points_but_requested_unknown_conditions_do(vacancy, now):
    job = vacancy.model_copy(
        update={
            "skills": ["Git"],
            "description": "Требования: Git",
            "experience": "no_experience",
            "salary_from": None,
            "salary_to": None,
            "remote": None,
            "published_at": now,
        }
    )
    candidate = CandidateProfile(skills={"git": {}})
    assert score_vacancy(job, candidate, now=now).score == 100
    candidate.preferences.salary_min = 0
    assert score_vacancy(job, candidate, now=now).score == 100
    candidate.preferences.salary_min = 100000
    candidate.preferences.remote = "preferred"
    match = score_vacancy(job, candidate, now=now)
    assert match.score == 80 and match.evidence_coverage == 0.8


def test_job_open_to_beginners_does_not_get_extra_experience_credit(vacancy, now):
    job = vacancy.model_copy(
        update={
            "skills": ["Java", "Docker", "Git"],
            "description": "Требования: Java, Docker, Git",
            "experience": "no_experience",
            "published_at": now,
        }
    )
    candidate = CandidateProfile(skills={"git": {}})
    result = score_vacancy(job, candidate, now=now)
    assert result.components["experience"] == 1 and result.components["skills"] == pytest.approx(
        1 / 3
    )
    assert (
        result.score == 50
    )  # 75 points for skills, 20 for the open experience band, 5 for recency.


@pytest.mark.parametrize("remote", [False, None])
def test_required_remote_filters_nonmatching_and_unknown(vacancy, profile, now, remote):
    data = deepcopy(vacancy.model_dump())
    data["remote"] = remote
    profile.preferences.remote = "required"
    job = Vacancy(**data)
    assert not score_vacancy(job, profile, now=now).eligible
    assert rank_vacancies([job], profile, now=now) == []


@pytest.mark.parametrize(
    "changes",
    [
        {"currency": "USD"},
        {"salary_gross": False},
        {"salary_gross": None},
        {"salary_period": "HOUR"},
        {"salary_from": None, "salary_to": None},
    ],
)
def test_incomparable_salary_is_unknown(vacancy, profile, now, changes):
    data = vacancy.model_dump()
    data.update(changes)
    result = score_vacancy(Vacancy(**data), profile, now=now)
    assert result.components["salary"] is None
    assert result.warnings


def test_unknown_experience_not_inferred_from_skill_years(vacancy, profile, now):
    profile.experience_years = None
    assert score_vacancy(vacancy, profile, now=now).components["experience"] is None


def test_freshness_clamps_future_and_old_dates(vacancy, profile, now):
    for date, expected in [(now + timedelta(days=5), 1), (now - timedelta(days=100), 0)]:
        data = vacancy.model_dump()
        data["published_at"] = date
        result = score_vacancy(Vacancy(**data), profile, now=now)
        assert result.components["freshness"] == expected
        assert 0 <= result.score <= 100


def test_spring_is_recognized_in_description_and_aliases_without_false_substrings():
    assert canonical_skill("Spring Boot") == "spring"
    assert extract_skills("Java, Spring Boot, SQL") == ["java", "spring", "sql"]
    assert "spring" not in extract_skills("Springfield")


def test_desired_job_wording_does_not_change_professional_fit(vacancy, now):
    profile = CandidateProfile.model_validate(
        {
            "skills": {"java": {}, "spring": {}, "sql": {}},
            "experience_years": 0,
            "preferences": {"positions": ["Java Developer"]},
        }
    )
    job = Vacancy(
        **{
            **vacancy.model_dump(),
            "title": "Java-разработчик",
            "skills": [],
            "description": "Java, Spring Boot, SQL",
            "experience": "no_experience",
            "published_at": now,
        }
    )
    result = score_vacancy(job, profile, now=now)
    assert result.components["title"] is None
    assert result.components["skills"] == 1
    assert result.matched_skills == ["java", "spring", "sql"]
    assert result.score == 100
    for positions in (["Java-разработчик"], ["Разроботчик на Java"], [], ["Инженер ПО"]):
        profile.preferences.positions = positions
        assert score_vacancy(job, profile, now=now).score == result.score
    job.experience = "1-3"
    no_experience = score_vacancy(job, profile, now=now)
    assert no_experience.components["experience"] == 0
    assert no_experience.score == pytest.approx(result.score - 95 / 3, abs=0.01)
    assert profile.experience_years == 0
