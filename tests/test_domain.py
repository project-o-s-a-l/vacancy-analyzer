from copy import deepcopy
from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from domain import CandidateProfile, Vacancy


def values(vacancy):
    return vacancy.model_dump()


def test_domain_strips_fields_without_touching_raw_json(vacancy):
    data = values(vacancy)
    data.update(
        title="  Python  ", skills=[" Python "], raw_payload={"name": "  Keep  "}
    )
    result = Vacancy(**data)
    assert result.title == "Python"
    assert result.skills == ["Python"]
    assert result.raw_payload == {"name": "  Keep  "}


@pytest.mark.parametrize("field", ["source", "source_id", "title", "company"])
def test_blank_required_string_rejected(vacancy, field):
    data = values(vacancy)
    data[field] = "   "
    with pytest.raises(ValidationError):
        Vacancy(**data)


@pytest.mark.parametrize("url", ["ftp://example.com/a", "not-a-url", ""])
def test_invalid_url_rejected(vacancy, url):
    data = values(vacancy)
    data["url"] = url
    with pytest.raises(ValidationError):
        Vacancy(**data)


@pytest.mark.parametrize("field", ["published_at", "collected_at"])
def test_naive_dates_rejected(vacancy, field):
    data = values(vacancy)
    data[field] = datetime(2026, 10, 1)
    with pytest.raises(ValidationError):
        Vacancy(**data)


def test_dates_are_normalized_to_utc(vacancy):
    data = values(vacancy)
    data["published_at"] = datetime(
        2026, 10, 16, 12, tzinfo=timezone(timedelta(hours=11))
    )
    result = Vacancy(**data)
    assert result.published_at == datetime(2026, 10, 16, 1, tzinfo=UTC)


@pytest.mark.parametrize("salary", [-1, 12.5, True, "100"])
def test_salary_must_be_nonnegative_integer(vacancy, salary):
    data = values(vacancy)
    data["salary_from"] = salary
    with pytest.raises(ValidationError):
        Vacancy(**data)


def test_reversed_salary_rejected(vacancy):
    data = values(vacancy)
    data.update(salary_from=200, salary_to=100)
    with pytest.raises(ValidationError):
        Vacancy(**data)


def test_unknown_fields_rejected(vacancy):
    with pytest.raises(ValidationError):
        Vacancy(**values(vacancy), unknown="value")


def test_default_id_and_skills_are_independent(vacancy):
    data = values(vacancy)
    for name in ["id", "collected_at", "skills"]:
        data.pop(name)
    first, second = Vacancy(**data), Vacancy(**data)
    first.skills.append("python")
    assert first.id != second.id
    assert second.skills == []
    assert first.collected_at.tzinfo == UTC


def test_profile_validates_and_does_not_guess_experience(profile):
    data = deepcopy(profile.model_dump())
    data.pop("experience_years")
    result = CandidateProfile(**data)
    assert result.experience_years is None


@pytest.mark.parametrize(
    "data",
    [
        {"experience_years": -1},
        {"skills": {" ": {}}},
        {"skills": {"python": {"years": float("nan")}}},
        {"preferences": {"remote": "maybe"}},
        {"preferences": {"salary_min": True}},
        {"preferences": {"currency": " "}},
        {"preferences": {"positions": [" "]}},
    ],
)
def test_invalid_profile_rejected(data):
    with pytest.raises(ValidationError):
        CandidateProfile.model_validate(data)
