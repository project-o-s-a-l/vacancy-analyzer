from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from jobs.matching_audit import AuditCases, audit_cases, catalog_snapshot, load_audit_cases
from nlp.resume import extract_resume

CASES_PATH = Path(__file__).parents[1] / "config/evaluation/multirole.yaml"
CASES = load_audit_cases(CASES_PATH).cases


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_resume_acceptance_cases_cover_six_professions(case):
    actual = extract_resume(case.text).model_dump(exclude={"notes"})
    assert actual == case.expected.model_dump(exclude={"notes"})


@pytest.mark.parametrize("heading", ["Документы", "Допуски", "Портфолио", "Licenses", "Portfolio"])
def test_non_skill_sections_do_not_become_skills(heading):
    result = extract_resume(f"Компетенции: Excel\n{heading}\nНекоторые дополнительные сведения")
    assert result.skills == ["excel"]


@pytest.mark.parametrize("label", ["Общий стаж работы", "Total work experience"])
def test_explicit_unknown_and_zero_experience_remain_distinct(label):
    assert extract_resume(label + ": no experience").experience_years == 0
    assert extract_resume(label + ": уточните").experience_years is None


def test_audit_case_identifiers_cannot_repeat():
    data = load_audit_cases(CASES_PATH).model_dump()
    data["cases"].append(data["cases"][0])
    with pytest.raises(ValidationError, match="уникальны"):
        AuditCases.model_validate(data)


async def test_audit_uses_isolated_database_and_preserves_source_snapshot(vacancy, tmp_path):
    case = load_audit_cases(CASES_PATH).cases[0]
    now = datetime.now(UTC)
    vacancy = vacancy.model_copy(update={"published_at": now, "valid_until": None})
    snapshot = catalog_snapshot([vacancy], now=now)
    assert snapshot["vacancies"][0]["raw_payload"] == {}
    original = repr(snapshot)
    specification = AuditCases(synthetic_resumes=True, cases=[case])
    report = await audit_cases(specification, snapshot, tmp_path / "audit", min_score=0)
    assert report["working_database_modified"] is False
    assert report["live_telegram_messages_sent"] is False
    assert report["resume_source"] == "synthetic"
    checks = report["cases"][0]
    assert all(checks["extraction_checks"].values())
    assert checks["operational_checks"] == {
        "draft_preserved_profile": True,
        "delivery_and_feedback": True,
    }
    assert repr(snapshot) == original
    with pytest.raises(FileExistsError):
        await audit_cases(specification, snapshot, tmp_path / "audit")


async def test_audit_does_not_accept_naive_snapshot_time(tmp_path):
    specification = AuditCases(synthetic_resumes=True, cases=[CASES[0]])
    with pytest.raises(ValueError, match="пояс"):
        await audit_cases(
            specification,
            {
                "version": "multirole-catalog-v1",
                "captured_at": "2026-10-19T00:00:00",
                "vacancies": [],
            },
            tmp_path / "audit",
        )
    assert not (tmp_path / "audit").exists()


async def test_same_vacancy_is_delivered_and_saved_independently_for_two_cases(vacancy, tmp_path):
    now = datetime.now(UTC)
    vacancy = vacancy.model_copy(update={"published_at": now, "valid_until": None})
    specification = AuditCases(synthetic_resumes=True, cases=[CASES[0], CASES[2]])
    report = await audit_cases(
        specification, catalog_snapshot([vacancy], now=now), tmp_path / "two", min_score=0
    )
    assert len(report["cases"]) == 2
    for case in report["cases"]:
        assert case["operational_checks"]["delivery_and_feedback"] is True
        assert case["local_transport_messages"] == [vacancy.source_id]
    assert report["cases"][0]["confirmed_profile"] != report["cases"][1]["confirmed_profile"]
