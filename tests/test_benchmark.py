import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from domain import CandidateProfile
from domain.benchmark import BenchmarkCase, BenchmarkCases, BenchmarkCatalog, BenchmarkJudgment
from jobs.benchmark import acceptance, evaluate_benchmark, prepare_benchmark, run_baseline
from jobs.benchmark_data import (
    blank_judgments,
    deduplicate,
    judgments_text,
    load_bundle,
    load_cases,
    validate_cases,
)
from jobs.benchmark_labels import label_benchmark
from jobs.cli import parser, run_command
from scoring.matching_evaluation import ranking_metrics


def specification():
    return BenchmarkCases(
        cases=[
            BenchmarkCase(
                id="python-zero",
                group="graduate",
                profession="python",
                split="training",
                resume="Fictitious graduate seeks their first Python backend developer job.",
                profile=CandidateProfile.model_validate(
                    {
                        "skills": {"python": {}},
                        "experience_years": 0,
                        "preferences": {"positions": ["Python developer"]},
                    }
                ),
            ),
            BenchmarkCase(
                id="accountant",
                group="finance",
                profession="accountant",
                split="validation",
                resume="A fictional accounting specialist manages primary financial documents.",
                profile=CandidateProfile.model_validate(
                    {
                        "skills": {"excel": {}},
                        "experience_years": 5,
                        "preferences": {"positions": ["Бухгалтер"]},
                    }
                ),
            ),
        ]
    )


@pytest.fixture
def bundle(tmp_path, monkeypatch, vacancy, now):
    import jobs.benchmark as benchmark

    jobs = [
        vacancy.model_copy(
            update={
                "title": "Senior Python Backend Developer",
                "skills": ["python"],
                "experience": "3-6",
                "description": "Требуется коммерческий опыт работы.",
                "published_at": now - timedelta(days=1),
                "archived": False,
                "valid_until": None,
            }
        ),
        vacancy.model_copy(
            update={
                "id": uuid4(),
                "source_id": "first-job",
                "title": "Python developer trainee",
                "skills": ["python"],
                "experience": "no_experience",
                "description": "Без опыта.\nТребования:\nЗнание Python.",
                "published_at": now - timedelta(days=2),
                "archived": False,
                "valid_until": None,
            }
        ),
    ]
    catalog = BenchmarkCatalog(
        reference_time=now, sources=[], skipped=[], duplicates=[], vacancies=jobs
    )
    monkeypatch.setattr(benchmark, "load_cases", lambda _: specification())
    monkeypatch.setattr(benchmark, "snapshot_html", lambda *a, **k: catalog)
    output = tmp_path / "bundle"
    prepare_benchmark(tmp_path, tmp_path / "unused.yaml", output, now=now)
    return output


def test_checked_in_cases_cover_roles_experience_splits_and_no_users():
    cases = load_cases(Path(__file__).parents[1] / "config/evaluation/benchmark.yaml")
    assert len({c.profession for c in cases.cases}) >= 5
    assert {"driver", "accountant", "sales"} <= {c.profession for c in cases.cases}
    assert {c.split for c in cases.cases} == {"training", "validation"}
    years = [c.profile.experience_years for c in cases.cases]
    assert 0 in years and None in years and any(year and year > 0 for year in years)


@pytest.mark.parametrize("change", ["group", "profile", "resume", "duplicate-id"])
def test_split_leakage_is_rejected(change):
    cases = specification()
    left, right = cases.cases
    if change == "duplicate-id":
        right.id = left.id
    else:
        setattr(right, change, getattr(left, change))
    with pytest.raises(ValueError, match="Утечка|Повторные"):
        validate_cases(cases)


def test_dedup_keeps_meaningfully_different_conditions(vacancy):
    body = "Описание обязанностей и требований к работе с важными документами"
    original = vacancy.model_copy(update={"description": body})
    duplicate = original.model_copy(update={"id": uuid4(), "source_id": "duplicate"})
    different = original.model_copy(
        update={
            "id": uuid4(),
            "source_id": "different",
            "remote": not original.remote,
        }
    )
    kept, removed = deduplicate([original, duplicate, different])
    assert len(kept) == 2 and len(removed) == 1
    assert different in kept


def test_recall_uses_unretrieved_relevant_catalog_items():
    metrics = ranking_metrics(["a"], {"a": 3, "b": 3, "c": 0}, top=5)
    assert metrics["recall_at_k"] == 0.5
    assert metrics["precision_at_k"] == 0.2
    assert metrics["precision_among_shown"] == 1
    assert 0 < metrics["ndcg_at_k"] < 1
    empty = ranking_metrics([], {"a": 0})
    assert empty["recall_at_k"] is None and empty["ndcg_at_k"] is None
    assert empty["precision_at_k"] == 0


def test_precision_and_ndcg_are_hand_calculated():
    from math import log2

    metrics = ranking_metrics(["bad", "good"], {"bad": 0, "good": 3, "missed": 2}, top=2)
    assert metrics["precision_at_k"] == 0.5
    assert metrics["ndcg_at_k"] == pytest.approx((7 / log2(3)) / (7 + 3 / log2(3)))
    assert metrics["recall_at_k"] == 0.5


@pytest.mark.parametrize("ranking", [["a", "a"], ["foreign"]])
def test_ranking_ids_are_validated(ranking):
    with pytest.raises(ValueError):
        ranking_metrics(ranking, {"a": 3})


def test_snapshot_allows_rejected_pairs_and_replays_fixed_time(bundle, monkeypatch):
    import scoring.baseline as baseline
    import scoring.eligibility as eligibility

    class NoWallClock(datetime):
        @classmethod
        def now(cls, tz=None):
            raise AssertionError("Ambient clock used for a frozen evaluation")

    monkeypatch.setattr(baseline, "datetime", NoWallClock)
    monkeypatch.setattr(eligibility, "datetime", NoWallClock)
    catalog, cases, rows, manifest = load_bundle(bundle)
    zero = cases.cases[0]
    result = run_baseline(catalog, cases)[zero.id]
    assert len(rows) == len(catalog.vacancies) * len(cases.cases)
    assert len(result["retrieved"]) == 2 and len(result["eligible"]) == 1
    assert any(row.vacancy_key not in result["eligible"] for row in rows if row.case_id == zero.id)
    first = run_baseline(catalog, cases)
    assert first == run_baseline(catalog, cases)
    assert manifest["reference_time"] == catalog.reference_time.isoformat()
    # The relevant second row is absent from the latest-one retrieval but stays in labels.
    limited = run_baseline(catalog, cases, limit=1)[zero.id]
    assert not limited["eligible"]
    assert len(blank_judgments(catalog, cases)) == 4


@pytest.mark.parametrize("name", ["catalog.json", "cases.json", "baseline.json", "stand.json"])
def test_frozen_files_cannot_be_changed_after_labeling(bundle, name):
    path = bundle / name
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="Изменён"):
        load_bundle(bundle)


def test_pending_and_draft_judgments_do_not_become_confirmed_metrics(bundle):
    catalog, cases, rows, _ = load_bundle(bundle)
    rows[0] = rows[0].model_copy(update={"origin": "assistant_draft", "relevance": 3})
    (bundle / "labels.jsonl").write_text(judgments_text(rows), encoding="utf-8")
    report = evaluate_benchmark(bundle)
    assert report["quality_ready"] is False
    assert report["acceptance"]["status"] == "pending"
    assert report["cases"][0]["methods"]["current_rules"]["ndcg_at_k"] is None
    assert report["macro_averages"]["current_rules"]["all"]["all"]["ndcg_at_k"]["cases"] == 0


def confirm_all(bundle, *, relevant=True, violates=False):
    _, _, rows, _ = load_bundle(bundle)
    rows = [
        BenchmarkJudgment.model_validate(
            {
                **row.model_dump(),
                "origin": "human",
                "relevance": 3 if relevant and not violates else 0,
                "violates_required": violates,
                "reviewer": "test-reviewer",
                "reviewed_at": datetime(2026, 10, 26, tzinfo=UTC),
                "evidence": "Test evidence",
                "reasons": ["role"] if not relevant or violates else [],
            }
        )
        for row in rows
    ]
    (bundle / "labels.jsonl").write_text(judgments_text(rows), encoding="utf-8")


def test_human_metrics_include_full_catalog_and_profession_breakdown(bundle):
    confirm_all(bundle)
    report = evaluate_benchmark(bundle)
    assert report["quality_ready"] is True
    methods = report["cases"][0]["methods"]
    assert methods["current_rules"]["recall_at_k"] == 0.5
    assert methods["current_rules"]["retrieval_recall_at_k"]["5"] == 1
    assert report["cases"][0]["experience_group"] == "zero"
    assert {"python", "accountant"} == set(report["macro_averages"]["current_rules"]["profession"])


def test_confirmed_violations_and_empty_positive_catalog_are_visible(bundle):
    confirm_all(bundle, violates=True)
    report = evaluate_benchmark(bundle)
    assert report["acceptance"]["status"] == "failed"
    assert report["cases"][0]["quality_status"] == "no_relevant_vacancies_in_catalog"
    assert report["cases"][0]["methods"]["current_rules"]["violation_rate"] == 1


@pytest.mark.parametrize("change", ["duplicate", "foreign", "wrong-case", "missing"])
def test_labels_reject_corruption(bundle, change):
    _, _, rows, _ = load_bundle(bundle)
    if change == "duplicate":
        rows[-1] = rows[0]
    elif change == "foreign":
        rows[0].pair_id = "0" * 64
    elif change == "wrong-case":
        rows[0].case_id = "other-case"
    else:
        rows.pop()
    (bundle / "labels.jsonl").write_text(judgments_text(rows), encoding="utf-8")
    with pytest.raises(ValueError):
        load_bundle(bundle)


@pytest.mark.parametrize(
    "fields",
    [
        {"origin": "human", "relevance": 3},
        {"relevance": 1},
        {"origin": "assistant_draft", "relevance": True},
        {"origin": "assistant_draft", "relevance": 3, "violates_required": True},
    ],
)
def test_provenance_and_grade_validation(fields):
    with pytest.raises(ValidationError):
        BenchmarkJudgment(case_id="case", vacancy_key="hh:1", pair_id="a" * 64, **fields)


def test_blind_labeling_saves_human_provenance_and_resumes(bundle):
    outputs = []
    answers = iter(["0", "y", "experience", "Required commercial experience, candidate zero."])
    count = label_benchmark(
        bundle,
        case_id="python-zero",
        reviewer="Tester",
        limit=1,
        input_fn=lambda _: next(answers),
        output_fn=outputs.append,
    )
    assert count == 1
    rows = load_bundle(bundle)[2]
    assert sum(row.origin == "human" for row in rows) == 1
    assert not any('"scores"' in value or "baseline_rank" in value for value in outputs)
    before = (bundle / "labels.jsonl").read_bytes()
    label_benchmark(
        bundle,
        case_id="python-zero",
        reviewer="Tester",
        input_fn=lambda _: "q",
        output_fn=lambda _: None,
    )
    assert before == (bundle / "labels.jsonl").read_bytes()


async def test_cli_evaluation_does_not_load_settings_or_working_database(bundle, monkeypatch):
    import jobs.cli as cli

    monkeypatch.setattr(cli, "Settings", lambda: pytest.fail("Working settings accessed"))
    output = bundle / "report.json"
    args = parser().parse_args(
        ["benchmark-evaluate", "--bundle", str(bundle), "--output", str(output)]
    )
    await run_command(args)
    assert json.loads(output.read_text(encoding="utf-8"))["quality_ready"] is False
    with pytest.raises(ValueError, match="существует"):
        await run_command(args)


def test_resource_gate_detects_regression():
    stand = {
        "python": "test",
        "platform": "test",
        "dependencies": {},
        "median_seconds_per_profile": 0.1,
        "peak_python_allocated_bytes": 100,
    }
    criteria = {
        "minimum_validation_professions": 0,
        "minimum_relevant_per_validation_case": 1,
        "maximum_constraint_violations": 0,
        "quality_regression_tolerance": 0,
        "latency_ratio": 2,
        "latency_slack_seconds": 0.05,
        "python_peak_growth_bytes": 100,
    }
    current = {**stand, "median_seconds_per_profile": 0.4}
    assert acceptance([], current, stand, criteria)["status"] == "failed"


def test_blind_labeling_detects_concurrent_editor(bundle):
    path = bundle / "labels.jsonl"
    answers = iter(["0", "y", "experience", "Required experience"])

    def change_file(prompt):
        answer = next(answers)
        if "обоснование" in prompt:
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        return answer

    with pytest.raises(ValueError, match="другом окне"):
        label_benchmark(
            bundle,
            case_id="python-zero",
            reviewer="Tester",
            limit=1,
            input_fn=change_file,
            output_fn=lambda _: None,
        )
