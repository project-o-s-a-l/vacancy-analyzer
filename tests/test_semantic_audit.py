import json
from datetime import timedelta

import pytest
from test_benchmark import specification
from test_candidates import job

from domain.benchmark import BenchmarkCatalog
from jobs import semantic_audit
from jobs.benchmark_data import blank_judgments, judgments_text
from jobs.cli import parser, run_command


def stand(monkeypatch, tmp_path, now, cards, grades, *, origin="assistant_draft"):
    cases = specification()
    catalog = BenchmarkCatalog(
        reference_time=now,
        origin="saved_hh_html",
        sources=[],
        skipped=[],
        duplicates=[],
        vacancies=cards,
    )
    labels = [
        row.model_copy(
            update={
                "relevance": grades.get((row.case_id, row.vacancy_key), 0),
                "origin": origin,
            }
        )
        for row in blank_judgments(catalog, cases)
    ]
    (tmp_path / "labels.jsonl").write_text(judgments_text(labels), encoding="utf-8")
    monkeypatch.setattr(semantic_audit, "load_bundle", lambda path: (catalog, cases, labels, {}))
    return labels


def key(card):
    return f"{card.source}:{card.source_id}"


def cards(vacancy, now):
    high = job(
        vacancy,
        title="Python developer",
        skills=["python"],
        description="Знание Python.",
        experience="no_experience",
        published_at=now - timedelta(days=1),
    )
    low = job(
        vacancy,
        title="Engineer",
        skills=[],
        description="Automation of network routines.",
        experience="no_experience",
        published_at=now - timedelta(days=2),
    )
    return high, low


async def test_missing_wording_is_review_candidate_not_claimed_model_gain(
    monkeypatch, tmp_path, vacancy, now
):
    high, low = cards(vacancy, now)
    stand(monkeypatch, tmp_path, now, [high, low], {("python-zero", key(low)): 3})
    report = await semantic_audit.audit_semantic_need(tmp_path, k=1)
    assert report["all"]["eligible_recall"] == 0
    assert report["decision"] == "review_retrieval_losses"
    assert report["retrieval_misses_for_review"][0]["stage"] == "retrieval_no_overlap"
    assert report["runtime_changed"] is False and report["external_calls"] == 0
    assert "labels_not_independently_confirmed" in report["reasons"]


async def test_overlap_loss_is_distinguished_from_zero_overlap(monkeypatch, tmp_path, vacancy, now):
    high, low = cards(vacancy, now)
    low.description = "Using Python to automate tasks"
    stand(monkeypatch, tmp_path, now, [high, low], {("python-zero", key(low)): 3})
    report = await semantic_audit.audit_semantic_need(tmp_path, k=1)
    assert report["retrieval_misses_for_review"][0]["stage"] == "retrieval_with_overlap"


async def test_low_score_is_not_counted_as_retrieval_loss(monkeypatch, tmp_path, vacancy, now):
    high, _ = cards(vacancy, now)
    stand(monkeypatch, tmp_path, now, [high], {("python-zero", key(high)): 3})
    report = await semantic_audit.audit_semantic_need(tmp_path, min_score=100)
    assert report["all"]["eligible_recall"] == 1
    assert report["all"]["positive_stages"] == {"score_threshold": 1}
    assert report["retrieval_misses_for_review"] == []
    assert report["decision"] == "defer_semantic_runtime"


@pytest.mark.parametrize("change", ["closed", "experience"])
async def test_conditions_are_not_bypassed_by_positive_labels(
    monkeypatch, tmp_path, vacancy, now, change
):
    high, _ = cards(vacancy, now)
    if change == "closed":
        high.archived = True
    else:
        high.experience = "6+"
        high.description = "Требуется коммерческий опыт работы от 6 лет. Python."
    stand(monkeypatch, tmp_path, now, [high], {("python-zero", key(high)): 3})
    report = await semantic_audit.audit_semantic_need(tmp_path)
    assert report["all"]["eligible_positive_pairs"] == 0
    assert report["all"]["eligible_recall"] is None
    assert report["positive_pairs"][0]["stage"] == (
        "inactive" if change == "closed" else "required_conditions"
    )
    assert report["retrieval_misses_for_review"] == []


async def test_duplicate_equivalent_does_not_create_semantic_loss(
    monkeypatch, tmp_path, vacancy, now
):
    high, _ = cards(vacancy, now)
    duplicate = job(
        high,
        title=high.title,
        skills=high.skills,
        description=high.description,
        experience=high.experience,
        published_at=high.published_at - timedelta(days=1),
    )
    stand(monkeypatch, tmp_path, now, [high, duplicate], {("python-zero", key(duplicate)): 3})
    report = await semantic_audit.audit_semantic_need(tmp_path, k=1)
    assert report["all"]["eligible_recall"] == 1
    assert report["positive_pairs"][0]["stage"] == "duplicate_equivalent"
    assert report["retrieval_misses_for_review"] == []


async def test_unknown_labels_are_not_treated_as_negative(monkeypatch, tmp_path, vacancy, now):
    high, _ = cards(vacancy, now)
    labels = stand(monkeypatch, tmp_path, now, [high], {})
    labels[0].relevance = None
    with pytest.raises(ValueError, match="полная явная разметка"):
        await semantic_audit.audit_semantic_need(tmp_path)


async def test_command_never_creates_settings_or_database(monkeypatch, tmp_path, vacancy, now):
    from jobs import cli

    high, _ = cards(vacancy, now)
    stand(monkeypatch, tmp_path, now, [high], {("python-zero", key(high)): 3})

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline audit must not read environment or contact services")

    monkeypatch.setattr(cli, "Settings", forbidden)
    monkeypatch.setattr(cli, "create_engine", forbidden)
    output = tmp_path / "out/report.json"
    args = parser().parse_args(
        ["semantic-audit", "--bundle", str(tmp_path), "--output", str(output)]
    )
    await run_command(args)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["runtime_changed"] is False and output.with_suffix(".md").exists()
    assert "resume" not in result and "profile" not in result
    with pytest.raises(ValueError, match="новые"):
        await run_command(args)


@pytest.mark.parametrize("score", [-1, 101, float("nan")])
async def test_invalid_threshold_rejected(tmp_path, score):
    with pytest.raises(ValueError, match="Порог"):
        await semantic_audit.audit_semantic_need(tmp_path, min_score=score)


@pytest.mark.parametrize("suffix", [".json", ".md"])
def test_writer_never_overwrites_report_or_companion(tmp_path, suffix):
    output = tmp_path / "result.json"
    output.with_suffix(suffix).write_text("Keep me", encoding="utf-8")
    with pytest.raises(ValueError, match="новые"):
        semantic_audit.write_semantic_audit({}, output)
    assert output.with_suffix(suffix).read_text(encoding="utf-8") == "Keep me"
