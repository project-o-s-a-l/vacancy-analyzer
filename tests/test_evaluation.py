import copy
import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest

from domain.evaluation import SPLITS, EvaluationData
from jobs.evaluation import evaluate, save_report
from jobs.evaluation_compare import compare, read_report
from jobs.evaluation_data import load_data, prepare, synthetic_data, tasks, write_new
from jobs.evaluation_labels import export_review, import_review, resolve_labels
from scoring.evaluation_metrics import catalog_metrics, fact_metrics, interval

PROTOCOL = Path(__file__).resolve().parents[1] / "config/evaluation/protocol.json"


@pytest.fixture(scope="module")
def dataset():
    return synthetic_data(20261102)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory, dataset):
    output = tmp_path_factory.mktemp("evaluation") / "bundle"
    prepare(output, PROTOCOL)
    data = dataset.model_dump(mode="json")
    data["profiles"] = [p for p in data["profiles"] if p["profession"] == "python"]
    data["vacancies"] = [v for v in data["vacancies"] if "-python-" in v["id"]]
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["bootstrap_repeats"] = 100
    (output / "data.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    (output / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest["sha256"] = {
        name: sha256((output / name).read_bytes()).hexdigest()
        for name in ("data.json", "protocol.json")
    }
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return output


def test_dataset_has_breadth_and_sealed_splits(dataset):
    assert len(dataset.profiles) == 64 and len(dataset.vacancies) == 192
    assert len({p.profession for p in dataset.profiles}) == 8
    assert {p.origin for p in dataset.profiles} == {"synthetic"}
    assert max(len(p.text) for p in dataset.profiles) > 6000
    assert {"long-over-6000", "incomplete", "career-change", "non-it"} <= {
        s for p in dataset.profiles for s in p.scenarios
    }
    assert all(len(tasks(dataset, split)) == 832 for split in SPLITS)


@pytest.mark.parametrize(
    "leak", ["owner", "profile", "text", "vacancy_group", "vacancy_copy", "time"]
)
def test_split_and_temporal_leakage(dataset, leak):
    data = dataset.model_dump()
    p1, p2 = data["profiles"][0], data["profiles"][16]
    v1, v2 = data["vacancies"][0], data["vacancies"][48]
    if leak in {"owner", "profile", "text"}:
        p2[leak] = copy.deepcopy(p1[leak])
        if leak == "text":
            p2["draft_facts"] = p1["draft_facts"]
    elif leak == "vacancy_group":
        v2["duplicate_group"] = v1["duplicate_group"]
    elif leak == "vacancy_copy":
        for key in ("title", "company", "description", "experience", "remote", "skills"):
            v2["vacancy"][key] = v1["vacancy"][key]
        v2["text"], v2["draft_facts"] = v1["text"], v1["draft_facts"]
    else:
        v2["vacancy"]["published_at"] = v1["vacancy"]["published_at"]
    with pytest.raises(ValueError, match="leakage"):
        EvaluationData.model_validate(data)


def test_full_catalog_denominators_and_partial_pool():
    result = catalog_metrics(["a"], {"a": 3, "b": 2, "c": 0}, ["a", "b", "c"])
    assert result["precision_at_5"] == 0.2
    assert result["precision_among_shown"] == 1
    assert result["recall"] == 0.5
    assert result["precision_denominator"] == 5 and result["recall_denominator"] == 2
    assert 0 < result["ndcg_at_5"] < 1
    empty = catalog_metrics([], {"a": 3}, ["a"])
    assert empty["precision_at_5"] == 0 and empty["ndcg_at_5"] == 0
    partial = catalog_metrics(["a", "c"], {"a": 3}, ["a", "b", "c"])
    assert partial["recall"] is None and partial["ndcg_at_5"] is None
    assert partial["pooled_recall"] == 1 and partial["unjudged_groups"] == 2
    assert partial["recall_lower_bound"] == 0.5
    assert partial["precision_at_5_upper_bound"] == 0.4
    assert partial["metric_scope"] == "partial_pool_not_catalog_recall"


def test_uncertainty_is_seeded_and_does_not_claim_tiny_sample():
    assert interval([0, 1, -1], repeats=100) == interval([0, 1, -1], repeats=100)
    assert interval([1])["low"] is None
    assert interval([0, 1, -1])["low"] < 0


def test_fact_denominator_includes_unsupported_fields_and_unknown_is_separate():
    def fact(field, value, status="known"):
        return {
            "field": field,
            "value": value,
            "status": status,
            "importance": "unspecified",
            "alternative_group": None,
        }

    metrics = fact_metrics(
        [fact("skill", "Python")],
        [fact("skill", "python"), fact("task", "API"), fact("experience_total", None, "unknown")],
    )
    assert metrics["expected"] == 2 and metrics["fact_recall"] == 0.5
    assert metrics["unknown_gold"] == 1


def submitted(bundle, tmp_path, reviewer, *, grade=2, role="reviewer", resolves=(), fact=False):
    output = tmp_path / reviewer
    export_review(bundle, output, split="development", reviewer=reviewer, role=role)
    data, _, _ = load_data(bundle)
    cards = tasks(data, "development")
    key = next(k for k in sorted(cards) if k.startswith("facts:" if fact else "pair:"))
    submission = json.loads((output / "review.json").read_text(encoding="utf-8"))
    payload = (
        {
            "complete": True,
            "facts": [],
            "unknowns": [],
            "rationale": "Human reviewed the entire source document",
        }
        if fact
        else {
            "relevance": grade,
            "violates_required": False,
            "reasons": ["role"],
            "unknowns": [],
            "evidence": [{"source": "profile_text", "quote": cards[key]["profile_text"][:30]}],
            "rationale": "Human reviewed role and mandatory conditions",
        }
    )
    submission.update(
        {
            "reviewed_at": datetime.now(UTC).isoformat(),
            "attestation": "I personally reviewed the frozen sources; no automatic labels",
            "records": [{"task_id": key, "payload": payload, "resolves": list(resolves)}],
        }
    )
    path = output / "submitted.json"
    write_new(path, submission)
    return output / "assignment.json", path, key


@pytest.mark.parametrize("tamper", ["origin", "reviewer", "assignment", "evidence", "attestation"])
def test_provenance_spoofing_rejected(bundle, tmp_path, tamper):
    assignment, path, _ = submitted(bundle, tmp_path, "reviewer-1")
    value = json.loads(path.read_text(encoding="utf-8"))
    if tamper == "origin":
        value["records"][0]["payload"]["origin"] = "human"
    elif tamper == "reviewer":
        value["reviewer"] = "another-person"
    elif tamper == "assignment":
        value["assignment_hash"] = "0" * 64
    elif tamper == "evidence":
        value["records"][0]["payload"]["evidence"][0]["quote"] = "invented evidence"
    else:
        value["attestation"] = "assistant_draft"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        import_review(bundle, assignment, path, tmp_path / "ledger", attest_human=True)
    assert not list((tmp_path / "ledger").glob("*.json"))


def test_human_import_consensus_and_overwrite_guard(bundle, tmp_path):
    assignment, path, key = submitted(bundle, tmp_path, "reviewer-1")
    ledger = tmp_path / "ledger"
    with pytest.raises(ValueError, match="attest-human"):
        import_review(bundle, assignment, path, ledger)
    import_review(bundle, assignment, path, ledger, attest_human=True)
    _, _, bundle_hash = load_data(bundle)
    assert resolve_labels(ledger, bundle_hash)[0] == {}
    assert import_review(bundle, assignment, path, ledger, attest_human=True)["imported"] == 0
    original = path.read_text(encoding="utf-8")
    changed = json.loads(original)
    changed["records"][0]["payload"]["relevance"] = 0
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="overwritten"):
        import_review(bundle, assignment, path, ledger, attest_human=True)
    path.write_text(original, encoding="utf-8")
    assert not (ledger / ".import.lock").exists()
    assignment2, path2, _ = submitted(bundle, tmp_path, "reviewer-2")
    import_review(bundle, assignment2, path2, ledger, attest_human=True)
    assert resolve_labels(ledger, bundle_hash)[0][key]["relevance"] == 2


def test_disagreement_requires_independent_evidence_linked_adjudication(bundle, tmp_path):
    ledger = tmp_path / "ledger"
    refs = []
    for reviewer, grade in (("reviewer-1", 1), ("reviewer-2", 2)):
        assignment, path, key = submitted(bundle, tmp_path, reviewer, grade=grade)
        refs.append(import_review(bundle, assignment, path, ledger, attest_human=True)["sha256"])
    _, _, bundle_hash = load_data(bundle)
    assert resolve_labels(ledger, bundle_hash)[1]["disagreements"] == [key]
    assignment, path, _ = submitted(
        bundle, tmp_path, "arbiter", grade=1, role="adjudicator", resolves=refs
    )
    import_review(bundle, assignment, path, ledger, attest_human=True)
    assert resolve_labels(ledger, bundle_hash)[0][key]["relevance"] == 1


def test_fact_source_spans_are_checked(bundle, tmp_path):
    assignment, path, _ = submitted(bundle, tmp_path, "reviewer-1", fact=True)
    value = json.loads(path.read_text(encoding="utf-8"))
    data, _, _ = load_data(bundle)
    fact = data.profiles[2].draft_facts[0].model_dump()
    fact["quote"] = "fake quote"
    value["records"][0]["payload"]["facts"] = [fact]
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="span"):
        import_review(bundle, assignment, path, tmp_path / "ledger", attest_human=True)


@pytest.fixture(scope="module")
def report(bundle):
    import asyncio

    return asyncio.run(evaluate(bundle, repeats=2))


async def test_frozen_evaluation_replays_and_draft_does_not_become_human(bundle, report):
    repeated = await evaluate(bundle, repeats=2)
    assert repeated["deterministic_result_sha256"] == report["deterministic_result_sha256"]
    assert report["quality"] == "waiting_external"
    assert all(row["display"]["precision_at_5"] is None for row in report["cases"])
    assert all(row["display"]["precision_denominator"] == 5 for row in report["cases"])
    assert all(f["human"] is None for f in report["facts"])
    assert all(f["assistant_draft_diagnostic"] is not None for f in report["facts"])
    assert all(not row["safety"]["stale"]["numerator"] for row in report["cases"])


async def test_final_test_is_not_used_by_default(bundle):
    with pytest.raises(ValueError, match="Final test"):
        await evaluate(bundle, split="test", repeats=2)


async def test_resume_offline_adapter_matches_actual_prepare_confirm_service(dataset, sessions):
    from jobs.evaluation import extract_facts
    from jobs.users import UserService

    users = UserService(sessions)
    await users.register(52)
    row = next(p for p in dataset.profiles if p.split == "development" and p.profession == "python")
    draft = await users.resumes.prepare(52, row.text)
    user = await users.resumes.confirm(52, draft.token)
    from jobs.evaluation import fact_atoms
    from nlp.professional_facts import profile_payload

    offline = extract_facts(row, owner_id=52)[1]
    assert profile_payload(offline) == profile_payload(user.profile)
    assert fact_atoms(offline.professional_facts) == fact_atoms(user.profile.professional_facts)
    assert offline.professional_facts.owner_id == user.profile.professional_facts.owner_id


async def test_final_release_slots_cannot_be_reused(bundle, report, tmp_path, monkeypatch):
    from jobs.cli import parser, run_command

    async def fake_evaluate(*args, **kwargs):
        return copy.deepcopy(report)

    monkeypatch.setattr("jobs.evaluation.evaluate", fake_evaluate)
    # Only a private unit fixture is marked; the actual saved test is never evaluated.
    args = parser().parse_args(
        [
            "evaluation-run",
            "--bundle",
            str(bundle),
            "--split",
            "test",
            "--final-test",
            "--release-id",
            "unit-test-only",
            "--test-role",
            "reference",
            "--output",
            str(tmp_path / "final-a"),
        ]
    )
    await run_command(args)
    args.output = tmp_path / "final-b"
    with pytest.raises(FileExistsError):
        await run_command(args)


def test_comparison_cannot_accept_missing_labels_or_changed_population(report, tmp_path):
    result = compare(report, report)
    assert result["status"] == "inconclusive"
    assert any("human" in s for s in result["pending"])
    changed = copy.deepcopy(report)
    changed["cases"].pop()
    with pytest.raises(ValueError, match="population"):
        compare(report, changed)
    changed = copy.deepcopy(report)
    changed["provenance"]["imports"] = ["0" * 64]
    with pytest.raises(ValueError, match="provenance"):
        compare(report, changed)
    output = tmp_path / "run"
    save_report(report, output)
    assert read_report(output)["bundle_hash"] == report["bundle_hash"]
    with pytest.raises(FileExistsError):
        save_report(report, output)
    (output / "report.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        read_report(output)


def human_fixture_report(report):
    """Unit-test values only; never writes these invented labels to the human ledger."""
    value = copy.deepcopy(report)
    value["quality"] = "ready_for_review"
    for row in value["cases"]:
        row["display"].update({"ndcg_at_5": 0.5, "precision_at_5": 0.4})
        row["retrieval_recall_at_k"]["64"]["recall"] = 0.5
        row["safety"]["mandatory_unjudged_shown"] = 0
    for row in value["facts"]:
        row["human"] = {"fact_recall": 0.5, "fact_precision": 0.5}
    return value


def test_paired_gate_detects_quality_coverage_and_fact_precision_regressions(report):
    old = human_fixture_report(report)
    assert compare(old, old)["status"] == "passed"
    assert compare(old, old, claim_improvement=True)["status"] == "inconclusive"
    changed = copy.deepcopy(old)
    for row in changed["cases"]:
        row["display"]["precision_at_5"] = 0
    assert compare(old, changed)["status"] == "failed"
    changed = copy.deepcopy(old)
    for row in changed["facts"]:
        row["human"]["fact_precision"] = 0.1
    assert any("fact_precision" in reason for reason in compare(old, changed)["failures"])
    changed = copy.deepcopy(old)
    changed["cases"][0]["safety"]["mandatory"]["numerator"] = 1
    assert compare(old, changed)["status"] == "failed"


async def test_public_cli_import_and_compare_commands(bundle, tmp_path, report, capsys):
    from jobs.cli import parser, run_command

    assignment, path, _ = submitted(bundle, tmp_path, "cli-reviewer")
    args = parser().parse_args(
        [
            "evaluation-import",
            "--bundle",
            str(bundle),
            "--assignment",
            str(assignment),
            "--submission",
            str(path),
            "--ledger",
            str(tmp_path / "ledger"),
            "--attest-human",
        ]
    )
    await run_command(args)
    assert '"imported": 1' in capsys.readouterr().out
    save_report(report, tmp_path / "a")
    save_report(report, tmp_path / "b")
    args = parser().parse_args(
        [
            "evaluation-compare",
            "--reference",
            str(tmp_path / "a"),
            "--candidate",
            str(tmp_path / "b"),
            "--output",
            str(tmp_path / "c.json"),
        ]
    )
    await run_command(args)
    assert json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))["status"] == "inconclusive"


def test_review_html_is_self_contained_and_blind(bundle, tmp_path):
    export_review(bundle, tmp_path / "review", split="development", reviewer="reviewer-1")
    text = (tmp_path / "review/cards.html").read_text(encoding="utf-8")
    assert "__F01_PACKAGE__" not in text
    assert "draft_facts" not in text and "assistant_draft" not in text
    assert "Скачать завершённые оценки JSON" in text
    assert "fetch(" not in text and "https://cdn" not in text


@pytest.mark.parametrize("partial_human", [False, True])
async def test_legacy_retrieval_drafts_cannot_choose_k(
    dataset, monkeypatch, partial_human, tmp_path
):
    from types import SimpleNamespace

    from jobs.retrieval_evaluation import evaluate_retrieval

    profiles = [p for p in dataset.profiles if p.split == "development"][:2]
    vacancies = [v.vacancy for v in dataset.vacancies if v.split == "development"][:6]
    catalog = SimpleNamespace(
        reference_time=dataset.reference_times["development"], vacancies=vacancies
    )
    cases = SimpleNamespace(
        cases=[
            SimpleNamespace(id=p.id, profession=p.profession, split=split, profile=p.profile)
            for p, split in zip(profiles, ("training", "validation"), strict=True)
        ]
    )
    labels = [
        SimpleNamespace(
            case_id=p.id,
            vacancy_key=f"{v.source}:{v.source_id}",
            relevance=3,
            origin="assistant_draft",
        )
        for p in profiles
        for v in vacancies
    ]
    if partial_human:
        labels[0].origin = "human"
    monkeypatch.setattr(
        "jobs.retrieval_evaluation.load_bundle", lambda _: (catalog, cases, labels, {})
    )
    (tmp_path / "labels.jsonl").write_text("fixture-only", encoding="utf-8")
    result = await evaluate_retrieval(tmp_path, ks=(5,))
    assert result["chosen_k_on_training"] is None and not result["human_quality_ready"]
    assert result["experiments"][0]["training"]["eligible_recall"] is None
    assert all(row["recall"] is None for row in result["experiments"][0]["cases"])


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def evaluation_sessions(request):
    name = "postgres_sessions" if request.param == "postgres" else "sessions"
    return request.getfixturevalue(name)


async def test_offline_sql_runtime_and_notification_agree(bundle, evaluation_sessions, monkeypatch):
    from jobs.admission import admit
    from jobs.notify import NotificationService
    from jobs.rank import rank_recent
    from jobs.retrieval_evaluation import MemoryCandidateSource
    from storage.repositories import VacancyRepository

    sessions = evaluation_sessions
    data, _, _ = load_data(bundle)
    now = data.reference_times["development"]
    case = next(p for p in data.profiles if p.split == "development")
    vacancies = [v.vacancy for v in data.vacancies if v.split == "development"]
    async with sessions.begin() as session:
        for vacancy in vacancies:
            await VacancyRepository(session).save(vacancy)
    offline_trace, runtime_trace = {}, {}
    offline = await rank_recent(
        None,
        case.profile,
        now=now,
        candidate_source=MemoryCandidateSource(vacancies),
        trace=offline_trace,
    )
    async with sessions.begin() as session:
        runtime = await rank_recent(
            VacancyRepository(session), case.profile, now=now, trace=runtime_trace
        )
    for stage in ("retrieved", "filtered", "ranked"):
        assert offline_trace[stage] == runtime_trace[stage]
    assert [(v.id, m.score) for v, m in offline] == [(v.id, m.score) for v, m in runtime]

    # Freeze only the wall clock, not rank_recent or admission implementation.
    class FrozenDate(datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    monkeypatch.setattr("jobs.rank.datetime", FrozenDate)
    service = NotificationService(
        sessions, case.profile, chat_id=123, saved_selections_enabled=False
    )
    displayed = await service.pending()
    assert [v.id for v, _ in displayed] == [v.id for v, _ in admit(offline)[2]]
