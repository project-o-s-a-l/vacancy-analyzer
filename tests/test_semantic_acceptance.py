from copy import deepcopy

import pytest
from test_candidates import candidate, job
from test_semantic_retrieval import vector

from jobs.hybrid_candidates import novelty_fusion
from jobs.semantic_acceptance import acceptance, current_code_hashes
from jobs.semantic_experiment import _query_cache
from jobs.semantic_model import MODEL_ID
from jobs.semantic_text import TEXT_SPEC


def approved_evidence():
    rows = []
    for n in range(5):
        base = {"positive_recall": 0.6, "serving": {"ndcg_at_k": 0.5}, "eligibility_violations": 0}
        variant = {
            "positive_recall": 0.8,
            "serving": {"ndcg_at_k": 0.7},
            "eligibility_violations": 0,
        }
        rows.append(
            {
                "profession": f"synthetic-{n}",
                "split": "validation",
                "positive_pairs": 2,
                "label_origins": ["human"],
                "channels": {
                    "lexical": deepcopy(base),
                    "rrf": deepcopy(variant),
                    "novelty": deepcopy(variant),
                },
            }
        )
    return {
        "version": "semantic-experiment-v1",
        "model_id": MODEL_ID,
        "text_spec": TEXT_SPEC,
        "experiment_code_sha256": current_code_hashes(),
        "metrics": rows,
        "independent_validation_confirmed": True,
        "query_measurement": {"peak_rss_bytes": 500_000_000, "wall_seconds": 0.1},
        "search_measurements": [
            {"lexical_seconds": 0.4, "hybrid_seconds": {"rrf": 0.2, "novelty": 0.2}}
        ],
    }


@pytest.mark.parametrize("channel", ["rrf", "novelty"])
def test_gate_only_passes_complete_measured_synthetic_evidence(channel):
    result = acceptance(approved_evidence(), channel=channel)
    assert result["allowed"] and result["reasons"] == []
    assert result["working_runtime_changed"] is False


@pytest.mark.parametrize(
    "failure,reason",
    [
        ("draft", "validation_labels_not_human"),
        ("independence", "independent_validation_not_confirmed"),
        ("coverage", "positive_profession_coverage_insufficient"),
        ("ndcg", "serving_quality_gain_insufficient"),
        ("recall", "retrieval_gain_insufficient"),
        ("regression", "profession_quality_regression"),
        ("recall_regression", "profession_recall_regression"),
        ("violations", "required_conditions_not_verified"),
        ("memory", "worker_memory_budget_not_verified"),
        ("latency", "cold_request_latency_budget_exceeded"),
        ("no_latency", "request_latency_not_measured"),
        ("nan", "quality_metrics_incomplete"),
    ],
)
def test_gate_rejects_missing_or_failed_acceptance(failure, reason):
    proof = approved_evidence()
    row = proof["metrics"][0]
    variant = row["channels"]["novelty"]
    if failure == "draft":
        row["label_origins"] = ["assistant_draft"]
    if failure == "independence":
        proof["independent_validation_confirmed"] = False
    if failure == "coverage":
        row["positive_pairs"] = 0
    if failure == "ndcg":
        for r in proof["metrics"]:
            r["channels"]["novelty"]["serving"]["ndcg_at_k"] = 0.5
    if failure == "recall":
        for r in proof["metrics"]:
            r["channels"]["novelty"]["positive_recall"] = 0.6
    if failure == "regression":
        variant["serving"]["ndcg_at_k"] = 0.4
    if failure == "recall_regression":
        variant["positive_recall"] = 0.5
    if failure == "violations":
        variant["eligibility_violations"] = 1
    if failure == "memory":
        proof["query_measurement"]["peak_rss_bytes"] = 2 * 1024**3
    if failure == "latency":
        proof["query_measurement"]["wall_seconds"] = 2.0
    if failure == "no_latency":
        proof["search_measurements"] = []
    if failure == "nan":
        variant["serving"]["ndcg_at_k"] = float("nan")
    result = acceptance(proof, channel="novelty")
    assert not result["allowed"] and reason in result["reasons"]


@pytest.mark.parametrize("malformed", [None, {}, {"metrics": [None]}, {"metrics": "bad"}])
def test_malformed_report_cannot_allow_activation(malformed):
    assert acceptance(malformed, channel="rrf")["allowed"] is False


def test_novelty_budget_keeps_new_semantic_items_without_expanding_k(vacancy):
    lexical = [job(vacancy, title=f"Lexical {n}") for n in range(8)]
    new = [job(vacancy, title=f"Semantic {n}") for n in range(5)]
    result = novelty_fusion(lexical, [*lexical[:3], *new], limit=8)
    assert len(result) == 8
    assert [v.id for v in result[:6]] == [v.id for v in lexical[:6]]
    assert [v.id for v in result[6:]] == [v.id for v in new[:2]]


def test_novelty_does_not_waste_slots_when_no_new_candidates(vacancy):
    lexical = [job(vacancy, title=f"Job {n}") for n in range(8)]
    assert novelty_fusion(lexical, lexical, limit=8) == lexical
    duplicate = job(lexical[0], title=lexical[0].title, description=lexical[0].description)
    assert novelty_fusion(lexical, [duplicate], limit=8) == lexical


def test_query_cache_reuses_only_matching_profile_and_detects_corruption(tmp_path):
    calls = []

    def encode(tasks):
        calls.append(len(tasks))
        return {"values": [{"vector": vector()} for _ in tasks]}

    path = tmp_path / "queries.json"
    profiles = {"fictitious": candidate()}
    _query_cache(profiles, path, encode)
    _, resources = _query_cache(profiles, path, encode)
    assert calls == [1, 0] and resources["reused"] == 1
    _query_cache({"fictitious": candidate(experience_years=5)}, path, encode)
    assert calls[-1] == 1
    path.write_text('{"sha256":"bad"}', encoding="utf-8")
    with pytest.raises(ValueError, match="повреждённый"):
        _query_cache(profiles, path, encode)


def test_zero_relevance_query_has_undefined_ndcg_but_still_checks_false_positives():
    proof = approved_evidence()
    row = deepcopy(proof["metrics"][0])
    row.update(maximum_relevance=0, positive_pairs=0)
    for values in row["channels"].values():
        values["positive_recall"] = None
        values["serving"] = {"ndcg_at_k": None, "shown": 0}
    proof["metrics"].append(row)
    assert acceptance(proof, channel="novelty")["allowed"]
    row["channels"]["novelty"]["serving"]["shown"] = 1
    assert (
        "negative_query_false_positive_regression"
        in acceptance(proof, channel="novelty")["reasons"]
    )


@pytest.mark.parametrize("change", ["code", "model"])
def test_old_code_or_other_model_cannot_reuse_quality_proof(change):
    proof = approved_evidence()
    if change == "code":
        proof["experiment_code_sha256"] = {}
    else:
        proof["model_id"] = "other"
    assert not acceptance(proof, channel="rrf")["allowed"]


async def test_readonly_check_rejects_failed_proof_without_settings_or_models(
    tmp_path, monkeypatch, capsys
):
    import json

    from jobs import cli

    proof = approved_evidence()
    proof["independent_validation_confirmed"] = False
    path = tmp_path / "report.json"
    path.write_text(json.dumps(proof), encoding="utf-8")

    def forbidden(*args, **kwargs):
        pytest.fail("must not connect or run model")

    monkeypatch.setattr(cli, "Settings", forbidden)
    monkeypatch.setattr(cli, "create_engine", forbidden)
    args = cli.parser().parse_args(
        ["semantic-check", "--report", str(path), "--channel", "novelty"]
    )
    with pytest.raises(SystemExit) as failure:
        await cli.run_command(args)
    assert failure.value.code == 2
    assert '"allowed": false' in capsys.readouterr().out
