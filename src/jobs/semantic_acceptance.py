"""Measured deployment decision. Missing or failed evidence never enables a retriever."""

import hashlib
from math import isfinite
from pathlib import Path

from jobs.semantic_model import MODEL_ID
from jobs.semantic_text import TEXT_SPEC
from scoring.serving_features import feature_contract

CRITERIA = {
    "minimum_recall_gain": 0.02,
    "minimum_serving_ndcg_gain": 0.01,
    "minimum_positive_validation_professions": 5,
    "latency_ratio": 2.0,
    "latency_slack_seconds": 0.05,
    "maximum_worker_rss_bytes": 1024**3,
}


def current_code_hashes():
    root = Path(__file__).parent
    names = [
        "hybrid_candidates.py",
        "candidates.py",
        "retrieval_evaluation.py",
        *(p.name for p in root.glob("semantic_*.py")),
    ]
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sorted(names)
    } | {"feature_contract": feature_contract()}


def acceptance(report, *, channel):
    if channel not in {"rrf", "novelty"}:
        raise ValueError("Неизвестный вариант смыслового отбора")
    try:
        return _acceptance(report, channel=channel)
    except (TypeError, ValueError, AttributeError, KeyError, OverflowError):
        return {
            "channel": channel,
            "criteria": CRITERIA.copy(),
            "allowed": False,
            "reasons": ["acceptance_evidence_invalid"],
            "working_runtime_changed": False,
        }


def _acceptance(report, *, channel):
    if channel not in {"rrf", "novelty"}:
        raise ValueError("Неизвестный вариант смыслового отбора")
    reasons = []
    if (
        report.get("model_id") != MODEL_ID
        or report.get("text_spec") != TEXT_SPEC
        or report.get("version") != "semantic-experiment-v1"
    ):
        reasons.append("semantic_contract_mismatch")
    if report.get("experiment_code_sha256") != current_code_hashes():
        reasons.append("evaluation_code_changed_or_missing")

    def metric(value):
        return type(value) in {int, float} and isfinite(value) and 0 <= value <= 1

    validation = [row for row in report.get("metrics", []) if row.get("split") == "validation"]
    if not validation:
        reasons.append("validation_missing")
    if not report.get("independent_validation_confirmed"):
        reasons.append("independent_validation_not_confirmed")
    if not validation or any(row.get("label_origins") != ["human"] for row in validation):
        reasons.append("validation_labels_not_human")
    professions = {row["profession"] for row in validation if row.get("positive_pairs", 0) > 0}
    if len(professions) < CRITERIA["minimum_positive_validation_professions"]:
        reasons.append("positive_profession_coverage_insufficient")
    before = []
    after = []
    recall_before = []
    recall_after = []
    for row in validation:
        base = row.get("channels", {}).get("lexical", {})
        variant = row.get("channels", {}).get(channel, {})
        b = base.get("serving", {}).get("ndcg_at_k")
        m = variant.get("serving", {}).get("ndcg_at_k")
        if row.get("maximum_relevance", 3) == 0 and b is None and m is None:
            baseline_shown = base.get("serving", {}).get("shown")
            variant_shown = variant.get("serving", {}).get("shown")
            if type(baseline_shown) is not int or type(variant_shown) is not int:
                reasons.append("negative_query_metrics_missing")
            elif variant_shown > baseline_shown:
                reasons.append("negative_query_false_positive_regression")
        elif not metric(b) or not metric(m):
            reasons.append("quality_metrics_incomplete")
        else:
            before.append(b)
            after.append(m)
        rb, ra = base.get("positive_recall"), variant.get("positive_recall")
        if row.get("positive_pairs", 0) > 0:
            if not metric(rb) or not metric(ra):
                reasons.append("recall_metrics_incomplete")
            else:
                recall_before.append(rb)
                recall_after.append(ra)
        violations = variant.get("eligibility_violations")
        if type(violations) is not int or violations != 0:
            reasons.append("required_conditions_not_verified")
    if (
        not before
        or sum(after) / len(after) - sum(before) / len(before)
        < CRITERIA["minimum_serving_ndcg_gain"] - 1e-12
    ):
        reasons.append("serving_quality_gain_insufficient")
    if (
        not recall_before
        or sum(recall_after) / len(recall_after) - sum(recall_before) / len(recall_before)
        < CRITERIA["minimum_recall_gain"] - 1e-12
    ):
        reasons.append("retrieval_gain_insufficient")
    # Check profession means, including groups with only partial relevance grades.
    for name in {row["profession"] for row in validation}:
        rows = [row for row in validation if row["profession"] == name]
        pairs = []
        for row in rows:
            try:
                b = row["channels"]["lexical"]["serving"]["ndcg_at_k"]
                m = row["channels"][channel]["serving"]["ndcg_at_k"]
                if metric(b) and metric(m):
                    pairs.append((b, m))
            except KeyError:
                pass
        if pairs and sum(m - b for b, m in pairs) < -1e-12:
            reasons.append("profession_quality_regression")
        recalled = []
        for row in rows:
            b = row.get("channels", {}).get("lexical", {}).get("positive_recall")
            m = row.get("channels", {}).get(channel, {}).get("positive_recall")
            if metric(b) and metric(m):
                recalled.append((b, m))
        if recalled and sum(m - b for b, m in recalled) < -1e-12:
            reasons.append("profession_recall_regression")
    rss = report.get("query_measurement", {}).get("peak_rss_bytes")
    latency = report.get("query_measurement", {}).get("wall_seconds")
    timings = report.get("search_measurements", [])
    if (
        type(rss) not in {int, float}
        or not isfinite(rss)
        or not 0 < rss <= CRITERIA["maximum_worker_rss_bytes"]
    ):
        reasons.append("worker_memory_budget_not_verified")
    if type(latency) not in {int, float} or not isfinite(latency) or latency <= 0 or not timings:
        reasons.append("request_latency_not_measured")
    else:
        for row in timings:
            b, m = row.get("lexical_seconds"), row.get("hybrid_seconds", {}).get(channel)
            if (
                type(b) not in {int, float}
                or type(m) not in {int, float}
                or not isfinite(b)
                or not isfinite(m)
                or b <= 0
                or m < 0
            ):
                reasons.append("request_latency_not_measured")
            elif latency + m > b * CRITERIA["latency_ratio"] + CRITERIA["latency_slack_seconds"]:
                reasons.append("cold_request_latency_budget_exceeded")
    return {
        "channel": channel,
        "criteria": CRITERIA.copy(),
        "allowed": not reasons,
        "reasons": sorted(set(reasons)),
        "working_runtime_changed": False,
    }
