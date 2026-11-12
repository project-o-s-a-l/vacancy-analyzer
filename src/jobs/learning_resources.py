"""Derive release resource proof from raw paired runs, not a caller's allowed flag."""

from datetime import datetime
from math import isfinite

from scoring.evaluation_metrics import interval
from scoring.serving_features import digest


def resource_proof(raw):
    if raw.get("version") != "learning-resource-runs-v1" or raw.get("origin") != "measured":
        raise ValueError("Measured raw resource runs required")
    protocol = raw["protocol"]
    planned, started = (
        datetime.fromisoformat(v) for v in (protocol["frozen_at"], raw["started_at"])
    )
    if planned.tzinfo is None or started.tzinfo is None or planned > started:
        raise ValueError("Resource budgets must be frozen before measurements")
    if (
        raw.get("rss_scope") != "whole_process_tree"
        or not raw.get("stand")
        or not raw.get("workload")
    ):
        raise ValueError("Explicit stand/workload/whole-process-tree scope required")
    warm, cold, rss = raw["warm_pairs_ms"], raw["cold_ms"], raw["rss_bytes"]
    if len(warm) < 3 or len(cold) < 2 or not rss:
        raise ValueError("Insufficient cold/warm resource runs")
    values = [
        *cold,
        *rss,
        *(v for pair in warm for v in pair),
        protocol["cold_budget_ms"],
        protocol["rss_budget_bytes"],
    ]
    if any(type(v) not in {int, float} or not isfinite(v) or v <= 0 for v in values):
        raise ValueError("Resource measurements must be finite and positive")
    if any(len(pair) != 2 for pair in warm):
        raise ValueError("Paired reference/candidate measurements required")
    ratios = [candidate / reference for reference, candidate in warm]
    confidence = interval(ratios, seed=20261111, repeats=2000, confidence=0.95)
    return {
        "origin": "measured",
        "evidence_sha256": digest(raw),
        "stand": raw["stand"],
        "workload": raw["workload"],
        "protocol_sha256": digest(protocol),
        "warm_latency_ratio_upper95": confidence["high"],
        "cold_latency_ms": max(cold),
        "cold_budget_ms": protocol["cold_budget_ms"],
        "whole_tree_rss_bytes": max(rss),
        "rss_budget_bytes": protocol["rss_budget_bytes"],
        "warm_repeats": len(warm),
        "cold_repeats": len(cold),
        "failures": raw["failures"],
    }
