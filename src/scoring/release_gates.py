"""Common measured resource/independent comparison gate, separate from model fitting."""

from math import isfinite


def release_reasons(proof):
    if not isinstance(proof, dict):
        return ["release_evidence_missing"]
    reasons = []
    if proof.get("comparison_status") != "passed" or not proof.get("comparison_sha256"):
        reasons.append("independent_comparison_not_passed")
    if proof.get("version") != "model-release-gates-v1":
        reasons.append("release_protocol_mismatch")
    resource = proof.get("resource", {})
    if (
        resource.get("origin") != "measured"
        or not resource.get("evidence_sha256")
        or not resource.get("stand")
    ):
        reasons.append("resource_evidence_missing")
    for key, maximum in (
        ("warm_latency_ratio_upper95", 1),
        ("cold_latency_ms", resource.get("cold_budget_ms")),
        ("whole_tree_rss_bytes", resource.get("rss_budget_bytes")),
    ):
        value = resource.get(key)
        if (
            type(value) not in {int, float}
            or not isfinite(value)
            or value <= 0
            or type(maximum) not in {int, float}
            or not isfinite(maximum)
            or maximum <= 0
            or value > maximum
        ):
            reasons.append(key + "_not_passed")
    if resource.get("warm_repeats", 0) < 3 or resource.get("cold_repeats", 0) < 2:
        reasons.append("resource_repeats_missing")
    if resource.get("failures") != 0 or type(resource.get("failures")) is not int:
        reasons.append("resource_failures")
    return reasons
