"""Fail-closed measured gates. No editable allowed flag is consumed."""

from math import isfinite

from scoring.admission_policy import PROTOCOL
from scoring.serving_features import digest


def _lower(value, minimum=0, *, strict=False):
    low = value.get("low") if isinstance(value, dict) else None
    return (
        type(low) in {int, float}
        and isfinite(low)
        and (low > minimum if strict else low >= minimum)
    )


def gate_reasons(artifact):
    from scoring.release_gates import release_reasons

    provenance, proof = artifact.provenance, artifact.validation
    reasons = release_reasons(proof.get("release_gates"))
    if provenance.get("label_origins") != ["human"] or not provenance.get("label_imports"):
        reasons.append("labels_not_human")
    if proof.get("protocol_hash") != digest(PROTOCOL):
        reasons.append("protocol_mismatch")
    splits = provenance.get("splits", {})
    for key in ("owners", "profiles", "vacancy_groups"):
        groups = [
            set(splits.get(s, {}).get(key, [])) for s in ("train", "development", "calibration")
        ]
        if any(not g for g in groups) or any(
            groups[i] & groups[j] for i in range(3) for j in range(i)
        ):
            reasons.append("split_overlap_or_missing")
    if provenance.get("test_used") is not False:
        reasons.append("final_test_not_reserved")
    if not _lower(proof.get("development", {}).get("rank_delta"), strict=True):
        reasons.append("development_gain_unconfirmed")
    result = proof.get("calibration", {})
    if result.get("owners", 0) < PROTOCOL["minimum_owners"]:
        reasons.append("calibration_too_small")
    if not _lower(result.get("precision_cluster"), PROTOCOL["precision_floor"]):
        reasons.append("precision_floor_unconfirmed")
    if type(result.get("violations")) is not int or result["violations"] != 0:
        reasons.append("eligibility_unverified")
    for metric in (
        "ndcg_at_5",
        "precision_at_5",
        "recall",
        "slots",
        "nonempty",
        "retained_positives",
    ):
        if not _lower(result.get("deltas", {}).get(metric)):
            reasons.append(metric + "_nonregression_unconfirmed")
    for dimension in ("by_profession", "by_experience"):
        groups = result.get(dimension, {})
        if not groups:
            reasons.append(dimension + "_missing")
        for group in groups.values():
            if group.get("owners", 0) < PROTOCOL["minimum_stratum_owners"]:
                reasons.append(dimension + "_too_small")
            for metric in ("precision_at_5", "recall", "slots", "nonempty", "retained_positives"):
                if not _lower(group.get("deltas", {}).get(metric)):
                    reasons.append(dimension + "_nonregression_unconfirmed")
    return sorted(set(reasons))
