"""Rules/shadow/model policy on the SAME eligible candidates and rule explanations."""

import logging
from math import isfinite
from pathlib import Path

from domain.serving_ranker import RankerPolicy, RankingDecision, ServingRankerArtifact
from scoring.linear_ranker import FEATURE_NAMES, feature_vector
from scoring.serving_features import FEATURE_SPEC, feature_contract, pair_features

logger = logging.getLogger(__name__)
_last_failure = None


def _warn_fallback(reason):
    global _last_failure
    if _last_failure != reason:
        logger.warning("Ранжировщик: %s; используются правила", reason)
        _last_failure = reason


def activation_reasons(artifact):
    try:
        return _activation_reasons(artifact)
    except (TypeError, ValueError, AttributeError):
        return ["validation_invalid"]


def _activation_reasons(artifact):
    def valid(value):
        return type(value) in {int, float} and isfinite(value) and 0 <= value <= 1

    proof, provenance = artifact.validation, artifact.provenance
    reasons = []
    if provenance.get("label_origins") != ["human"]:
        reasons.append("labels_not_human")
    if not proof.get("split_verified") or proof.get("validation_groups", 0) < 1:
        reasons.append("split_unverified")
    if (
        not provenance.get("validation_groups")
        or not provenance.get("validation_profile_ids")
        or set(artifact.model.training_groups) & set(provenance["validation_groups"])
        or set(artifact.model.training_profile_ids) & set(provenance["validation_profile_ids"])
        or set(artifact.model.training_query_ids) & set(provenance.get("validation_query_ids", []))
    ):
        reasons.append("split_overlap_or_missing")
    if (
        type(proof.get("eligibility_violations")) is not int
        or proof.get("eligibility_violations") != 0
    ):
        reasons.append("eligibility_unverified")
    for name in ("all_candidates", "serving_threshold"):
        result = proof.get(name, {})
        baseline, learned = result.get("baseline_ndcg"), result.get("model_ndcg")
        if not valid(baseline) or not valid(learned) or learned - baseline < 0.01:
            reasons.append(name + "_gain_unconfirmed")
        groups = result.get("by_profession", {})
        expected = provenance.get("professions", [])
        if not expected or set(groups) != set(expected):
            reasons.append(name + "_coverage_missing")
        for group in groups.values():
            b, m = group.get("baseline_ndcg"), group.get("model_ndcg")
            if not valid(b) or not valid(m) or group.get("positive_pairs", 0) < 1:
                reasons.append(name + "_positive_examples_missing")
            elif m + 1e-9 < b:
                reasons.append(name + "_group_regression")
    return sorted(set(reasons))


def load_artifact(path, parameters):
    if path is None:
        return None, "model_unavailable"
    try:
        path = Path(path)
        if not path.is_file():
            return None, "model_unavailable"
        with path.open("rb") as stream:
            content = stream.read(512 * 1024 + 1)
        if len(content) > 512 * 1024:
            return None, "artifact_too_large"
        artifact = ServingRankerArtifact.model_validate_json(content)
    except OSError:
        return None, "model_unavailable"
    except ValueError:
        return None, "artifact_invalid"
    if (
        artifact.feature_contract != feature_contract()
        or artifact.model.feature_names != FEATURE_NAMES
        or artifact.model.text_feature_spec != FEATURE_SPEC
    ):
        return None, "feature_contract_mismatch"
    if artifact.retrieval_parameters != parameters:
        return None, "retrieval_parameters_mismatch"
    return artifact, None


def apply_ranker(ranking, profile, policy, parameters):
    policy = policy or RankerPolicy()
    if policy.mode == "rules" or not ranking:
        return ranking
    artifact, reason = load_artifact(policy.model_path, parameters)
    if artifact and policy.mode == "model" and activation_reasons(artifact):
        reason = "quality_gate_not_passed"
    if reason:
        _warn_fallback(reason)
        for index, (_, match) in enumerate(ranking, 1):
            match.ranking = RankingDecision(
                requested_mode=policy.mode, baseline_rank=index, fallback_reason=reason
            )
        return ranking
    predictions = []
    try:
        for index, (vacancy, match) in enumerate(ranking, 1):
            vector = feature_vector(pair_features(vacancy, profile, match))
            contributions = {
                name: (value - mean) / scale * coefficient
                for name, value, mean, scale, coefficient in zip(
                    FEATURE_NAMES,
                    vector,
                    artifact.model.means,
                    artifact.model.scales,
                    artifact.model.coefficients,
                    strict=True,
                )
            }
            decision = RankingDecision(
                requested_mode=policy.mode,
                model_id=artifact.model_id,
                feature_contract=artifact.feature_contract,
                baseline_rank=index,
                model_score=sum(contributions.values()),
                contributions=contributions,
            )
            predictions.append((vacancy, match, decision))
        predictions.sort(key=lambda item: (-item[2].model_score, item[2].baseline_rank))
        for index, (_, _, decision) in enumerate(predictions, 1):
            decision.model_rank = index
            decision.applied = policy.mode == "model"
    except (ValueError, OverflowError):
        _warn_fallback("prediction_invalid")
        for index, (_, match) in enumerate(ranking, 1):
            match.ranking = RankingDecision(
                requested_mode=policy.mode,
                baseline_rank=index,
                fallback_reason="prediction_invalid",
            )
        return ranking
    for _, match, decision in predictions:
        match.ranking = decision
    # Ineligible diagnostic rows can never be promoted over admissible ones.
    if policy.mode == "model":
        return [
            (v, m)
            for v, m, _ in sorted(
                predictions, key=lambda item: (not item[1].eligible, item[2].model_rank)
            )
        ]
    return ranking
