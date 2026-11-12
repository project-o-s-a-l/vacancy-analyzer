"""Rules/shadow/model policy on the SAME eligible candidates and rule explanations."""

import logging
from pathlib import Path

from domain.contracts import normalization_contract
from domain.serving_ranker import RankerPolicy, RankingDecision, ServingRankerArtifact
from scoring.admission_policy import model_admission, policy_contract
from scoring.linear_ranker import FEATURE_NAMES, feature_vector
from scoring.serving_features import (
    FEATURE_SPEC,
    feature_contract,
    pair_features,
    retrieval_contract,
)

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
    except (TypeError, ValueError, AttributeError, KeyError):
        return ["validation_invalid"]


def _activation_reasons(artifact):
    from scoring.ranker_gates import gate_reasons

    return gate_reasons(artifact)


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
    if artifact.normalization_contract != normalization_contract():
        return None, "normalization_contract_mismatch"
    if artifact.policy_contract != policy_contract():
        return None, "policy_contract_mismatch"
    if artifact.retrieval_contract != retrieval_contract():
        return None, "retrieval_contract_mismatch"
    if artifact.retrieval_parameters != parameters:
        return None, "retrieval_parameters_mismatch"
    return artifact, None


def apply_ranker(ranking, profile, policy, parameters, *, experimental=False):
    policy = policy or RankerPolicy()
    if policy.mode == "rules" or not ranking:
        return ranking
    artifact, reason = load_artifact(policy.model_path, parameters)
    if artifact and policy.mode == "model" and not experimental and activation_reasons(artifact):
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
            passed, reason, support = model_admission(
                match, decision.model_score, artifact.admission
            )
            decision.admission_policy = artifact.admission.version
            decision.admission_threshold = artifact.admission.threshold
            decision.evidence_floor = artifact.admission.evidence_floor
            decision.evidence_support = support
            decision.admission_max_display = artifact.admission.max_display
            decision.admission_passed = passed
            decision.admission_reason = reason
            decision.experimental = experimental
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
