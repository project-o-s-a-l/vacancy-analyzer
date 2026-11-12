"""Versioned rank/evidence admission. Scores are margins, never probabilities."""

from functools import lru_cache

from domain.contracts import source_contract
from scoring.serving_features import digest

MODEL_ADMISSION_VERSION = "rank-evidence-admission-v1"
LEGACY_ADMISSION_VERSION = "legacy-baseline-threshold-v1"
# Fixed before the experiment; calibration chooses only thresholds, not these gates.
PROTOCOL = {
    "version": "f10-selection-protocol-v1",
    "C": [0.1, 1.0, 10.0],
    "evidence_floors": [0.0, 0.25, 0.5, 0.75],
    "score_quantiles": [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
    "precision_floor": 0.8,
    "minimum_owners": 8,
    "minimum_stratum_owners": 2,
    "top": 5,
    "seed": 20261110,
    "bootstrap_repeats": 2000,
    "confidence": 0.95,
    "precision_interval": "owner-cluster ratio bootstrap; Wilson reported descriptively",
    "normalization": "train-only-weighted-mean-scale-v1",
    "model_selection": "development full-catalog NDCG@5; ties smaller C",
    "admission_selection": "calibration: precision floor; maximize positives and slots",
    "release": "human complete; paired nonregression; independent validation still required",
}


@lru_cache(maxsize=1)
def policy_contract():
    return digest(
        {
            "protocol": PROTOCOL,
            "sources": source_contract(
                ("scoring/admission_policy.py", "jobs/admission.py", "scoring/ranker_gates.py")
            ),
        }
    )


def evidence_support(match):
    """Known matching/contradicting professional comparisons / all comparisons.

    Unknown and conflicting values provide no support; absence of comparisons is zero.
    This measures available evidence, not confidence or the probability of relevance.
    """
    items = match.comparison.items if match.comparison else []
    items = [
        i
        for i in items
        if i.kind != "condition" and i.importance not in {"context", "not_required"}
    ]
    return sum(i.state in {"match", "contradiction"} for i in items) / len(items) if items else 0.0


def model_admission(match, score, policy):
    support = evidence_support(match)
    if not match.eligible or (match.comparison and match.comparison.exclusions):
        return False, "ineligible", support
    if support == 0 or support < policy.evidence_floor:
        return False, "insufficient_evidence", support
    if score < policy.threshold:
        return False, "below_model_threshold", support
    return True, "model_rank_and_evidence", support
