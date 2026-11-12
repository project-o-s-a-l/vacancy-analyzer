"""Common eligibility, history, duplicate and policy admission; no mandatory fill."""

from domain.vacancy_duplicates import vacancy_duplicate_key
from scoring.admission_policy import LEGACY_ADMISSION_VERSION

ADMISSION_VERSION = "dispatch-admission-v2"


def admit(ranking, *, min_score=50, top=5, blocked=(), blocked_keys=()):
    if not 1 <= top <= 20 or not 0 <= min_score <= 100:
        raise ValueError("Invalid admission settings")
    unseen, seen_keys = [], set()
    for vacancy, match in ranking:
        decision = match.ranking
        applied = decision is not None and decision.applied
        passed = decision.admission_passed if applied else match.score >= min_score
        match.admission = {
            "version": ADMISSION_VERSION,
            "policy": decision.admission_policy if applied else LEGACY_ADMISSION_VERSION,
            "model_id": decision.model_id if applied else None,
            "threshold": decision.admission_threshold if applied else min_score,
            "evidence_floor": decision.evidence_floor if applied else None,
            "evidence_support": decision.evidence_support if applied else None,
            "max_display": decision.admission_max_display if applied else top,
            "passed": False,
            "reason": decision.admission_reason if applied else "below_legacy_threshold",
        }
        if not match.eligible or (match.comparison and match.comparison.exclusions):
            match.admission["reason"] = "ineligible"
            continue
        key = vacancy_duplicate_key(vacancy)
        if vacancy.id in blocked or key is not None and (key in blocked_keys or key in seen_keys):
            match.admission["reason"] = "history_or_duplicate"
            continue
        unseen.append((vacancy, match))
        match.admission["passed"] = passed
        if passed:
            match.admission["reason"] = "model_rank_and_evidence" if applied else "legacy_threshold"
        if key is not None:
            seen_keys.add(key)
    selected = [(v, m) for v, m in unseen if m.admission["passed"]]
    cap = min(
        [
            top,
            *[
                m.ranking.admission_max_display
                for _, m in ranking
                if m.ranking and m.ranking.applied
            ],
        ]
    )
    return unseen, selected, selected[:cap]
