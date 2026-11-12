"""Historical signals for review. Actions are not relevance judgments."""

from collections import Counter
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import select

from domain import CandidateProfile
from domain.comparison import COMPARISON_FEATURES
from domain.matching_dataset import PairFeatures
from jobs.feedback_context import slices
from scoring.linear_ranker import feature_vector
from scoring.serving_features import FEATURE_SPEC, digest, feature_contract, profile_revision
from storage.models import ExposureModel, FeedbackEventModel


def utc(value):
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def observation(row, exposure):
    context, action = row.exposure_context or {}, row.action_context or {}
    snapshot = context.get("ranker_context") or {}
    evidence = context.get("feedback_evidence") or {}
    doubts = []
    if (
        exposure is None
        or exposure.chat_id != row.user_id
        or exposure.vacancy_id != row.vacancy_id
        or exposure.context != row.exposure_context
        or action.get("exposure_resolution") != "confirmed_snapshot"
        or utc(exposure.confirmed_at) > utc(row.created_at)
    ):
        return None, "unresolved_exposure"
    if (
        snapshot.get("feature_contract") != feature_contract()
        or snapshot.get("feature_spec") != FEATURE_SPEC
    ):
        return None, "feature_contract_mismatch"
    try:
        original = CandidateProfile.model_validate(context["profile"])
        features = PairFeatures.model_validate(snapshot["features"])
        feature_vector(features)
        if set(features.comparison_features) - set(COMPARISON_FEATURES):
            raise ValueError("Unknown feature names")
        if snapshot["profile_revision"] != profile_revision(original):
            raise ValueError("Profile hash mismatch")
        if original.professional_facts and original.professional_facts.owner_id != row.user_id:
            raise ValueError("Profile owner mismatch")
        if len(bytes.fromhex(context["payload_hash"])) != 32:
            raise ValueError("Missing source revision")
        rank = snapshot["display_rank"]
        if type(rank) is not int or not 1 <= rank <= 20:
            raise ValueError("Invalid position")
    except (ValueError, KeyError, TypeError):
        return None, "invalid_features"
    if (
        snapshot.get("version") != "ranking-exposure-v3"
        or evidence.get("version") != "feedback-evidence-v1"
    ):
        doubts.append("legacy_context")
    if context.get("surface", evidence.get("surface")) == "history":
        doubts.append("history_rerender")
    if context.get("profile_write_revision") is None or context.get("profile_epoch") is None:
        doubts.append("profile_write_version_missing")
    if not snapshot.get("vacancy_facts_revision") or not snapshot.get("candidate_facts_revision"):
        doubts.append("facts_revision_missing")
    if not evidence.get("retrieval"):
        doubts.append("search_context_missing")
    if not evidence.get("retrieval_contract") or not evidence.get("policy_contract"):
        doubts.append("policy_identity_missing")
    if snapshot.get("admission") is None or evidence.get("model_id") is None:
        doubts.append("admission_identity_missing")
    if (
        action.get("profile")
        and profile_revision(CandidateProfile.model_validate(action["profile"]))
        != snapshot["profile_revision"]
    ):
        doubts.append("preferences_changed_after_show")
    if action.get("payload_hash") != context["payload_hash"]:
        doubts.append("vacancy_changed_after_show")
    if utc(row.created_at) - utc(exposure.confirmed_at) > timedelta(days=30):
        doubts.append("late_action")
    if action.get("state_changed") is False:
        doubts.append("no_state_change")
    if action.get("state_changed") is None:
        doubts.append("transition_unknown")
    personal = row.reason in {"salary", "location", "company"}
    if personal:
        doubts.append("personal_preference_reason")
    # No raw decisions, profiles, evidence quotes, titles, URLs or arbitrary keys.
    return {
        "observation_id": digest([row.user_id, row.event_key, str(row.vacancy_id)]),
        "vacancy_ref": digest(str(row.vacancy_id)),
        "profile_revision": snapshot["profile_revision"],
        "source_revision": context["payload_hash"],
        "snapshot_id": str(exposure.id),
        "shown_at": utc(exposure.confirmed_at).isoformat(),
        "action": row.action,
        "reason": row.reason,
        "at": utc(row.created_at).isoformat(),
        "features": features.model_dump(mode="json"),
        "display_rank": rank,
        "context_hash": digest(context),
        "search_context_hash": digest(evidence.get("retrieval")),
        "model_context_hash": digest(snapshot.get("decision")),
        "admission_context_hash": digest([snapshot.get("admission"), evidence.get("min_score")]),
        "admission": {
            key: value
            for key, value in (snapshot.get("admission") or {}).items()
            if key in {"threshold", "evidence_floor", "evidence_support", "max_display", "passed"}
            and (value is None or type(value) in {int, float, bool})
        },
        "ranking_decision": {
            key: value
            for key, value in (snapshot.get("decision") or {}).items()
            if key in {"applied", "baseline_rank", "model_rank", "model_score", "admission_passed"}
            and (value is None or type(value) in {int, float, bool})
        },
        "duplicate_group": digest(evidence.get("vacancy_group") or str(row.vacancy_id)),
        "slices": slices(original),
        "candidate_channel": (evidence.get("retrieval") or {}).get("strategy")
        if (evidence.get("retrieval") or {}).get("strategy") in {"lexical", "recent_fallback"}
        else "unknown",
        "propensity": None,
        "state_changed": action.get("state_changed"),
        "state_version": (action.get("state_after") or {}).get("version"),
        "reason_parent_ref": digest(
            [row.user_id, action["reason_parent_event"], str(row.vacancy_id)]
        )
        if action.get("reason_parent_event")
        else None,
        "label": None,
        "label_origin": "user_behavior_not_relevance",
        "signal_scope": "personal_preference" if personal else "behavior",
        "context_available": "frozen_features_only",
        "training_eligible": False,
        "review_eligible": not any(
            d in doubts
            for d in (
                "legacy_context",
                "history_rerender",
                "search_context_missing",
                "profile_write_version_missing",
                "facts_revision_missing",
                "policy_identity_missing",
                "admission_identity_missing",
            )
        ),
        "doubts": doubts,
        "review_required": True,
        "split_group": None,
    }, None


async def feedback_observations(sessions, user_id, *, limit=500, as_of=None, session=None):
    """Trusted internal owner-scoped read; file export requires LearningAccess."""
    if user_id < 1 or not 1 <= limit <= 5000:
        raise ValueError("Некорректный владелец или лимит")
    as_of = as_of or datetime.now(UTC)
    if session is None:
        async with sessions() as own:
            return await feedback_observations(
                sessions, user_id, limit=limit, as_of=as_of, session=own
            )
    rows = list(
        await session.scalars(
            select(FeedbackEventModel)
            .where(FeedbackEventModel.user_id == user_id, FeedbackEventModel.created_at <= as_of)
            .order_by(FeedbackEventModel.created_at, FeedbackEventModel.event_key)
            .limit(limit + 1)
        )
    )
    observations, skipped, notes = [], Counter(), []
    ids = set()
    for row in rows[:limit]:
        try:
            ids.add(UUID((row.action_context or {}).get("snapshot_id", "")))
        except (ValueError, TypeError):
            pass
    exposures = (
        {
            e.id: e
            for e in await session.scalars(
                select(ExposureModel).where(
                    ExposureModel.chat_id == user_id, ExposureModel.id.in_(ids)
                )
            )
        }
        if ids
        else {}
    )
    for row in rows[:limit]:
        try:
            exposure = exposures.get(UUID((row.action_context or {}).get("snapshot_id", "")))
        except (ValueError, TypeError):
            exposure = None
        item, reason = observation(row, exposure)
        if reason:
            skipped[reason] += 1
            notes.append(
                {
                    "observation_id": digest([row.user_id, row.event_key]),
                    "action": row.action,
                    "context_available": False,
                    "training_eligible": False,
                    "reason": reason,
                }
            )
        else:
            observations.append(item)
    # Keep every action; do not transfer terminal values between exposures/profiles.
    chains = Counter(o["vacancy_ref"] for o in observations)
    positions = Counter(o["display_rank"] for o in observations)
    roles = Counter(role for o in observations for role in o["slices"]["roles"])
    for item in observations:
        if chains[item["vacancy_ref"]] > 1:
            item["doubts"].append("repeated_or_reversed_actions")
        item["review_priority"] = (
            len(item["doubts"])
            + 1 / positions[item["display_rank"]]
            + sum(1 / roles[role] for role in item["slices"]["roles"])
            + int(not item["slices"]["tasks_present"])
        )
    queue = sorted(observations, key=lambda o: (-o["review_priority"], o["observation_id"]))
    return {
        "version": "feedback-training-observations-v2",
        "as_of": as_of.isoformat(),
        "feature_contract": feature_contract(),
        "observations": observations,
        "review_queue": [o["observation_id"] for o in queue],
        "skipped": dict(skipped),
        "review_notes": notes,
        "truncated": len(rows) > limit,
        "policy": {
            "automatic_labels": False,
            "unseen_negatives": False,
            "no_response_label": None,
            "propensity": "unknown; position retained, no invented inverse weighting",
            "review": (
                "Two independent reviews and adjudication; complete frozen sources required"
            ),
            "privacy": "No free text exported. Pseudonymous features are still private data.",
        },
    }
