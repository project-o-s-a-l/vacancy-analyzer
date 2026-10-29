"""Owner-scoped observations for review, not automatic labels or online training."""

import re
from collections import Counter

from sqlalchemy import select

from domain import CandidateProfile
from domain.matching_dataset import PairFeatures
from scoring.linear_ranker import feature_vector
from scoring.serving_features import FEATURE_SPEC, digest, feature_contract, profile_revision
from storage.models import FeedbackEventModel


async def feedback_observations(sessions, user_id, *, limit=500):
    if user_id < 1 or not 1 <= limit <= 5000:
        raise ValueError("Некорректный владелец или лимит")
    observations, rejected, review_notes = [], Counter(), []
    async with sessions() as session:
        rows = await session.scalars(
            select(FeedbackEventModel)
            .where(FeedbackEventModel.user_id == user_id)
            .order_by(FeedbackEventModel.created_at, FeedbackEventModel.event_key)
            .limit(limit)
        )
        for row in rows:
            context = row.exposure_context or {}
            snapshot = context.get("ranker_context") or {}
            if (row.action_context or {}).get("exposure_resolution") not in {
                "current_message",
                "confirmed_snapshot",
            }:
                rejected["unresolved_exposure"] += 1
                review_notes.append(
                    {
                        "vacancy_ref": digest(str(row.vacancy_id)),
                        "action": row.action,
                        "reason": row.reason,
                        "at": row.created_at.isoformat(),
                        "review_required": True,
                    }
                )
                continue
            if (
                snapshot.get("feature_contract") != feature_contract()
                or snapshot.get("feature_spec") != FEATURE_SPEC
            ):
                rejected["feature_contract_mismatch"] += 1
                continue
            try:
                features = PairFeatures.model_validate(snapshot["features"])
                feature_vector(features)
                group = snapshot["profile_revision"]
                original = CandidateProfile.model_validate(context["profile"])
                if (
                    snapshot.get("version") not in {"ranking-exposure-v1", "ranking-exposure-v2"}
                    or not isinstance(group, str)
                    or not re.fullmatch(r"[a-f0-9]{64}", group)
                    or group != profile_revision(original)
                    or not re.fullmatch(r"[a-f0-9]{64}", context.get("payload_hash") or "")
                ):
                    raise ValueError("Неполный контекст показа")
            except (ValueError, KeyError, TypeError):
                rejected["invalid_features"] += 1
                continue
            observations.append(
                {
                    "observation_id": digest([row.event_key, str(row.vacancy_id)]),
                    "vacancy_ref": digest(str(row.vacancy_id)),
                    "profile_revision": group,
                    "source_revision": context.get("payload_hash"),
                    "snapshot_id": (row.action_context or {}).get("snapshot_id"),
                    "action": row.action,
                    "reason": row.reason,
                    "at": row.created_at.isoformat(),
                    "features": features.model_dump(mode="json"),
                    "display_rank": snapshot.get("display_rank"),
                    "ranking_decision": snapshot.get("decision"),
                    "label": None,
                    "review_required": True,
                    "split_group": None,
                }
            )
    return {
        "version": "feedback-training-observations-v1",
        "feature_contract": feature_contract(),
        "observations": observations,
        "skipped": dict(rejected),
        "review_notes": review_notes,
        "policy": (
            "Only explicit events from this owner; no unseen negatives. "
            "Review reversals and reasons before labels; no automatic training."
        ),
    }
