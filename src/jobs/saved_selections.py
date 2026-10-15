"""Reuse invariant rule work, always select current Top-K and refresh time-dependent scores."""

from datetime import UTC, timedelta
from functools import lru_cache
from pathlib import Path

from domain.matching import MatchResult
from scoring import score_vacancy
from scoring.baseline import WEIGHTS
from scoring.serving_features import digest, feature_contract, profile_revision
from storage.repositories.selection_repository import MAX_POOL, SelectionRepository

TTL = timedelta(minutes=15)


@lru_cache(maxsize=1)
def selection_contract():
    # Also version cache adapter logic; no feature-contract changes to artifacts.
    return digest(
        {
            "features": feature_contract(),
            "adapter": Path(__file__).read_text(encoding="utf-8"),
            "version": "saved-selection-v1",
        }
    )


def vacancy_revision(vacancy):
    return digest(
        dict(
            source=vacancy.model_dump(
                mode="json", exclude={"raw_payload", "collected_at", "professional_facts"}
            ),
            facts=vacancy.professional_facts.revision if vacancy.professional_facts else None,
        )
    )


def refresh_time(match, vacancy, now):
    # All other date-dependent evidence expires at the UTC date boundary (request key).
    match.components["freshness"] = max(
        0.0, 1 - max(0.0, (now - vacancy.published_at).total_seconds() / 86400) / 30
    )
    match.score = round(
        100
        * sum(
            WEIGHTS[name] * value for name, value in match.components.items() if value is not None
        ),
        2,
    )
    match.retrieval = None
    match.ranking = None
    return match


async def score_selection(
    repository,
    selection,
    profile,
    *,
    owner_id,
    now,
    days,
    category,
    include_ineligible,
    stats=None,
):
    if owner_id < 1:
        raise ValueError("Invalid selection owner")
    context = {
        "version": "saved-selection-v1",
        "profile_revision": profile_revision(profile),
        "contract": selection_contract(),
        "parameters": selection.parameters.model_dump(),
        "days": days,
        "category": category.value if category else None,
        "include_ineligible": include_ineligible,
        "utc_date": now.astimezone(UTC).date().isoformat(),
        # No single catalogue counter is needed: current selection is read every request,
        # and each retained entry is addressed by the full normalized vacancy revision.
        "catalogue_strategy": "live-selection-and-per-vacancy-revision",
        "model_strategy": "always-apply-current-model-after-baseline",
    }
    revision = digest(context)
    store = SelectionRepository(repository.session)
    entries = await store.load(owner_id, revision, now)
    old = {entry.get("id"): entry for entry in entries if isinstance(entry, dict)}
    current, ranked, hits = [], [], 0
    for vacancy in selection.items:
        content = vacancy_revision(vacancy)
        entry = old.get(str(vacancy.id), {})
        match = None
        if entry.get("revision") == content:
            try:
                match = MatchResult.model_validate(entry["match"])
                if match.vacancy_id != str(vacancy.id) or not match.eligible:
                    match = None
            except (ValueError, TypeError, KeyError):
                pass
        if match is None:
            match = score_vacancy(vacancy, profile, now=now)
        else:
            hits += 1
            refresh_time(match, vacancy, now)
        current.append(
            {
                "id": str(vacancy.id),
                "revision": content,
                "match": match.model_dump(mode="json", exclude={"retrieval", "ranking"}),
            }
        )
        if match.eligible or include_ineligible:
            ranked.append((vacancy, match))
    current_ids = {entry["id"] for entry in current}
    retained = [
        entry for entry in entries if isinstance(entry, dict) and entry.get("id") not in current_ids
    ]
    # Sliding retention is bounded; incompatible/expired pools are replaced wholesale.
    await store.save(owner_id, revision, now, now + TTL, context, (current + retained)[:MAX_POOL])
    if stats is not None:
        stats.update(
            selected=len(selection.items), reused=hits, recomputed=len(selection.items) - hits
        )
    return sorted(
        ranked,
        key=lambda pair: (-pair[1].score, -pair[0].published_at.timestamp(), pair[0].source_id),
    )
