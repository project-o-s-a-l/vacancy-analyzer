"""Bounded baseline score cache; current candidate IDs and exact-time freshness stay separate."""

from datetime import UTC, timedelta
from functools import lru_cache
from pathlib import Path

from domain.contracts import source_contract
from domain.exchange_rates import current_rates
from domain.matching import MatchResult
from scoring import score_vacancy
from scoring.baseline import score_totals
from scoring.serving_cache import vacancy_revision  # public compatibility import
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
            "serving": source_contract(
                (
                    "jobs/candidate_cache.py",
                    "jobs/candidates.py",
                    "storage/repositories/candidate_repository.py",
                    "storage/retrieval_sql.py",
                    "scoring/serving_cache.py",
                    "scoring/request_context.py",
                )
            ),
            "version": "saved-selection-v2",
        }
    )


def refresh_time(match, vacancy, profile, now):
    # All other date-dependent evidence expires at the UTC date boundary (request key).
    match.components["freshness"] = max(
        0.0, 1 - max(0.0, (now - vacancy.published_at).total_seconds() / 86400) / 30
    )
    match.score, match.evidence_coverage = score_totals(
        match.components, profile, vacancy, match.comparison
    )
    match.retrieval = None
    match.ranking = None
    match.admission = None
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
    owner_epoch=None,
    release_read=None,
):
    if owner_id < 1:
        raise ValueError("Invalid selection owner")
    context = {
        "version": "saved-selection-v2",
        "profile_revision": profile_revision(profile),
        "salary_rates": current_rates().revision if profile.preferences.salary_min else None,
        "contract": selection_contract(),
        "parameters": selection.parameters.model_dump(),
        "days": days,
        "category": category.value if category else None,
        "include_ineligible": include_ineligible,
        "utc_date": now.astimezone(UTC).date().isoformat(),
        "catalogue_strategy": "revision-fenced-candidates-and-per-vacancy-baseline",
        "model_strategy": "always-apply-current-model-after-baseline",
    }
    revision = digest(context)
    store = SelectionRepository(repository.session)
    entries = await store.load(owner_id, revision, now)
    if release_read is not None:
        await release_read()
    from jobs.serving_compute import compute

    current, ranked, hits = await compute(
        _score_entries, entries, selection, profile, now, include_ineligible
    )
    if not entries or hits != len(selection.items):
        await store.save(
            owner_id, revision, now, now + TTL, context, current, owner_epoch=owner_epoch
        )
    if stats is not None:
        stats.update(
            selected=len(selection.items), reused=hits, recomputed=len(selection.items) - hits
        )
    return ranked


def _score_entries(entries, selection, profile, now, include_ineligible):
    old = {entry.get("id"): entry for entry in entries if isinstance(entry, dict)}
    current, ranked, hits = [], [], 0
    for vacancy in selection.items:
        content = vacancy_revision(vacancy)
        entry = old.get(str(vacancy.id), {})
        match = None
        if entry.get("revision") == content:
            try:
                match = MatchResult.model_validate(entry["match"])
                if (
                    match.vacancy_id != str(vacancy.id)
                    or not match.eligible
                    or match.comparison is None
                ):
                    match = None
            except (ValueError, TypeError, KeyError):
                pass
        if match is None:
            match = score_vacancy(vacancy, profile, now=now)
        else:
            hits += 1
            refresh_time(match, vacancy, profile, now)
            from scoring.eligibility import check_eligibility

            # Pair atoms depend on the UTC date, already in the key. The exact-time
            # availability and publication constraints are still checked below.
            match.comparison = match.comparison.model_copy(update={"checked_at": now})
            match.eligibility = check_eligibility(
                vacancy, profile, now=now, comparison=match.comparison
            )
            match.eligible = match.eligibility.eligible
        current.append(
            {
                "id": str(vacancy.id),
                "revision": content,
                "match": match.model_dump(
                    mode="json", exclude={"retrieval", "ranking", "admission"}
                ),
            }
        )
        if match.eligible or include_ineligible:
            ranked.append((vacancy, match))
    current_ids = {entry["id"] for entry in current}
    retained = [
        entry for entry in entries if isinstance(entry, dict) and entry.get("id") not in current_ids
    ]
    # Incompatible/expired pools are replaced. Hits alone do not extend retention.
    ranked = sorted(
        ranked,
        key=lambda pair: (-pair[1].score, -pair[0].published_at.timestamp(), pair[0].source_id),
    )
    return (current + retained)[:MAX_POOL], ranked, hits
