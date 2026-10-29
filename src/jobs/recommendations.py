"""Shared ranking/admission boundary for Telegram, CLI and evaluation adapters."""

from dataclasses import dataclass

from jobs.admission import admit
from jobs.rank import rank_recent


@dataclass
class Recommendations:
    ranking: list
    unseen: list
    admitted: list
    displayed: list


async def recommend(repository, profile, *, min_score=50, top=5, **options):
    ranking = await rank_recent(repository, profile, **options)
    unseen, admitted, displayed = admit(
        ranking,
        min_score=min_score,
        top=top,
        blocked=options.get("excluded_ids", ()),
        blocked_keys=options.get("excluded_keys", ()),
    )
    return Recommendations(ranking, unseen, admitted, displayed)
