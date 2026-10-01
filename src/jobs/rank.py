import logging
from datetime import UTC, datetime

from domain import CandidateProfile, Vacancy
from domain.matching import MatchResult
from domain.retrieval import RetrievalParameters
from domain.vacancy_analysis import VacancyCategory, is_fresh
from jobs.candidates import retrieve_candidates
from observability import emit_event
from observability.actions import observed
from scoring import rank_vacancies
from scoring.serving_ranker import apply_ranker
from storage.repositories import VacancyRepository


@observed("matching")
async def rank_recent(
    repository: VacancyRepository,
    profile: CandidateProfile,
    *,
    days: int | None = None,
    limit: int | None = None,
    category: VacancyCategory | None = None,
    include_ineligible: bool = False,
    excluded_ids=(),
    excluded_keys=(),
    ranker_policy=None,
    cache_owner: int | None = None,
    cache_stats: dict | None = None,
    now: datetime | None = None,
    diagnostics: dict | None = None,
) -> list[tuple[Vacancy, MatchResult]]:
    if days is not None and days < 1:
        raise ValueError("Количество дней должно быть положительным")
    now = now or datetime.now(UTC)
    stats = diagnostics if diagnostics is not None else {}
    selection = await retrieve_candidates(
        repository,
        profile,
        parameters=RetrievalParameters(k=limit) if limit is not None else RetrievalParameters(),
        now=now,
        days=days,
        category=category,
        include_ineligible=include_ineligible,
        excluded_ids=excluded_ids,
        excluded_keys=excluded_keys,
        diagnostics=stats,
    )
    selection.items = [v for v in selection.items if is_fresh(v, now=now, days=days)]
    if cache_owner is not None:
        from jobs.saved_selections import score_selection

        ranked = await score_selection(
            repository,
            selection,
            profile,
            owner_id=cache_owner,
            now=now,
            days=days,
            category=category,
            include_ineligible=include_ineligible,
            stats=cache_stats,
        )
    else:
        ranked = rank_vacancies(
            selection.items,
            profile,
            now=now,
            include_ineligible=include_ineligible,
        )
    for vacancy, match in ranked:
        match.retrieval = selection.evidence[vacancy.id]
    result = apply_ranker(ranked, profile, ranker_policy, selection.parameters)
    stats.update(scored_count=len(result), top_k=selection.parameters.k)
    emit_event(logging.getLogger(__name__), "selection.scored", result="success", **stats)
    return result
