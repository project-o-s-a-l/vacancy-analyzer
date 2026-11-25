import logging
from datetime import UTC, datetime
from time import perf_counter

from domain import CandidateProfile, Vacancy
from domain.exchange_rates import use_rates
from domain.matching import MatchResult
from domain.retrieval import RetrievalParameters
from domain.vacancy_analysis import VacancyCategory, is_fresh
from jobs.candidates import retrieve_candidates
from jobs.exchange_rates import rates_for
from jobs.serving_compute import bounded_recommendations, compute
from observability import emit_event
from observability.actions import observed
from scoring import rank_vacancies
from scoring.request_context import prepared_profile_request
from scoring.serving_ranker import apply_ranker
from storage.repositories import VacancyRepository


def default_retrieval_parameters():
    from jobs.settings import Settings

    settings = Settings()
    return RetrievalParameters(
        k=settings.retrieval_k,
        pool_factor=settings.retrieval_pool_factor,
        normalization=settings.retrieval_normalization,
        backend=settings.retrieval_backend,
    )


@observed("matching")
@bounded_recommendations
@prepared_profile_request
async def rank_recent(repository: VacancyRepository, profile: CandidateProfile, **kwargs):
    live = repository is not None and kwargs.get("candidate_source") is None
    with use_rates(await rates_for(profile, live=live)):
        return await _rank_with_cache(repository, profile, **kwargs)


async def _rank_with_cache(repository, profile, **kwargs):
    owner = kwargs.get("cache_owner")
    if owner is None or kwargs.get("candidate_source") is not None:
        return await _rank_recent(repository, profile, **kwargs)
    from jobs.candidate_cache import identity
    from jobs.saved_selections import selection_contract
    from scoring.serving_cache import cache_scope
    from scoring.serving_features import profile_revision

    revisions = await identity(repository.session, owner)
    profile_key = await compute(profile_revision, profile)
    with cache_scope(
        (id(repository.session.bind), owner, revisions[2], profile_key, selection_contract())
    ):
        return await _rank_recent(repository, profile, _cache_revisions=revisions, **kwargs)


async def _rank_recent(
    repository: VacancyRepository,
    profile: CandidateProfile,
    *,
    days: int | None = None,
    limit: int | None = None,
    category: VacancyCategory | None = None,
    include_ineligible: bool = False,
    history_owner=None,
    release_read=None,
    excluded_ids=(),
    excluded_keys=(),
    ranker_policy=None,
    experimental_ranker: bool = False,
    pair_reranker=None,
    pair_owner=None,
    cache_owner: int | None = None,
    cache_stats: dict | None = None,
    now: datetime | None = None,
    diagnostics: dict | None = None,
    candidate_source=None,
    trace: dict | None = None,
    retrieval_parameters: RetrievalParameters | None = None,
    _cache_revisions=None,
) -> list[tuple[Vacancy, MatchResult]]:
    if days is not None and days < 1:
        raise ValueError("Количество дней должно быть положительным")
    now = now or datetime.now(UTC)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must contain a timezone")
    now = now.astimezone(UTC)
    stats = diagnostics if diagnostics is not None else {}
    if experimental_ranker and (
        candidate_source is None or repository is not None or cache_owner is not None
    ):
        raise ValueError("Experimental model is restricted to frozen offline sources")
    if pair_reranker is not None and (not experimental_ranker or pair_owner is None):
        raise ValueError("Pair reranker requires an explicit offline experiment and owner")
    started = perf_counter()
    if retrieval_parameters is None:
        retrieval_parameters = default_retrieval_parameters()
    if limit is not None:
        retrieval_parameters = retrieval_parameters.model_copy(update={"k": limit})
    options = dict(
        parameters=retrieval_parameters,
        now=now,
        days=days,
        category=category,
        include_ineligible=include_ineligible,
        excluded_ids=excluded_ids,
        excluded_keys=excluded_keys,
        diagnostics=stats,
        candidate_source=candidate_source,
    )
    if release_read is not None:
        options["release_read"] = release_read
    if history_owner is not None:
        options["history_owner"] = history_owner
    cache_values = None
    if _cache_revisions is not None:
        from jobs.candidate_cache import retrieve

        selection, cache_values = await retrieve(
            repository, profile, owner=cache_owner, revisions=_cache_revisions, **options
        )
    else:
        selection = await retrieve_candidates(repository, profile, **options)
    if release_read is not None:
        await release_read()
    if trace is not None:
        trace["retrieval_seconds"] = perf_counter() - started
        trace["retrieved"] = [f"{v.source}:{v.source_id}" for v in selection.items]
        trace["scanned"] = selection.scanned
        trace["retrieval_diagnostics"] = {
            key: value for key, value in stats.items() if not key.endswith("_ms")
        }
        for key, value in stats.items():
            if key.endswith("_ms"):
                trace["retrieval_" + key[:-3] + "_seconds"] = value / 1000
    selection.items = [v for v in selection.items if is_fresh(v, now=now, days=days)]
    started = perf_counter()
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
            release_read=release_read,
            owner_epoch=_cache_revisions[2] if _cache_revisions else None,
        )
    else:
        ranked = await compute(
            rank_vacancies,
            selection.items,
            profile,
            now=now,
            include_ineligible=include_ineligible,
            trace=trace,
        )
    if trace is not None:
        trace["filter_rank_seconds"] = perf_counter() - started
        trace["filtered"] = [f"{v.source}:{v.source_id}" for v, _ in ranked]
    for vacancy, match in ranked:
        match.retrieval = selection.evidence[vacancy.id]
    started = perf_counter()
    if release_read is not None:
        await release_read()
    result = await compute(
        apply_ranker,
        ranked,
        profile,
        ranker_policy,
        selection.parameters,
        experimental=experimental_ranker,
    )
    if pair_reranker is not None:
        result = await compute(pair_reranker.apply, result, profile, owner=pair_owner, trace=trace)
    if cache_values is not None:
        from jobs.candidate_cache import save

        await save(repository.session, cache_owner, _cache_revisions, cache_values)
    if trace is not None:
        trace["rerank_seconds"] = perf_counter() - started
        trace["ranked"] = [f"{v.source}:{v.source_id}" for v, _ in result]
        trace["ranking_decisions"] = {
            f"{v.source}:{v.source_id}": m.ranking.model_dump(mode="json") if m.ranking else None
            for v, m in result
        }
    stats.update(scored_count=len(result), top_k=selection.parameters.k)
    emit_event(logging.getLogger(__name__), "selection.scored", result="success", **stats)
    return result
