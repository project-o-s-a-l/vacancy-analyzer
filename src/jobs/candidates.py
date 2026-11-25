"""Shared Top-K selection before expensive matching; no network or user state cache."""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from time import perf_counter

from domain.retrieval import RetrievalParameters
from domain.vacancy_duplicates import vacancy_duplicate_key
from jobs.serving_compute import compute
from scoring.fact_projection import fact_projection, projection_contract
from scoring.request_context import prepared_profile_request
from scoring.retrieval import (
    lexical_evidence,
    profile_query,
    retrieval_eligibility,
)
from storage.repositories.candidate_repository import CandidateRepository, compact_order

logger = logging.getLogger(__name__)


@dataclass
class CandidateSelection:
    items: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    scanned: int = 0
    eligible: int = 0
    parameters: RetrievalParameters = field(default_factory=RetrievalParameters)
    cache_entries: list = field(default_factory=list)


def ready(row, version):
    return bool(
        row.get("lexical_document")
        and row.get("lexical_version") == version
        and row.get("lexical_revision")
        and row["lexical_revision"] == row.get("source_revision")
        and row.get("lexical_contract") == projection_contract()
        and row.get("facts_revision")
        and row["facts_revision"] == row.get("projection_facts_revision")
    )


@prepared_profile_request
async def retrieve_candidates(
    repository,
    profile,
    *,
    parameters=None,
    now=None,
    days=None,
    category=None,
    include_ineligible=False,
    history_owner=None,
    release_read=None,
    excluded_ids=(),
    excluded_keys=(),
    candidate_source=None,
    diagnostics=None,
    catalog_diagnostics=False,
):
    """Compact SQL Top-N, bounded residual refill, bounded observable projection repair.

    No implicit date window. PostgreSQL rank CPU still depends on matching rows.
    At most pool_size + repair_budget detailed rows, K final ranking operations.
    A changed compact revision is skipped and refilled; a later request sees its new rank.
    """
    params = parameters or RetrievalParameters()
    now = now or datetime.now(UTC)
    if now.tzinfo is None or now.utcoffset() is None or days is not None and days < 1:
        raise ValueError("Некорректное время или период отбора")
    result = CandidateSelection(parameters=params)
    source = candidate_source or CandidateRepository(repository.session)
    query = await compute(profile_query, profile)
    options = dict(
        now=now,
        since=now - timedelta(days=days) if days else None,
        category=category,
        include_ineligible=include_ineligible,
        excluded_ids=set(excluded_ids),
        excluded_keys=set(excluded_keys),
    )
    if history_owner is not None:
        options["history_owner"] = history_owner
    stats = dict(
        scanned_count=0,
        required_excluded=0,
        copy_excluded=0,
        recheck_excluded=0,
        compact_count=0,
        detail_count=0,
        repair_count=0,
        refill_rounds=0,
        pool_limit=params.pool_size,
        repair_limit=params.repair_budget,
        pool_budget_exhausted=0,
        projection_budget_exhausted=0,
        source_exhausted=0,
        catalog_counts_sampled=int(catalog_diagnostics),
        history_ids=len(excluded_ids),
        history_groups=len(excluded_keys),
        shortlist_ms=0.0,
        detail_ms=0.0,
        residual_ms=0.0,
        repair_ms=0.0,
    )
    if history_owner is not None:
        # IDs stay in SQL. Their full count would require scanning the owner's history.
        stats.pop("history_ids")
        stats.pop("history_groups")
    if catalog_diagnostics:
        stats.update(await source.diagnostic_counts(profile, **options))

    # ID order is deliberate: bounded degradation must never masquerade as a date filter.
    started = perf_counter()
    stale = await source.stale_ids(profile, limit=params.repair_budget + 1, **options)
    stats["compact_count"] += len(stale)
    stats["projection_budget_exhausted"] = int(len(stale) > params.repair_budget)
    repairs = {}
    repair_rows = await source.load_selected([r["id"] for r in stale[: params.repair_budget]])

    def repair():
        for row in repair_rows:
            vacancy = source.vacancy(row)
            document = fact_projection(vacancy, vacancy.professional_facts)
            evidence = lexical_evidence(document, query, params)
            repairs[vacancy.id] = dict(row) | dict(
                lexical_score=evidence.lexical_score, copy_count=1
            )

    if release_read is not None:
        await release_read()
    await compute(repair)
    stats["repair_count"] = len(repairs)
    stats["detail_count"] += len(repairs)
    stats["repair_ms"] = (perf_counter() - started) * 1000

    attempted, accepted_groups = set(), set()
    selected_orders = {}
    for positive in (True, False):
        if len(result.items) >= params.k:
            break
        buffer = []
        while len(attempted) < params.pool_size and len(result.items) < params.k:
            remaining = params.pool_size - len(attempted)
            page_size = min(params.batch_size, remaining, params.k - len(result.items))
            current = options | dict(
                excluded_ids=options["excluded_ids"] | attempted,
                excluded_keys=options["excluded_keys"] | accepted_groups,
            )
            buffer = [r for r in buffer if r["duplicate_group"] not in accepted_groups]
            if not buffer:
                started = perf_counter()
                rows = await source.shortlist(
                    profile, query, params, limit=remaining + 1, positive=positive, **current
                )
                stats["shortlist_ms"] += (perf_counter() - started) * 1000
                stats["compact_count"] += len(rows)
                pending = [
                    r
                    for key, r in repairs.items()
                    if key not in attempted
                    and (r["lexical_score"] > 0) == positive
                    and r["duplicate_group"] not in current["excluded_keys"]
                ]
                combined = {r["id"]: r for r in [*rows, *pending]}
                buffer = sorted(combined.values(), key=compact_order)[:remaining]
            picks, buffer = buffer[:page_size], buffer[page_size:]
            if not picks:
                if not positive:
                    stats["source_exhausted"] = 1 - stats["projection_budget_exhausted"]
                break
            stats["refill_rounds"] += 1
            started = perf_counter()
            loaded = {r["id"]: r for r in await source.load_selected([p["id"] for p in picks])}
            stats["detail_ms"] += (perf_counter() - started) * 1000
            stats["detail_count"] += len(loaded)
            started = perf_counter()

            def residual(picks=picks, loaded=loaded):
                for pick in picks:
                    attempted.add(pick["id"])
                    row = loaded.get(pick["id"])
                    if row is None or any(
                        row.get(key) != pick.get(key)
                        for key in ("source_revision", "facts_revision", "projection_revision")
                    ):
                        stats["recheck_excluded"] += 1
                        continue
                    vacancy = source.vacancy(row)
                    result.scanned += 1
                    from domain.vacancy_analysis import is_fresh

                    group = (
                        row["duplicate_group"] or vacancy_duplicate_key(vacancy) or str(vacancy.id)
                    )
                    if (
                        group in options["excluded_keys"]
                        or group in accepted_groups
                        or not is_fresh(vacancy, now=now, days=days)
                    ):
                        stats["recheck_excluded"] += 1
                        continue
                    if category is not None and (
                        vacancy.analysis is None or vacancy.analysis.extraction.category != category
                    ):
                        stats["recheck_excluded"] += 1
                        continue
                    if (
                        not include_ineligible
                        and not retrieval_eligibility(vacancy, profile, now).eligible
                    ):
                        stats["required_excluded"] += 1
                        continue
                    result.eligible += 1
                    accepted_groups.add(group)
                    stats["copy_excluded"] += pick["copy_count"] - 1
                    document = (
                        row["lexical_document"]
                        if ready(row, params.version)
                        else fact_projection(vacancy, vacancy.professional_facts)
                    )
                    evidence = lexical_evidence(document, query, params)
                    evidence.source_payload_hash = row["source_payload_hash"]
                    result.evidence[vacancy.id] = evidence
                    result.cache_entries.append(
                        dict(
                            id=str(vacancy.id),
                            evidence=evidence.model_dump(),
                            **{
                                k: row.get(k)
                                for k in (
                                    "source_revision",
                                    "facts_revision",
                                    "projection_revision",
                                )
                            },
                        )
                    )
                    result.items.append(vacancy)
                    selected_orders[vacancy.id] = compact_order(
                        dict(pick) | {"lexical_score": evidence.lexical_score}
                    )
                    if len(result.items) >= params.k:
                        break

            if release_read is not None:
                await release_read()
            await compute(residual)
            stats["residual_ms"] += (perf_counter() - started) * 1000
    result.items.sort(key=lambda v: selected_orders[v.id])
    order = {str(v.id): i for i, v in enumerate(result.items)}
    result.cache_entries.sort(key=lambda e: order[e["id"]])
    stats["scanned_count"] = result.scanned
    stats["retained_groups"] = len(result.items)
    stats["pool_budget_exhausted"] = int(
        len(result.items) < params.k and len(attempted) >= params.pool_size
    )
    if diagnostics is not None:
        diagnostics.update(stats)
    return result
