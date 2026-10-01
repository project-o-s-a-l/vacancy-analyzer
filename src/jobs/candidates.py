"""Shared Top-K selection before expensive matching; no network or user state cache."""

import heapq
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from domain.retrieval import RetrievalParameters
from domain.vacancy_duplicates import vacancy_duplicate_key
from observability import emit_event
from scoring.retrieval import (
    lexical_evidence,
    profile_query,
    retrieval_eligibility,
    search_document,
)
from storage.repositories.candidate_repository import CandidateRepository

logger = logging.getLogger(__name__)


@dataclass
class CandidateSelection:
    items: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    scanned: int = 0
    eligible: int = 0
    parameters: RetrievalParameters = field(default_factory=RetrievalParameters)


@dataclass
class _Pick:
    order: tuple
    key: object
    group: object
    evidence: object

    def __lt__(self, other):
        return self.order > other.order  # The worst retained item is the heap root.


async def retrieve_candidates(
    repository,
    profile,
    *,
    parameters=None,
    now=None,
    days=None,
    category=None,
    include_ineligible=False,
    excluded_ids=(),
    excluded_keys=(),
    candidate_source=None,
    diagnostics=None,
):
    params = parameters or RetrievalParameters()
    now = now or datetime.now(UTC)
    if now.tzinfo is None or now.utcoffset() is None or days is not None and days < 1:
        raise ValueError("Некорректное время или период отбора")
    result = CandidateSelection(parameters=params)
    source = candidate_source or CandidateRepository(repository.session)
    stats = dict(scanned_count=0, required_excluded=0, copy_excluded=0, recheck_excluded=0)
    if diagnostics is not None and logger.isEnabledFor(logging.INFO):
        diagnostics.update(
            await source.diagnostic_counts(
                profile,
                now=now,
                since=now - timedelta(days=days) if days is not None else None,
                category=category,
                include_ineligible=include_ineligible,
                excluded_ids=excluded_ids,
                excluded_keys=excluded_keys,
            )
        )
    query = profile_query(profile)
    heap, groups = [], {}

    def retain(pick):
        old = groups.get(pick.group)
        if old is not None:
            stats["copy_excluded"] += 1
            # Bounded details only; the final counter includes every collision.
            if stats["copy_excluded"] <= 16:
                emit_event(
                    logger,
                    "selection.copy_excluded",
                    result="skipped",
                    code="duplicate_copy",
                    vacancy_id=pick.key if old.order <= pick.order else old.key,
                    related_vacancy_id=old.key if old.order <= pick.order else pick.key,
                    duplicate_group=pick.group,
                )
            if old.order <= pick.order:
                return
            heap.remove(old)
            heapq.heapify(heap)
            del groups[old.group]
        if len(heap) < params.k:
            heapq.heappush(heap, pick)
            groups[pick.group] = pick
        elif pick.order < heap[0].order:
            removed = heapq.heapreplace(heap, pick)
            del groups[removed.group]
            groups[pick.group] = pick

    for positive in (True, None, False):
        if positive is False and len(heap) == params.k and heap[0].evidence.lexical_score > 0:
            break
        done = False
        async for batch in source.batches(
            profile,
            query,
            now=now,
            since=now - timedelta(days=days) if days is not None else None,
            category=category,
            include_ineligible=include_ineligible,
            excluded_ids=excluded_ids,
            excluded_keys=excluded_keys,
            positive=positive,
            batch_size=params.batch_size,
        ):
            for row in batch:
                result.scanned += 1
                stats["scanned_count"] += 1
                vacancy = source.vacancy(row)
                if (
                    not include_ineligible
                    and not retrieval_eligibility(vacancy, profile, now).eligible
                ):
                    stats["required_excluded"] += 1
                    continue
                result.eligible += 1
                document = row["lexical_document"]
                if (
                    row["lexical_version"] != params.version
                    or not document
                    or not row["lexical_revision"]
                    or row["lexical_revision"] != row["source_revision"]
                ):
                    document = search_document(vacancy.model_dump())
                evidence = lexical_evidence(document, query, params)
                evidence.source_payload_hash = row["source_payload_hash"]
                pick = _Pick(
                    (
                        -evidence.lexical_score,
                        -vacancy.published_at.timestamp(),
                        vacancy.source_id,
                        str(vacancy.id),
                    ),
                    vacancy.id,
                    row["duplicate_group"] or vacancy_duplicate_key(vacancy) or vacancy.id,
                    evidence,
                )
                if positive is False and len(heap) == params.k and pick.order >= heap[0].order:
                    done = True  # Remaining zero-score rows are older, in stable SQL order.
                    break
                retain(pick)
            if done:
                break
    selected = {pick.key: pick.evidence for pick in heap}
    stats["retained_groups"] = len(selected)
    for row in await source.load_selected(list(selected)):
        vacancy = source.vacancy(row)
        if row["duplicate_group"] is not None and row["duplicate_group"] in excluded_keys:
            continue
        # READ COMMITTED may observe an update/deletion after the streaming pass.
        from domain.vacancy_analysis import is_fresh

        if not is_fresh(vacancy, now=now, days=days):
            continue
        if not include_ineligible and not retrieval_eligibility(vacancy, profile, now).eligible:
            continue
        result.items.append(vacancy)
        document = (
            row["lexical_document"]
            if (
                row["lexical_version"] == params.version
                and row["lexical_revision"]
                and row["lexical_revision"] == row["source_revision"]
            )
            else None
        )
        evidence = lexical_evidence(
            document or search_document(vacancy.model_dump()), query, params
        )
        evidence.source_payload_hash = row["source_payload_hash"]
        result.evidence[vacancy.id] = evidence
    orders = {pick.key: pick.order for pick in heap}
    result.items.sort(key=lambda v: orders[v.id])
    stats["recheck_excluded"] = len(selected) - len(result.items)
    if diagnostics is not None:
        diagnostics.update(stats)
    return result
