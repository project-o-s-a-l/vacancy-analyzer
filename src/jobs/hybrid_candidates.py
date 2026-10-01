"""Exact semantic channel and RRF on the same full admissibility policy, offline-first."""

from dataclasses import dataclass, field

from domain.retrieval import RetrievalParameters
from domain.vacancy_analysis import is_fresh
from domain.vacancy_duplicates import vacancy_duplicate_key
from jobs.benchmark_data import vacancy_key
from jobs.candidates import retrieve_candidates
from jobs.retrieval_evaluation import MemoryCandidateSource
from jobs.semantic_index import valid_vector
from jobs.semantic_model import MODEL_ID
from jobs.semantic_text import TEXT_SPEC, text_revision, vacancy_text
from scoring.retrieval import retrieval_eligibility, tokens

RRF_CONSTANT = 60


def _technology_conflict(vacancy, profile):
    wanted = set().union(*(tokens(v) for v in profile.preferences.positions))
    actual = tokens(vacancy.title)
    # Preserve exact identities of close spelling families; no profession whitelist.
    return any(
        wanted & family and actual & family and not wanted & actual & family
        for family in ({"java", "javascript"}, {"c", "c++", "c#"})
    )


def reciprocal_rank_fusion(*rankings):
    scores = {}
    for ranking in rankings:
        if len(set(ranking)) != len(ranking):
            raise ValueError("Повторы в канале RRF")
        for rank, key in enumerate(ranking, 1):
            scores[key] = scores.get(key, 0) + 1 / (RRF_CONSTANT + rank)
    return sorted(scores, key=lambda key: (-scores[key], str(key)))


def novelty_fusion(lexical, semantic, *, limit):
    """A fixed <=25% exploration budget; no score threshold or known-pair exceptions."""
    if limit < 1:
        raise ValueError("Предел пула должен быть положительным")

    def group(v):
        return vacancy_duplicate_key(v) or v.id

    lex_groups = {group(v) for v in lexical}
    novel = []
    seen = set()
    for vacancy in semantic:
        key = group(vacancy)
        if key not in lex_groups and key not in seen:
            novel.append(vacancy)
            seen.add(key)
    reserve = min(limit // 4, len(novel))
    result = []
    seen = set()
    for vacancy in [*lexical[: limit - reserve], *novel[:reserve], *lexical, *semantic]:
        key = group(vacancy)
        if key not in seen:
            result.append(vacancy)
            seen.add(key)
            if len(result) == limit:
                break
    return result


@dataclass
class HybridResult:
    lexical: list
    semantic: list = field(default_factory=list)
    fused: list = field(default_factory=list)
    novelty: list = field(default_factory=list)
    fallback_reason: str | None = None
    stale_or_missing: int = 0


async def hybrid_candidates(
    vacancies,
    profile,
    index,
    query_vector,
    *,
    now,
    parameters=None,
    excluded_ids=(),
    excluded_keys=(),
    days=None,
    category=None,
    candidate_source=None,
):
    params = parameters or RetrievalParameters()
    lexical = await retrieve_candidates(
        None,
        profile,
        parameters=params,
        now=now,
        candidate_source=candidate_source or MemoryCandidateSource(vacancies),
        excluded_ids=excluded_ids,
        excluded_keys=excluded_keys,
        days=days,
        category=category,
    )
    result = HybridResult(lexical=lexical.items, fused=lexical.items, novelty=lexical.items)
    if (
        not index
        or index.get("model_id") != MODEL_ID
        or index.get("text_spec") != TEXT_SPEC
        or not valid_vector(query_vector)
    ):
        result.fallback_reason = "semantic_unavailable_or_incompatible"
        return result
    candidates, groups = [], set()
    for vacancy in vacancies:
        key = vacancy_duplicate_key(vacancy) or str(vacancy.id)
        if (
            vacancy.id in excluded_ids
            or key in excluded_keys
            or not is_fresh(vacancy, now=now, days=days)
            or category
            and (not vacancy.analysis or vacancy.analysis.extraction.category != category)
            or not retrieval_eligibility(vacancy, profile, now).eligible
            or _technology_conflict(vacancy, profile)
        ):
            continue
        entry = index["entries"].get(vacancy_key(vacancy))
        if not entry or entry.get("input_revision") != text_revision(vacancy_text(vacancy)):
            result.stale_or_missing += 1
            continue
        if not valid_vector(entry.get("vector")):
            result.stale_or_missing += 1
            continue
        similarity = sum(a * b for a, b in zip(query_vector, entry["vector"], strict=True))
        candidates.append(
            (
                -similarity,
                -vacancy.published_at.timestamp(),
                vacancy.source_id,
                str(vacancy.id),
                key,
                vacancy,
            )
        )
    for *_, group, vacancy in sorted(candidates, key=lambda row: row[:4]):
        if group not in groups:
            result.semantic.append(vacancy)
            groups.add(group)
            if len(result.semantic) == params.k:
                break
    if not result.semantic:
        result.fallback_reason = "no_current_semantic_entries"
        return result
    by_id = {v.id: v for v in [*result.lexical, *result.semantic]}
    order = reciprocal_rank_fusion([v.id for v in result.lexical], [v.id for v in result.semantic])
    result.fused, groups = [], set()
    for key in order:
        vacancy = by_id[key]
        group = vacancy_duplicate_key(vacancy) or vacancy.id
        if group not in groups:
            groups.add(group)
            result.fused.append(vacancy)
            if len(result.fused) == params.k:
                break
    result.novelty = novelty_fusion(result.lexical, result.semantic, limit=params.k)
    return result
