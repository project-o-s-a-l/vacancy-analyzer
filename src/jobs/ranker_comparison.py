"""Exercise the actual serving shadow path on frozen validation profiles only."""

import statistics
import time
import tracemalloc

from domain.retrieval import RetrievalParameters
from domain.serving_ranker import RankerPolicy
from jobs.benchmark_data import load_bundle, vacancy_key
from jobs.candidates import retrieve_candidates
from jobs.retrieval_evaluation import MemoryCandidateSource
from scoring import rank_vacancies
from scoring.serving_ranker import apply_ranker, load_artifact


async def compare_ranker(bundle, model_path):
    artifact, reason = load_artifact(model_path, RetrievalParameters())
    if artifact is None:
        raise ValueError("Непригодный артефакт сравнения: " + reason)
    catalog, cases, _, _ = load_bundle(bundle)
    source = MemoryCandidateSource(catalog.vacancies)
    reports = []
    for case in cases.cases:
        if case.split != "validation":
            continue
        selection = await retrieve_candidates(
            None, case.profile, now=catalog.reference_time, candidate_source=source
        )
        ranking = rank_vacancies(selection.items, case.profile, now=catalog.reference_time)
        original_order = [vacancy_key(v) for v, _ in ranking]
        tracemalloc.start()
        started = time.perf_counter()
        compared = apply_ranker(
            ranking,
            case.profile,
            RankerPolicy(mode="shadow", model_path=model_path),
            RetrievalParameters(),
        )
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert [vacancy_key(v) for v, _ in compared] == original_order
        if not all(
            m.ranking and not m.ranking.applied and not m.ranking.fallback_reason
            for _, m in compared
        ):
            raise ValueError("Артефакт изменился во время сравнения; повторите операцию")
        alternative = sorted(compared, key=lambda item: item[1].ranking.model_rank)
        reports.append(
            {
                "case": case.id,
                "profession": case.profession,
                "candidates": len(compared),
                "baseline_order": original_order,
                "model_order": [vacancy_key(v) for v, _ in alternative],
                "model_id": compared[0][1].ranking.model_id if compared else None,
                "seconds_shadow": elapsed,
                "python_peak_bytes": peak,
                "changed_live_order": False,
            }
        )
    return {
        "version": "ranker-shadow-comparison-v1",
        "cases": reports,
        "median_seconds": statistics.median(row["seconds_shadow"] for row in reports),
        "max_python_peak_bytes": max(row["python_peak_bytes"] for row in reports),
        "scope": (
            "Actual shadow feature generation, file parsing and scoring; "
            "precomputed rules/candidates; Python allocations, not RSS or PostgreSQL memory"
        ),
        "live_messages_sent": 0,
        "llm_calls": 0,
    }
