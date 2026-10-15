"""Offline comparison of indexed lexical selection with exhaustive matching."""

import hashlib
import json
import logging
import statistics
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path

from domain.retrieval import RetrievalParameters
from domain.vacancy_analysis import analysis_input_hash, is_fresh
from domain.vacancy_duplicates import vacancy_duplicate_key
from jobs.benchmark import implementation_hash
from jobs.benchmark_data import load_bundle, vacancy_key
from jobs.candidates import retrieve_candidates
from observability import emit_event
from scoring import rank_vacancies
from scoring.retrieval import (
    document_terms,
    retrieval_eligibility,
)
from storage.payloads import payload_hash
from storage.repositories.candidate_repository import CandidateRepository


class MemoryCandidateSource:
    """Frozen data adapter for the SAME selector, with SQL-equivalence tests."""

    vacancy = staticmethod(CandidateRepository.vacancy)

    def __init__(self, vacancies):
        from nlp.professional_facts import vacancy_facts
        from scoring.fact_projection import fact_projection, projection_contract

        self.rows = []
        for vacancy in vacancies:
            vacancy = vacancy.model_copy(
                update={"professional_facts": vacancy.professional_facts or vacancy_facts(vacancy)}
            )
            document = fact_projection(vacancy, vacancy.professional_facts)
            self.rows.append(
                vacancy.model_dump(exclude={"raw_payload", "professional_facts"})
                | {
                    "professional_facts": vacancy.professional_facts,
                    "lexical_document": document,
                    "lexical_version": document["version"],
                    "lexical_revision": analysis_input_hash(vacancy),
                    "source_revision": analysis_input_hash(vacancy),
                    "source_payload_hash": payload_hash(vacancy.raw_payload),
                    "duplicate_group": vacancy_duplicate_key(vacancy),
                    "lexical_contract": projection_contract(),
                }
            )
        self.rows.sort(key=lambda r: (-r["published_at"].timestamp(), r["source_id"], str(r["id"])))

    async def batches(self, profile, query, **options):
        if options["positive"] is None:
            return  # Frozen documents are always current; no repair lane.
        batch = []
        for row in self.rows:
            # Same boundary as SQL: rich facts/analysis are loaded only for retained IDs.
            projection = {
                key: value
                for key, value in row.items()
                if key not in {"professional_facts", "analysis"}
            }
            vacancy = self.vacancy(projection)
            if not is_fresh(vacancy, now=options["now"]):
                continue
            if options["since"] and vacancy.published_at < options["since"]:
                continue
            if options["category"] and (
                not row.get("analysis")
                or row["analysis"]["extraction"]["category"] != options["category"].value
            ):
                continue
            if (
                vacancy.id in options["excluded_ids"]
                or row["duplicate_group"] in options["excluded_keys"]
            ):
                continue
            overlap = bool(query.terms.intersection(document_terms(row["lexical_document"])))
            if overlap != options["positive"]:
                continue
            batch.append(projection)
            if len(batch) == options["batch_size"]:
                yield batch
                batch = []
        if batch:
            yield batch

    async def load_selected(self, ids):
        return [r for r in self.rows if r["id"] in ids]

    async def diagnostic_counts(self, profile, **options):
        # Memory adapter has no SQL diagnostics. Selection counters are supplied
        # by the shared selector; do not confuse these with production DB cost.
        return {}


def _recall(expected, actual):
    return len(expected & actual) / len(expected) if expected else None


async def evaluate_retrieval(bundle: Path, *, ks=(16, 32, 64, 128, 200, 500)):
    catalog, cases, judgments, manifest = load_bundle(bundle)
    now = catalog.reference_time
    source = MemoryCandidateSource(catalog.vacancies)
    labels = defaultdict(set)
    draft_labels = defaultdict(set)
    for row in judgments:
        if row.origin == "human" and row.relevance is not None and row.relevance >= 2:
            labels[row.case_id].add(row.vacancy_key)
        elif row.origin == "assistant_draft" and row.relevance is not None and row.relevance >= 2:
            draft_labels[row.case_id].add(row.vacancy_key)
    complete_human = all(row.origin == "human" for row in judgments)
    reference, recent = {}, {}
    for case in cases.cases:
        active = [v for v in catalog.vacancies if is_fresh(v, now=now)]
        ranking = rank_vacancies(active, case.profile, now=now)
        reference[case.id] = {vacancy_key(v) for v, _ in ranking[:5]}
        recent[case.id] = [
            vacancy_key(v)
            for v in sorted(
                catalog.vacancies, key=lambda v: (-v.published_at.timestamp(), v.source_id)
            )[:500]
        ]
    experiments = []
    for k in ks:
        parameters = RetrievalParameters(k=k)
        entries = []
        for case in cases.cases:
            tracemalloc.start()
            started = time.perf_counter()
            selection = await retrieve_candidates(
                None, case.profile, parameters=parameters, now=now, candidate_source=source
            )
            elapsed = time.perf_counter() - started
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            chosen = {vacancy_key(v) for v in selection.items}
            positives = labels[case.id]
            # Isolate retrieval loss from a deliberately inadmissible labelled pair.
            eligible = {
                vacancy_key(v)
                for v in catalog.vacancies
                if retrieval_eligibility(v, case.profile, now).eligible and is_fresh(v, now=now)
            }
            entries.append(
                {
                    "case": case.id,
                    "profession": case.profession,
                    "split": case.split,
                    "selected": sorted(chosen),
                    "selected_count": len(chosen),
                    "scanned": selection.scanned,
                    "relevant": len(positives),
                    "eligible_relevant": len(positives & eligible),
                    "recalled": len(positives & chosen),
                    "eligible_recalled": len(positives & eligible & chosen),
                    "recall": _recall(positives, chosen) if complete_human else None,
                    "eligible_recall": _recall(positives & eligible, chosen)
                    if complete_human
                    else None,
                    "pooled_human_recall": _recall(positives, chosen),
                    "assistant_draft_recall_diagnostic": _recall(draft_labels[case.id], chosen),
                    "exhaustive_top5_recall": _recall(reference[case.id], chosen),
                    "legacy_recent_recall": _recall(positives, set(recent[case.id])),
                    "seconds": elapsed,
                    "python_peak_bytes": peak,
                }
            )
        experiments.append(
            {
                "k": k,
                "parameters": parameters.model_dump(),
                "parameters_hash": payload_hash(parameters.model_dump()),
                "cases": entries,
            }
        )
        emit_event(logging.getLogger(__name__), "experiment.progress", count=k)

    def metrics(entries):
        relevant = sum(e["eligible_relevant"] for e in entries)
        by_profession = {}
        for name in {e["profession"] for e in entries}:
            group = [e for e in entries if e["profession"] == name]
            count = sum(e["eligible_relevant"] for e in group)
            by_profession[name] = (
                sum(e["eligible_recalled"] for e in group) / count
                if count and complete_human
                else None
            )
        return {
            "eligible_recall": sum(e["eligible_recalled"] for e in entries) / relevant
            if relevant and complete_human
            else None,
            "by_profession": by_profession,
            "exhaustive_top5_recall": statistics.mean(
                e["exhaustive_top5_recall"]
                for e in entries
                if e["exhaustive_top5_recall"] is not None
            ),
            "median_seconds": statistics.median(e["seconds"] for e in entries),
            "max_python_peak_bytes": max(e["python_peak_bytes"] for e in entries),
        }

    chosen_k = None
    for experiment in experiments:
        experiment["training"] = metrics(
            [e for e in experiment["cases"] if e["split"] == "training"]
        )
        experiment["validation"] = metrics(
            [e for e in experiment["cases"] if e["split"] == "validation"]
        )
        training = experiment["training"]
        if (
            complete_human
            and chosen_k is None
            and training["eligible_recall"] is not None
            and training["eligible_recall"] >= 0.95
            and all(value is None or value >= 0.9 for value in training["by_profession"].values())
            and training["exhaustive_top5_recall"] >= 0.95
        ):
            chosen_k = experiment["k"]
    return {
        "version": "retrieval-evaluation-v1",
        "parameters": RetrievalParameters().model_dump(),
        "bundle": str(bundle),
        "manifest_hash": payload_hash(manifest),
        "labels_hash": hashlib.sha256((bundle / "labels.jsonl").read_bytes()).hexdigest(),
        "label_origins": sorted({row.origin for row in judgments}),
        "human_quality_ready": complete_human,
        "metric_scope": "full_human_catalog" if complete_human else "partial_human_pool",
        "official_protocol": "evaluation-run/evaluation-compare (F01); this is legacy diagnostics",
        "implementation_hash": payload_hash(
            {
                "matching": implementation_hash(),
                "selector": hashlib.sha256(
                    (Path(__file__).parent / "candidates.py").read_bytes()
                ).hexdigest(),
                "sql": hashlib.sha256(
                    (
                        Path(__file__).parents[1] / "storage/repositories/candidate_repository.py"
                    ).read_bytes()
                ).hexdigest(),
            }
        ),
        "criteria_status": "engineering targets; independent human acceptance remains pending",
        "criteria": {
            "training_recall": 0.95,
            "profession_recall": 0.9,
            "exhaustive_top5_recall": 0.95,
        },
        "chosen_k_on_training": chosen_k,
        "experiments": experiments,
        "measurement_scope": (
            "selector only; prebuilt index; tracemalloc is Python allocations, "
            "not RSS or PostgreSQL memory"
        ),
    }


def write_report(report, output):
    if output.exists() or output.with_suffix(".md").exists():
        raise ValueError("Выберите новый файл отчёта")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Проверка предварительного отбора VA-05",
        "",
        f"K по training: {report['chosen_k_on_training']}",
        f"Независимая человеческая разметка: {report['human_quality_ready']}",
        "",
        "| K | Recall training | Recall validation | Top-5 validation | "
        "Время validation, мс | Пик Python, КиБ |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for e in report["experiments"]:
        t, v = e["training"], e["validation"]
        lines.append(
            f"| {e['k']} | {t['eligible_recall']} | {v['eligible_recall']} | "
            f"{v['exhaustive_top5_recall']:.3f} | {v['median_seconds'] * 1000:.1f} | "
            f"{v['max_python_peak_bytes'] / 1024:.1f} |"
        )
    lines += [
        "",
        "Черновая разметка не заменяет независимую приёмку VA-02.3.",
        report["measurement_scope"],
    ]
    chosen = next(
        (e for e in report["experiments"] if e["k"] == report["chosen_k_on_training"]), None
    )
    if chosen:
        missing = sorted(
            name for name, recall in chosen["validation"]["by_profession"].items() if recall is None
        )
        lines += [
            "",
            "Профессии без положительных validation-примеров: " + ", ".join(missing),
            "Для них полнота не измерена; единичные синтетические сценарии не заменяют разметку.",
        ]
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
