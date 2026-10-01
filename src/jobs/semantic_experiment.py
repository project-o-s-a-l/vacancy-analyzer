"""Frozen offline comparison; never trains a model or activates the bot's candidate path."""

import hashlib
import json
import logging
import statistics
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from domain import CandidateProfile, Vacancy
from domain.vacancy_analysis import is_fresh
from jobs.benchmark_data import load_bundle, vacancy_key
from jobs.candidates import retrieve_candidates
from jobs.hybrid_candidates import hybrid_candidates
from jobs.retrieval_evaluation import MemoryCandidateSource
from jobs.semantic_acceptance import acceptance, current_code_hashes
from jobs.semantic_index import build_index, valid_vector
from jobs.semantic_model import MODEL_ID, MODEL_SPEC
from jobs.semantic_process import MAX_RSS, MIN_AVAILABLE, encode_bounded
from jobs.semantic_text import TEXT_SPEC, profile_text, text_revision
from observability import emit_event
from scoring import rank_vacancies
from scoring.matching_evaluation import ranking_metrics
from scoring.retrieval import retrieval_eligibility
from scoring.serving_features import digest


def _identity(bundle):
    return {
        "baseline_labels_sha256": hashlib.sha256(
            (Path(bundle) / "labels.jsonl").read_bytes()
        ).hexdigest(),
        "experiment_code_sha256": current_code_hashes(),
    }


def _case_metrics(result, profile, now, labels):
    channels = {}
    for name, items in (
        ("lexical", result.lexical),
        ("semantic", result.semantic),
        ("rrf", result.fused),
        ("novelty", result.novelty),
    ):
        ranking = rank_vacancies(items, profile, now=now)
        positives = {key for key, value in labels.items() if value >= 2}
        keys = {vacancy_key(v) for v in items}
        channels[name] = {
            "selected": len(items),
            "positive_recall": len(positives & keys) / len(positives) if positives else None,
            "serving": ranking_metrics(
                [vacancy_key(v) for v, m in ranking if m.score >= 50], labels
            ),
            "eligibility_violations": sum(
                not is_fresh(v, now=now) or not retrieval_eligibility(v, profile, now).eligible
                for v in items
            ),
            "candidate_keys": [vacancy_key(v) for v in items],
        }
    return channels


def _query_cache(profiles, path, encode):
    old = {}
    if path.exists():
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("Кэш запросов превышает допустимый размер")
        value = json.loads(path.read_text(encoding="utf-8"))
        seal = value.pop("sha256", None)
        if (
            digest(value) != seal
            or value.get("version") != "synthetic-query-cache-v1"
            or value.get("model_id") != MODEL_ID
            or value.get("text_spec") != TEXT_SPEC
        ):
            raise ValueError("Несовместимый или повреждённый кэш запросов эксперимента")
        old = value.get("entries", {})
    entries = {}
    tasks = []
    names = []
    for name, profile in profiles.items():
        text = profile_text(profile)
        revision = text_revision(text)
        entry = old.get(name)
        if entry and entry.get("input_revision") == revision and valid_vector(entry.get("vector")):
            entries[name] = entry
        else:
            entries[name] = {"input_revision": revision}
            tasks.append({"kind": "query", "text": text})
            names.append(name)
    resources = encode(tasks)
    if len(resources["values"]) != len(names):
        raise ValueError("Неполный результат вычисления запросов")
    for name, value in zip(names, resources["values"], strict=True):
        if not valid_vector(value.get("vector")):
            raise ValueError("Некорректный вектор запроса")
        entries[name].update(value)
    value = {
        "version": "synthetic-query-cache-v1",
        "model_id": MODEL_ID,
        "text_spec": TEXT_SPEC,
        "entries": entries,
    }
    value["sha256"] = digest(value)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return {name: entry["vector"] for name, entry in entries.items()}, resources | {
        "computed": len(names),
        "reused": len(profiles) - len(names),
    }


async def run_semantic_experiment(
    bundle, review_directory, model_directory, output, *, reuse_from=None
):
    output = Path(output)
    if output.exists():
        raise ValueError("Выберите новый каталог эксперимента")
    catalog, cases, judgments, manifest = load_bundle(Path(bundle))
    if any(j.relevance is None for j in judgments):
        raise ValueError("Для метрик фиксированного набора нужна полная разметка")
    review_directory = Path(review_directory)
    review = json.loads((review_directory / "review-pairs.json").read_text(encoding="utf-8"))
    snapshot_path = review_directory / "catalogue.json"
    if hashlib.sha256(snapshot_path.read_bytes()).hexdigest() != review["catalogue_sha256"]:
        raise ValueError("Снимок диагностических пар изменился")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    observed = datetime.fromisoformat(snapshot["captured_at"])
    review_jobs = [Vacancy.model_validate(row) for row in snapshot["vacancies"]]
    query_profiles = {case.id: case.profile for case in cases.cases}
    for pair in review["pairs"]:
        query_profiles[pair["case_id"]] = CandidateProfile.model_validate(pair["synthetic_profile"])
    output.mkdir(parents=True)
    if reuse_from is not None:
        import shutil

        for name in ("benchmark-index.json", "review-index.json"):
            # The builder validates source hashes, model and text versions before reuse.
            shutil.copyfile(Path(reuse_from) / name, output / name)
        if (Path(reuse_from) / "query-vectors.json").exists():
            shutil.copyfile(Path(reuse_from) / "query-vectors.json", output / "query-vectors.json")

    def encode(tasks):
        return encode_bounded(tasks, model_directory)

    emit_event(logging.getLogger(__name__), "experiment.indexing", stage="benchmark")
    base_index, base_resources = build_index(
        catalog.vacancies, output / "benchmark-index.json", encode
    )
    emit_event(logging.getLogger(__name__), "experiment.indexing", stage="review")
    review_index, review_resources = build_index(review_jobs, output / "review-index.json", encode)
    vectors, query_resources = _query_cache(query_profiles, output / "query-vectors.json", encode)
    # A real cold single-query round trip, not an average of a batch.
    measured_case = cases.cases[0].id
    started = time.perf_counter()
    query_measurement = encode(
        [{"kind": "query", "text": profile_text(query_profiles[measured_case])}]
    )
    query_measurement = {k: v for k, v in query_measurement.items() if k != "values"} | {
        "wall_seconds": time.perf_counter() - started,
        "case_id": measured_case,
    }
    base_source = MemoryCandidateSource(catalog.vacancies)
    labels = defaultdict(dict)
    for j in judgments:
        labels[j.case_id][j.vacancy_key] = j.relevance
    metrics = []
    for case in cases.cases:
        started = time.perf_counter()
        selected = await hybrid_candidates(
            catalog.vacancies,
            case.profile,
            base_index,
            vectors[case.id],
            now=catalog.reference_time,
            candidate_source=base_source,
        )
        metrics.append(
            {
                "case_id": case.id,
                "profession": case.profession,
                "split": case.split,
                "positive_pairs": sum(value >= 2 for value in labels[case.id].values()),
                "maximum_relevance": max(labels[case.id].values()),
                "label_origins": sorted({j.origin for j in judgments if j.case_id == case.id}),
                "channels": _case_metrics(
                    selected, case.profile, catalog.reference_time, labels[case.id]
                ),
                "seconds_selection_and_scoring": time.perf_counter() - started,
                "missing_or_stale": selected.stale_or_missing,
            }
        )
    discoveries = []
    search_measurements = []
    review_source = MemoryCandidateSource(review_jobs)
    for name in sorted({p["case_id"] for p in review["pairs"]}):
        started = time.perf_counter()
        base = await retrieve_candidates(
            None, query_profiles[name], now=observed, candidate_source=review_source
        )
        rank_vacancies(base.items, query_profiles[name], now=observed)
        lexical_seconds = time.perf_counter() - started
        started = time.perf_counter()
        selected = await hybrid_candidates(
            review_jobs,
            query_profiles[name],
            review_index,
            vectors[name],
            now=observed,
            candidate_source=review_source,
        )
        candidate_seconds = time.perf_counter() - started
        hybrid_seconds = {}
        for channel, items in (("rrf", selected.fused), ("novelty", selected.novelty)):
            started = time.perf_counter()
            rank_vacancies(items, query_profiles[name], now=observed)
            hybrid_seconds[channel] = candidate_seconds + time.perf_counter() - started
        search_measurements.append(
            {"case_id": name, "lexical_seconds": lexical_seconds, "hybrid_seconds": hybrid_seconds}
        )
        ids = {
            method: {vacancy_key(v) for v in items}
            for method, items in (
                ("lexical", selected.lexical),
                ("semantic", selected.semantic),
                ("rrf", selected.fused),
                ("novelty", selected.novelty),
            )
        }
        scores = {
            method: {
                vacancy_key(v): m.score
                for v, m in rank_vacancies(items, query_profiles[name], now=observed)
            }
            for method, items in (
                ("lexical", selected.lexical),
                ("rrf", selected.fused),
                ("novelty", selected.novelty),
            )
        }
        for pair in review["pairs"]:
            if pair["case_id"] != name:
                continue
            key = pair["vacancy"]["source"] + ":" + pair["vacancy"]["source_id"]
            discoveries.append(
                {
                    "pair_id": pair["pair_id"],
                    "relevance": pair["relevance"],
                    "origin": pair["origin"],
                    "retrieved": {method: key in chosen for method, chosen in ids.items()},
                    "rule_score_after_rrf": scores["rrf"].get(key),
                    "rule_score_after_novelty": scores["novelty"].get(key),
                    "independent_holdout": False,
                }
            )

    # Check persistence without any further inference, including unloaded-model reuse.
    def no_inference(tasks):
        if tasks:
            raise AssertionError("Unchanged inputs must not invoke a model")
        return {"values": []}

    _, reuse = build_index(review_jobs, output / "review-index.json", no_inference)
    summaries = {}
    for split in ("training", "validation"):
        rows = [m for m in metrics if m["split"] == split]
        summaries[split] = {
            method: {
                "macro_serving_ndcg_at_5": statistics.mean(
                    [
                        r["channels"][method]["serving"]["ndcg_at_k"]
                        for r in rows
                        if r["channels"][method]["serving"]["ndcg_at_k"] is not None
                    ]
                ),
                "positive_recall_macro": statistics.mean(
                    [
                        r["channels"][method]["positive_recall"]
                        for r in rows
                        if r["channels"][method]["positive_recall"] is not None
                    ]
                ),
            }
            for method in ("lexical", "semantic", "rrf", "novelty")
        }
    resources = [
        {k: v for k, v in res.items() if k != "values"}
        for res in (base_resources, review_resources, query_resources)
    ]
    report = {
        **_identity(bundle),
        "version": "semantic-experiment-v1",
        "model_id": MODEL_ID,
        "model_spec": MODEL_SPEC,
        "text_spec": TEXT_SPEC,
        "exact_technology_guard_families": [["java", "javascript"], ["c", "c++", "c#"]],
        "baseline_manifest": manifest,
        "baseline_label_origins": sorted({j.origin for j in judgments}),
        "review_catalogue_sha256": review["catalogue_sha256"],
        "review_labels_sha256": hashlib.sha256(
            (review_directory / "review-pairs.json").read_bytes()
        ).hexdigest(),
        "rrf_constant": 60,
        "novelty_fraction": 0.25,
        "k": 64,
        "min_score": 50,
        "top": 5,
        "metrics": metrics,
        "summaries": summaries,
        "diagnostic_pairs": discoveries,
        "resources": resources,
        "query_measurement": query_measurement,
        "search_measurements": search_measurements,
        "independent_validation_confirmed": False,
        "unchanged_index_reused": reuse["reused"],
        "limits": {
            "max_rss_tree_bytes": MAX_RSS,
            "min_free_ram_bytes": MIN_AVAILABLE,
            "cpu_threads": 1,
            "cpu_logical_cores": 1,
            "gpu": False,
        },
        "activated": False,
        "working_database_used": False,
        "llm_calls": 0,
        "notes": [
            "Discovery examples were selected after seeing retrieval failures.",
            "The 132-card benchmark labels are assistant drafts, not human acceptance.",
            "Incomplete diagnostic labels cannot provide NDCG over the whole catalogue.",
            "No model/index is loaded by the bot; lexical runtime remains unchanged.",
            "RSS is sampled for the full worker tree; watchdog is not an OS hard cap.",
        ],
    }
    report["acceptance"] = {
        channel: acceptance(report, channel=channel) for channel in ("rrf", "novelty")
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
