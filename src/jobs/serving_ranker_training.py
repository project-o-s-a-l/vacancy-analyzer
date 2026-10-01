"""Train once on training profiles; evaluate held-out profiles without tuning on them."""

import json
import statistics
import time
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from domain.matching_dataset import MatchingDataset, ReviewPair
from domain.retrieval import RetrievalParameters
from domain.serving_ranker import ServingRankerArtifact
from jobs.benchmark_data import load_bundle, vacancy_key
from jobs.candidates import retrieve_candidates
from jobs.ranker_training import validate_split
from jobs.retrieval_evaluation import MemoryCandidateSource
from scoring import rank_vacancies
from scoring.linear_ranker import (
    LabeledQuery,
    fit_pairwise_ranker,
    profile_id,
    ranker_scores,
)
from scoring.matching_evaluation import ranking_metrics
from scoring.retrieval import retrieval_eligibility
from scoring.serving_features import FEATURE_SPEC, digest, feature_contract, pair_features
from scoring.serving_ranker import activation_reasons


def dataset_for_case(case, selection, now):
    ranking = rank_vacancies(selection.items, case.profile, now=now)
    if not ranking:
        return None
    query_id = digest({"profile": case.profile.model_dump(mode="json"), "case": case.id})
    pairs = [
        ReviewPair(
            pair_id=digest([query_id, vacancy_key(v)]),
            vacancy=v,
            baseline_rank=i,
            features=pair_features(v, case.profile, match),
        )
        for i, (v, match) in enumerate(ranking, 1)
    ]
    return MatchingDataset(
        created_at=now,
        query_id=query_id,
        query_source="profile",
        profile=case.profile,
        feature_spec=FEATURE_SPEC,
        corpus_size=len(pairs),
        word_known_fraction=1,
        warnings=["Фиксированные признаки профиля, не полный текст резюме и не TF-IDF."],
        pairs=pairs,
    )


def aggregate(entries, method):
    values = [e[method]["ndcg_at_k"] for e in entries if e[method]["ndcg_at_k"] is not None]
    return statistics.mean(values) if values else None


async def train_serving_ranker(bundle: Path, *, allow_assistant_draft=False):
    catalog, cases, labels, manifest = load_bundle(bundle)
    origins = sorted({row.origin for row in labels})
    if any(row.relevance is None for row in labels):
        raise ValueError("Нужна полная явная разметка: пропуски не считаются отказами")
    if origins != ["human"] and not (allow_assistant_draft and origins == ["assistant_draft"]):
        raise ValueError(
            "Нужна проверенная разметка; черновой эксперимент разрешается отдельным флагом"
        )
    source = MemoryCandidateSource(catalog.vacancies)
    judgments = {(r.case_id, r.vacancy_key): r.relevance for r in labels}
    prepared, training, validation, skipped = {}, [], [], []
    started = time.perf_counter()
    for case in cases.cases:
        selection = await retrieve_candidates(
            None, case.profile, now=catalog.reference_time, candidate_source=source
        )
        dataset = dataset_for_case(case, selection, catalog.reference_time)
        prepared[case.id] = dataset
        if dataset is None:
            skipped.append({"case": case.id, "reason": "no_admissible_candidates"})
            continue
        grades = {
            pair.pair_id: judgments[case.id, vacancy_key(pair.vacancy)] for pair in dataset.pairs
        }
        example = LabeledQuery(dataset, grades, case.group)
        if case.split == "training":
            if len(set(grades.values())) < 2:
                skipped.append({"case": case.id, "reason": "no_pairwise_preferences"})
            else:
                training.append(example)
        else:
            validation.append(example)
    validate_split(training, validation)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        raise ValueError("Для обучения установите дополнение проекта .[ml]") from None

    with threadpool_limits(limits=1):
        model = fit_pairwise_ranker(training)
    evaluations = []
    violations = 0
    for case in cases.cases:
        if case.split != "validation":
            continue
        dataset = prepared[case.id]
        predictions = ranker_scores(model, dataset) if dataset else {}
        ordered = (
            sorted(dataset.pairs, key=lambda pair: (-predictions[pair.pair_id], pair.baseline_rank))
            if dataset
            else []
        )
        violations += sum(
            not retrieval_eligibility(pair.vacancy, case.profile, catalog.reference_time).eligible
            for pair in ordered[:5]
        )
        whole_labels = {
            vacancy_key(v): judgments[case.id, vacancy_key(v)] for v in catalog.vacancies
        }
        entry = {
            "case": case.id,
            "group": case.group,
            "profession": case.profession,
            "eligible_candidates": len(ordered),
            "positive_catalog": sum(v >= 2 for v in whole_labels.values()),
        }
        for name, minimum in (("all_candidates", 0), ("serving_threshold", 50)):
            baseline = (
                [p for p in dataset.pairs if p.features.baseline_score >= minimum]
                if dataset
                else []
            )
            learned = [p for p in ordered if p.features.baseline_score >= minimum]
            entry[name] = {
                "baseline": ranking_metrics(
                    [vacancy_key(p.vacancy) for p in baseline], whole_labels
                ),
                "model": ranking_metrics([vacancy_key(p.vacancy) for p in learned], whole_labels),
                "baseline_order": [vacancy_key(p.vacancy) for p in baseline],
                "model_order": [vacancy_key(p.vacancy) for p in learned],
            }
        evaluations.append(entry)
    professions = sorted({case.profession for case in cases.cases})
    proof = {
        "split_verified": True,
        "validation_groups": len(
            {case.group for case in cases.cases if case.split == "validation"}
        ),
        "eligibility_violations": violations,
        "serving_min_score": 50,
        "cases": evaluations,
    }
    for name in ("all_candidates", "serving_threshold"):
        rows = [{"profession": e["profession"], **e[name]} for e in evaluations]
        proof[name] = {
            "baseline_ndcg": aggregate(rows, "baseline"),
            "model_ndcg": aggregate(rows, "model"),
            "by_profession": {
                profession: {
                    "baseline_ndcg": aggregate(
                        [r for r in rows if r["profession"] == profession], "baseline"
                    ),
                    "model_ndcg": aggregate(
                        [r for r in rows if r["profession"] == profession], "model"
                    ),
                    "queries": sum(r["profession"] == profession for r in rows),
                    "positive_pairs": sum(
                        e["positive_catalog"] for e in evaluations if e["profession"] == profession
                    ),
                }
                for profession in professions
            },
        }
    validation_datasets = [q.dataset for q in validation]
    provenance = {
        "bundle_manifest_hash": digest(manifest),
        "catalog_hash": sha256((bundle / "catalog.json").read_bytes()).hexdigest(),
        "labels_hash": sha256((bundle / "labels.jsonl").read_bytes()).hexdigest(),
        "label_origins": origins,
        "professions": professions,
        "validation_groups": sorted({q.group for q in validation}),
        "validation_profile_ids": sorted({profile_id(d) for d in validation_datasets}),
        "validation_query_ids": sorted(d.query_id for d in validation_datasets),
        "skipped_training_queries": skipped,
        "training_parameters": {"C": 1.0, "max_comparisons_per_query": 1000, "threads": 1},
        "limitations": [
            "Проверка на общем каталоге; переноса на новое время не доказано.",
            "Непоказанные вакансии не имеют пользовательской оценки.",
            "Свободный текст резюме не входит в эти признаки; используется подтверждённый профиль.",
        ],
    }
    artifact = ServingRankerArtifact.seal(
        created_at=datetime.now(UTC),
        feature_contract=feature_contract(),
        retrieval_parameters=RetrievalParameters(),
        model=model,
        provenance=provenance,
        validation=proof,
    )
    report = {
        "version": "serving-ranker-report-v1",
        "model_id": artifact.model_id,
        "feature_contract": artifact.feature_contract,
        "feature_names": model.feature_names,
        "training_comparisons": model.training_comparisons,
        "provenance": provenance,
        "validation": proof,
        "activation_blockers": activation_reasons(artifact),
        "elapsed_seconds": time.perf_counter() - started,
        "activated": False,
    }
    return artifact, report


def save_serving_ranker(artifact, report, directory):
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "model.json").write_text(
        artifact.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (directory / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# VA-06: эксперимент линейного ранжирования",
        "",
        f"Сравнений training: {report['training_comparisons']}",
        "Модель не включена в выдачу.",
        "",
        "| Режим оценки | NDCG@5 правил | NDCG@5 модели |",
        "| --- | --- | --- |",
    ]
    for name in ("all_candidates", "serving_threshold"):
        p = report["validation"][name]
        lines.append(f"| {name} | {p['baseline_ndcg']} | {p['model_ndcg']} |")
    lines += [
        "",
        "Причины запрета включения:",
        *["- " + reason for reason in report["activation_blockers"]],
        "",
        "Профессии без положительных примеров не получают оценку качества. "
        "Черновая разметка не является независимой приёмкой.",
    ]
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
