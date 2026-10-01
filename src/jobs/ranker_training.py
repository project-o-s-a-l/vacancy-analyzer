"""Explicit train/validation manifests and local, reviewable ranking artifacts."""

import json
from hashlib import sha256
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from domain.matching_dataset import MatchingDataset
from domain.ranker import LinearRankerModel
from scoring.linear_ranker import LabeledQuery, fit_pairwise_ranker, profile_id, ranker_scores
from scoring.matching_evaluation import evaluate_matching, load_judgments


class QueryFiles(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: Path
    labels: Path
    group: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")


class TrainingManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    training: list[QueryFiles] = Field(min_length=2)
    validation: list[QueryFiles] = Field(min_length=1)


def load_training_manifest(path: Path) -> tuple[list[LabeledQuery], list[LabeledQuery], list[dict]]:
    manifest = TrainingManifest.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    groups = []
    audit = []
    for split, entries in (("training", manifest.training), ("validation", manifest.validation)):
        queries = []
        for entry in entries:
            dataset_path = path.parent / entry.dataset
            labels_path = path.parent / entry.labels
            dataset_bytes = dataset_path.read_bytes()
            dataset = MatchingDataset.model_validate_json(dataset_bytes)
            labels = load_judgments(labels_path, dataset)
            queries.append(LabeledQuery(dataset, labels, entry.group))
            audit.append(
                {
                    "split": split,
                    "group": entry.group,
                    "query_id": dataset.query_id,
                    "dataset_sha256": sha256(dataset_bytes).hexdigest(),
                    "labels_sha256": sha256(labels_path.read_bytes()).hexdigest(),
                    "judged_pairs": len(labels),
                }
            )
        groups.append(queries)
    return groups[0], groups[1], audit


def validate_split(training: list[LabeledQuery], validation: list[LabeledQuery]) -> None:
    if not training or not validation:
        raise ValueError("Нужны отдельные обучающие и проверочные запросы.")
    if {query.group for query in training} & {query.group for query in validation}:
        raise ValueError("Одна группа резюме попала и в обучение, и в проверку.")
    if {query.dataset.query_id for query in training} & {
        query.dataset.query_id for query in validation
    }:
        raise ValueError("Один запрос попал и в обучение, и в проверку.")
    if {profile_id(query.dataset) for query in training} & {
        profile_id(query.dataset) for query in validation
    }:
        raise ValueError("Одинаковый профиль обнаружен в обучении и проверке.")
    if len({query.dataset.query_id for query in validation}) != len(validation):
        raise ValueError("Проверочный запрос повторяется.")
    if not any(len(set(query.labels.values())) > 1 for query in validation):
        raise ValueError("Для проверки нужны вакансии с разными ручными оценками.")


def train_from_manifest(path: Path, *, top: int = 5) -> tuple[LinearRankerModel, dict]:
    if top < 1:
        raise ValueError("top must be positive")
    training, validation, audit = load_training_manifest(path)
    validate_split(training, validation)
    model = fit_pairwise_ranker(training)
    checks = [
        evaluate_matching(
            query.dataset, query.labels, top=top, learned_scores=ranker_scores(model, query.dataset)
        )
        for query in validation
    ]
    averages = {}
    for method in checks[0]["methods"]:
        averages[method] = {}
        for metric in ("ndcg_at_k", "recall_at_k", "reciprocal_rank_at_k"):
            values = [
                check["methods"][method][metric]
                for check in checks
                if check["methods"][method][metric] is not None
            ]
            averages[method][metric] = {
                "mean": sum(values) / len(values) if values else None,
                "queries": len(values),
            }
    training_jobs = {
        (pair.vacancy.source, pair.vacancy.source_id)
        for query in training
        for pair in query.dataset.pairs
    }
    validation_jobs = {
        (pair.vacancy.source, pair.vacancy.source_id)
        for query in validation
        for pair in query.dataset.pairs
    }
    return model, {
        "model_version": model.version,
        "training_groups": len({query.group for query in training}),
        "validation_groups": len({query.group for query in validation}),
        "training_comparisons": model.training_comparisons,
        "validation_query_count": len(checks),
        "macro_averages": averages,
        "validation": checks,
        "inputs": audit,
        "shared_vacancies_across_splits": len(training_jobs & validation_jobs),
        "limitations": [
            "Проверка отделена по указанным группам и точным профилям, но не по времени.",
            "Версии резюме одного человека нужно обозначать одной группой в manifest.",
            "Каталог вакансий может пересекаться; это не проверка на полностью новых вакансиях.",
            "TF-IDF считается по каталогу запроса без оценок; это не общий обученный словарь.",
            "Параметры и веса не подбираются по проверочной части.",
            "Небольшая ручная выборка не доказывает качество для всех профессий.",
            "Результат не включает модель в пользовательскую выдачу автоматически.",
        ],
    }


def write_training_result(model: LinearRankerModel, report: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "model.json").write_text(model.model_dump_json(indent=2), encoding="utf-8")
    (directory / "validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
