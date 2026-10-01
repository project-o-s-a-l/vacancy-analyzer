"""Compare frozen rankings using complete, explicit human relevance judgments."""

import csv
from math import isfinite, log2
from pathlib import Path

from domain.matching_dataset import MatchingDataset


def ranking_metrics(ranking: list[str], labels: dict[str, int], *, top: int = 5) -> dict:
    """Full-catalog denominator, including relevant items absent from retrieval."""
    if top < 1 or len(set(ranking)) != len(ranking) or not set(ranking) <= labels.keys():
        raise ValueError("Некорректный порядок или идентификаторы ранжирования")
    if any(type(grade) is not int or grade not in range(4) for grade in labels.values()):
        raise ValueError("Оценки должны быть целыми числами 0..3")
    grades = [labels[key] for key in ranking[:top]]
    ideal = sorted(labels.values(), reverse=True)[:top]
    ideal_dcg = sum((2**grade - 1) / log2(rank + 2) for rank, grade in enumerate(ideal))
    dcg = sum((2**grade - 1) / log2(rank + 2) for rank, grade in enumerate(grades))
    relevant = sum(grade >= 2 for grade in labels.values())
    found = sum(grade >= 2 for grade in grades)
    return {
        "ndcg_at_k": dcg / ideal_dcg if ideal_dcg else None,
        "precision_at_k": found / top,
        "precision_among_shown": found / len(grades) if grades else None,
        "recall_at_k": found / relevant if relevant else None,
        "relevant_catalog": relevant,
        "shown": len(grades),
    }


def load_judgments(
    path: Path, dataset: MatchingDataset, *, require_complete: bool = True
) -> dict[str, int]:
    expected = {pair.pair_id for pair in dataset.pairs}
    if len(expected) != len(dataset.pairs):
        raise ValueError("В наборе повторяются pair_id.")
    labels = {}
    seen = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"query_id", "pair_id", "relevance"}.issubset(reader.fieldnames or []):
            raise ValueError("В разметке нужны query_id, pair_id и relevance.")
        for row in reader:
            pair_id = row["pair_id"]
            if row["query_id"] != dataset.query_id or pair_id not in expected:
                raise ValueError("Разметка относится к другому запросу или снимку вакансии.")
            if pair_id in seen:
                raise ValueError("В разметке повторяется pair_id.")
            seen.add(pair_id)
            value = (row["relevance"] or "").strip()
            if not value:
                continue
            if value not in {"0", "1", "2", "3"}:
                raise ValueError("relevance должна быть целым числом 0..3 или пустой ячейкой.")
            labels[pair_id] = int(value)
    if require_complete and len(labels) != len(expected):
        raise ValueError(
            f"Оценено {len(labels)} из {len(expected)} пар. "
            "Заполните все оценки: пропуски не считаются отказами."
        )
    return labels


def evaluate_matching(
    dataset: MatchingDataset,
    labels: dict[str, int],
    *,
    top: int = 5,
    learned_scores: dict[str, float] | None = None,
) -> dict:
    if top < 1:
        raise ValueError("top must be positive")
    expected = {pair.pair_id for pair in dataset.pairs}
    if len(expected) != len(dataset.pairs) or set(labels) != expected:
        raise ValueError("Нужна полная разметка без повторяющихся пар.")
    if any(type(value) is not int or not 0 <= value <= 3 for value in labels.values()):
        raise ValueError("Оценки должны быть целыми числами 0..3.")
    k = min(top, len(dataset.pairs))
    ideal = sorted(labels.values(), reverse=True)[:k]
    ideal_dcg = sum((2**grade - 1) / log2(rank + 2) for rank, grade in enumerate(ideal))
    relevant = sum(grade >= 2 for grade in labels.values())
    rankings = {
        "baseline": sorted(dataset.pairs, key=lambda pair: pair.baseline_rank),
    }
    if learned_scores is not None:
        if set(learned_scores) != expected or not all(
            isfinite(value) for value in learned_scores.values()
        ):
            raise ValueError("Нужны конечные оценки модели для всех пар набора.")
        rankings["trained_linear"] = sorted(
            dataset.pairs, key=lambda pair: (-learned_scores[pair.pair_id], pair.pair_id)
        )
    for channel in ("word_cosine", "char_cosine"):
        rankings[channel] = sorted(
            dataset.pairs,
            key=lambda pair, channel=channel: (
                getattr(pair.features, channel) is None,
                -(getattr(pair.features, channel) or 0),
                pair.pair_id,
            ),
        )
    methods = {}
    for method, ranking in rankings.items():
        unavailable = (
            sum(getattr(pair.features, method) is None for pair in ranking)
            if method in {"word_cosine", "char_cosine"}
            else 0
        )
        if unavailable == len(ranking):
            methods[method] = {
                "ndcg_at_k": None,
                "recall_at_k": None,
                "reciprocal_rank_at_k": None,
                "unavailable_features": unavailable,
                "top_pair_ids": [],
            }
            continue
        grades = [labels[pair.pair_id] for pair in ranking[:k]]
        dcg = sum((2**grade - 1) / log2(rank + 2) for rank, grade in enumerate(grades))
        first_relevant = next((rank + 1 for rank, grade in enumerate(grades) if grade >= 2), None)
        methods[method] = {
            "ndcg_at_k": dcg / ideal_dcg if ideal_dcg else None,
            "recall_at_k": sum(grade >= 2 for grade in grades) / relevant if relevant else None,
            "reciprocal_rank_at_k": (
                (1 / first_relevant if first_relevant else 0.0) if relevant else None
            ),
            "unavailable_features": unavailable,
            "top_pair_ids": [pair.pair_id for pair in ranking[:k]],
        }
    return {
        "query_id": dataset.query_id,
        "judged_pairs": len(labels),
        "relevant_pairs": relevant,
        "k": k,
        "methods": methods,
        "limitations": [
            "Сравнение относится только к этому запросу и сохранённому корпусу.",
            "Эта команда не обучает модель; независимость проверки нужно обеспечивать отдельно.",
            "TF-IDF не учитывает смысл отрицаний, синонимы и достаточность квалификаций.",
        ],
    }
