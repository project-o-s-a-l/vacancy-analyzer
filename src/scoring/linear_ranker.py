"""Learn pairwise job preferences from explicit judgments, without pretrained weights."""

import json
import random
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from math import fsum, isfinite, log1p, sqrt

from domain.matching_dataset import MatchingDataset, PairFeatures
from domain.ranker import LinearRankerModel
from scoring.text_features import (
    FIXED_TEXT_FEATURES_VERSION,
    TEXT_FEATURES_VERSION,
    TextFeaturesUnavailable,
)

COMPONENTS = ("skills", "experience", "title", "salary", "preferences", "freshness")
SIMILARITIES = ("word_cosine", "char_cosine")
COUNTS = ("matched_skills", "missing_required", "missing_preferred", "missing_unspecified")
FEATURE_NAMES = [
    name for field in (*COMPONENTS, *SIMILARITIES) for name in (field, field + "_missing")
] + ["baseline_coverage", *("log1p_" + field for field in COUNTS)]
MAX_COMPARISONS = 1000


@dataclass(frozen=True)
class LabeledQuery:
    dataset: MatchingDataset
    labels: dict[str, int]
    group: str


def profile_id(dataset: MatchingDataset) -> str:
    value = json.dumps(dataset.profile.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    return sha256(value.encode()).hexdigest()


def comparable_spec(dataset: MatchingDataset) -> dict:
    if dataset.feature_spec.get("version") not in {
        TEXT_FEATURES_VERSION,
        FIXED_TEXT_FEATURES_VERSION,
    }:
        raise ValueError("Неподдерживаемая версия текстовых признаков.")
    # Each search corpus has its own unsupervised IDF; parameters must remain the same.
    return {key: value for key, value in dataset.feature_spec.items() if key != "corpus_text_hash"}


def feature_vector(features: PairFeatures) -> list[float]:
    if set(features.baseline_components) != set(COMPONENTS):
        raise ValueError("Неизвестный состав компонентов baseline.")
    values = []
    for field in (*COMPONENTS, *SIMILARITIES):
        value = (
            features.baseline_components[field] if field in COMPONENTS else getattr(features, field)
        )
        if value is not None and not 0 <= value <= 1:
            raise ValueError("Компоненты сравнения должны быть в диапазоне 0..1.")
        values.extend([float(value or 0), float(value is None)])
    values.append(features.baseline_coverage)
    values.extend(log1p(getattr(features, field)) for field in COUNTS)
    return values


def preferred_pairs(example: LabeledQuery) -> list[tuple[int, int]]:
    pairs = sorted(example.dataset.pairs, key=lambda pair: pair.pair_id)
    expected = {pair.pair_id for pair in pairs}
    if len(expected) != len(pairs) or set(example.labels) != expected:
        raise ValueError("Для обучения нужна полная разметка без повторов.")
    if any(type(grade) is not int or grade not in range(4) for grade in example.labels.values()):
        raise ValueError("Оценки должны быть целыми числами 0..3.")
    grades = {
        grade: [index for index, pair in enumerate(pairs) if example.labels[pair.pair_id] == grade]
        for grade in range(4)
    }
    blocks = [
        (grades[high], grades[low])
        for high in range(4)
        for low in range(high)
        if grades[high] and grades[low]
    ]
    total = sum(len(high) * len(low) for high, low in blocks)
    if not total:
        raise ValueError("В каждом обучающем запросе нужны разные оценки вакансий.")
    rng = random.Random(example.dataset.query_id)
    selected = sorted(rng.sample(range(total), min(MAX_COMPARISONS, total)))
    comparisons = []
    for offset in selected:
        for high, low in blocks:
            block_size = len(high) * len(low)
            if offset < block_size:
                comparisons.append((high[offset // len(low)], low[offset % len(low)]))
                break
            offset -= block_size
    return comparisons


def fit_pairwise_ranker(examples: list[LabeledQuery]) -> LinearRankerModel:
    if len({example.group for example in examples}) < 2:
        raise ValueError("Для обучения нужны как минимум две разные группы резюме.")
    if len({example.dataset.query_id for example in examples}) != len(examples):
        raise ValueError("Обучающий запрос повторяется.")
    spec = comparable_spec(examples[0].dataset)
    if any(comparable_spec(example.dataset) != spec for example in examples):
        raise ValueError("Нельзя смешивать разные параметры или версии текстовых признаков.")
    comparisons = [preferred_pairs(example) for example in examples]
    try:
        import numpy as np
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        raise TextFeaturesUnavailable(
            'Для обучения установите дополнение: python -m pip install -e ".[ml]"'
        ) from None

    group_counts = Counter(example.group for example in examples)
    rows, row_weights = [], []
    for example in examples:
        pairs = sorted(example.dataset.pairs, key=lambda pair: pair.pair_id)
        rows.extend(feature_vector(pair.features) for pair in pairs)
        row_weights.extend([1 / (group_counts[example.group] * len(pairs))] * len(pairs))
    # Two-pass weighted variance avoids a negative variance for constant columns.
    total_weight = fsum(row_weights)
    means, scales = [], []
    for column in zip(*rows, strict=True):
        mean = fsum(weight * value for weight, value in zip(row_weights, column, strict=True))
        mean /= total_weight
        variance = (
            fsum(
                weight * (value - mean) ** 2
                for weight, value in zip(row_weights, column, strict=True)
            )
            / total_weight
        )
        means.append(mean)
        scales.append(sqrt(variance) if variance > 0 and min(column) != max(column) else 1.0)
    differences, targets, weights = [], [], []
    total_rows = 2 * sum(len(pairs) for pairs in comparisons)
    for example, preferences in zip(examples, comparisons, strict=True):
        pairs = sorted(example.dataset.pairs, key=lambda pair: pair.pair_id)
        transformed = np.asarray(
            [
                [
                    (value - mean) / scale
                    for value, mean, scale in zip(
                        feature_vector(pair.features), means, scales, strict=True
                    )
                ]
                for pair in pairs
            ]
        )
        weight = total_rows / (
            2 * len(preferences) * group_counts[example.group] * len(group_counts)
        )
        for high, low in preferences:
            difference = transformed[high] - transformed[low]
            differences.extend([difference, -difference])
            targets.extend([1, 0])
            weights.extend([weight, weight])
    if not np.any(differences):
        raise ValueError("Признаки не различают вакансии с разными ручными оценками.")
    estimator = LogisticRegression(
        fit_intercept=False, solver="lbfgs", C=1.0, max_iter=1000, tol=1e-8
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            estimator.fit(differences, targets, sample_weight=weights)
        except ConvergenceWarning:
            raise ValueError("Обучение не сошлось; модель не сохранена.") from None
    return LinearRankerModel(
        trained_at=datetime.now(UTC),
        feature_names=FEATURE_NAMES,
        means=means,
        scales=scales,
        coefficients=estimator.coef_[0].tolist(),
        text_feature_spec=spec,
        training_query_ids=sorted(example.dataset.query_id for example in examples),
        training_profile_ids=sorted({profile_id(example.dataset) for example in examples}),
        training_groups=sorted(group_counts),
        training_comparisons=total_rows // 2,
        library_version=version("scikit-learn"),
    )


def ranker_scores(model: LinearRankerModel, dataset: MatchingDataset) -> dict[str, float]:
    if model.feature_names != FEATURE_NAMES or model.text_feature_spec != comparable_spec(dataset):
        raise ValueError("Модель несовместима с составом или версией признаков набора.")
    scores = {}
    for pair in dataset.pairs:
        vector = feature_vector(pair.features)
        score = sum(
            (value - mean) / scale * coefficient
            for value, mean, scale, coefficient in zip(
                vector, model.means, model.scales, model.coefficients, strict=True
            )
        )
        if not isfinite(score):
            raise ValueError("Получено нечисловое значение ранжирования.")
        scores[pair.pair_id] = score
    return scores
