"""Explicit denominators, incomplete-pool bounds and seeded clustered uncertainty."""

import math
import random
import statistics

from jobs.evaluation_data import digest


def interval(values, *, seed=20261102, repeats=2000, confidence=0.95):
    """Inputs are one observation per independent owner cluster, never pair rows."""
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "low": None, "high": None, "clusters": 0}
    mean = statistics.mean(values)
    if len(values) < 2:
        return {"mean": mean, "low": None, "high": None, "clusters": len(values)}
    rng = random.Random(seed)
    samples = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(repeats))
    tail = (1 - confidence) / 2
    return {
        "mean": mean,
        "low": samples[int(tail * repeats)],
        "high": samples[min(repeats - 1, math.ceil((1 - tail) * repeats) - 1)],
        "clusters": len(values),
    }


def repeated_interval(groups, *, seed, repeats, confidence):
    """Resample owner clusters, then paired warm repetitions within each owner."""
    groups = list(groups.values())
    result = interval(
        [statistics.mean(g) for g in groups], seed=seed, repeats=repeats, confidence=confidence
    )
    result["paired_repetitions"] = sum(map(len, groups))
    if len(groups) < 2:
        return result
    rng = random.Random(seed)
    samples = sorted(
        statistics.mean(
            statistics.mean(rng.choices(g, k=len(g))) for g in rng.choices(groups, k=len(groups))
        )
        for _ in range(repeats)
    )
    tail = (1 - confidence) / 2
    result.update(
        low=samples[int(tail * repeats)],
        high=samples[min(repeats - 1, math.ceil((1 - tail) * repeats) - 1)],
    )
    return result


def wilson(numerator, denominator):
    if not denominator:
        return {
            "numerator": numerator,
            "denominator": denominator,
            "rate": None,
            "low": None,
            "high": None,
        }
    z, p = 1.959963984540054, numerator / denominator
    center = (p + z * z / (2 * denominator)) / (1 + z * z / denominator)
    width = (
        z
        * math.sqrt(p * (1 - p) / denominator + z * z / (4 * denominator**2))
        / (1 + z * z / denominator)
    )
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": p,
        "low": max(0, center - width),
        "high": min(1, center + width),
    }


def catalog_metrics(order, labels, all_ids, *, top=5):
    if top < 1 or len(set(all_ids)) != len(all_ids):
        raise ValueError("Invalid denominator/catalog")
    if len(order) != len(set(order)) or not set(order) <= set(all_ids):
        raise ValueError("Duplicate/unknown ranking IDs")
    if not set(labels) <= set(all_ids) or any(
        type(v) is not int or v not in range(4) for v in labels.values()
    ):
        raise ValueError("Invalid catalog judgments")
    shown = order[:top]
    positives = {key for key, grade in labels.items() if grade >= 2}
    found = len(set(shown) & positives)
    unknown = set(all_ids) - labels.keys()
    unknown_shown = len(set(shown) & unknown)
    complete = not unknown
    ideal = sorted(labels.values(), reverse=True)[:top]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    dcg = sum((2 ** labels.get(key, 0) - 1) / math.log2(i + 2) for i, key in enumerate(shown))
    lower_denominator = len(positives) + len(unknown - set(shown))
    upper_denominator = len(positives) + unknown_shown
    return {
        "ndcg_at_5": dcg / idcg if complete and idcg else None,
        "precision_at_5": found / top if complete else None,
        "precision_among_shown": found / len(shown) if complete and shown else None,
        "recall": found / len(positives) if complete and positives else None,
        "relevant_found": found,
        "relevant_catalog": len(positives),
        "precision_denominator": top,
        "shown_denominator": len(shown),
        "recall_denominator": len(positives) if complete else None,
        "idcg_denominator": idcg if complete else None,
        "catalog_groups": len(all_ids),
        "judged_groups": len(labels),
        "unjudged_groups": len(unknown),
        "unjudged_shown": unknown_shown,
        "judgment_coverage": len(labels) / len(all_ids) if all_ids else 0,
        "metric_scope": "full_catalog" if complete else "partial_pool_not_catalog_recall",
        "pooled_recall": found / len(positives) if positives else None,
        "recall_lower_bound": found / lower_denominator if lower_denominator else None,
        "recall_upper_bound": (found + unknown_shown) / upper_denominator
        if upper_denominator
        else None,
        "precision_at_5_lower_bound": found / top,
        "precision_at_5_upper_bound": (found + unknown_shown) / top,
    }


def fact_key(fact):
    value = fact["value"]
    if isinstance(value, str):
        from nlp.extraction import skill_key

        value = skill_key(value) if fact["field"] == "skill" else value.casefold().strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = float(value)
    return digest(
        {
            k: (value if k == "value" else fact.get(k))
            for k in ("field", "value", "status", "importance", "alternative_group")
        }
    )


def normalize_fact_groups(facts):
    groups = {}
    for fact in facts:
        group = fact.get("alternative_group")
        if group:
            groups.setdefault(group, []).append(fact_key(fact | {"alternative_group": None}))
    return [
        fact
        | {
            "alternative_group": digest(sorted(set(groups[fact["alternative_group"]])))
            if fact.get("alternative_group")
            else None
        }
        for fact in facts
    ]


def fact_metrics(predicted, gold):
    predicted, gold = normalize_fact_groups(predicted), normalize_fact_groups(gold)
    known = [f for f in gold if f.get("status", "known") == "known"]
    expected = {fact_key(f) for f in known}
    actual = {fact_key(f) for f in predicted}
    found = len(actual & expected)
    fields = sorted({f["field"] for f in known} | {f["field"] for f in predicted})
    return {
        "fact_recall": found / len(expected) if expected else None,
        "fact_precision": found / len(actual) if actual else None,
        "found": found,
        "expected": len(expected),
        "predicted": len(actual),
        "unknown_gold": len(gold) - len(known),
        "by_field": {
            field: fact_metrics(
                [f for f in predicted if f["field"] == field],
                [f for f in gold if f["field"] == field],
            )
            for field in fields
        }
        if len(fields) > 1
        else {},
    }
