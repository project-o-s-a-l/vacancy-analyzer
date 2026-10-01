"""Evidence gate for optional semantic retrieval; no model, DB or working user data."""

import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path

from domain.retrieval import RetrievalParameters
from domain.vacancy_analysis import is_fresh
from domain.vacancy_duplicates import vacancy_duplicate_key
from jobs.benchmark_data import load_bundle, vacancy_key
from jobs.candidates import retrieve_candidates
from jobs.retrieval_evaluation import MemoryCandidateSource
from scoring import rank_vacancies
from scoring.matching_evaluation import ranking_metrics
from scoring.retrieval import (
    lexical_evidence,
    profile_query,
    retrieval_eligibility,
    search_document,
)
from scoring.serving_features import feature_contract
from storage.payloads import payload_hash

# Engineering comparison targets, not independently accepted quality guarantees.
CRITERIA = {
    "minimum_absolute_eligible_recall_gain": 0.02,
    "minimum_absolute_ndcg_at_5_gain": 0.01,
    "minimum_validation_professions_with_positives": 5,
    "maximum_required_condition_violations": 0,
    "profession_regressions_allowed": False,
    "human_confirmation_required": True,
    "confirmed_server_budget_required": True,
}


def _summary(cases):
    eligible = sum(c["eligible_relevant"] for c in cases)
    recalled = sum(c["recalled_or_equivalent"] for c in cases)
    ndcg = [
        c["serving_metrics"]["ndcg_at_k"]
        for c in cases
        if c["serving_metrics"]["ndcg_at_k"] is not None
    ]
    stages = Counter(stage for c in cases for stage in c["positive_stages"])
    return {
        "profiles": len(cases),
        "positive_pairs": sum(c["relevant"] for c in cases),
        "eligible_positive_pairs": eligible,
        "retrieved_or_duplicate_equivalent": recalled,
        "eligible_recall": recalled / eligible if eligible else None,
        "positive_stages": dict(sorted(stages.items())),
        "serving_macro_ndcg_at_5": statistics.mean(ndcg) if ndcg else None,
    }


def _implementation_hash():
    return payload_hash(
        {
            "features": feature_contract(),
            "selector": hashlib.sha256(
                (Path(__file__).parent / "candidates.py").read_bytes()
            ).hexdigest(),
            "audit": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )


async def audit_semantic_need(bundle: Path, *, k=64, min_score=50):
    params = RetrievalParameters(k=k)
    if not 0 <= min_score <= 100:
        raise ValueError("Порог должен быть от 0 до 100")
    catalog, cases, judgments, manifest = load_bundle(bundle)
    if any(row.relevance is None for row in judgments):
        raise ValueError("Для аудита нужна полная явная разметка; пропуски не являются отказами")
    labels = {c.id: {} for c in cases.cases}
    origins = Counter(row.origin for row in judgments)
    for row in judgments:
        labels[row.case_id][row.vacancy_key] = row
    source = MemoryCandidateSource(catalog.vacancies)
    now = catalog.reference_time
    by_key = {vacancy_key(v): v for v in catalog.vacancies}
    entries, positives = [], []
    for case in cases.cases:
        started = time.perf_counter()
        selection = await retrieve_candidates(
            None, case.profile, parameters=params, now=now, candidate_source=source
        )
        ranking = rank_vacancies(selection.items, case.profile, now=now)
        position = {vacancy_key(v): i for i, (v, _) in enumerate(ranking, 1)}
        scores = {vacancy_key(v): m.score for v, m in ranking}
        admitted = [(v, m) for v, m in ranking if m.score >= min_score]
        shown = {vacancy_key(v) for v, _ in admitted[:5]}
        chosen = {vacancy_key(v) for v in selection.items}
        chosen_groups = {vacancy_duplicate_key(v) or str(v.id) for v in selection.items}
        query = profile_query(case.profile)
        case_rows = []
        for key, label in sorted(labels[case.id].items()):
            if label.relevance < 2:
                continue
            vacancy = by_key[key]
            eligibility = retrieval_eligibility(vacancy, case.profile, now)
            active = is_fresh(vacancy, now=now)
            evidence = lexical_evidence(search_document(vacancy.model_dump()), query, params)
            if not active:
                stage = "inactive"
            elif not eligibility.eligible:
                stage = "required_conditions"
            elif key not in chosen:
                if (vacancy_duplicate_key(vacancy) or str(vacancy.id)) in chosen_groups:
                    stage = "duplicate_equivalent"
                else:
                    stage = (
                        "retrieval_no_overlap"
                        if not evidence.lexical_score
                        else "retrieval_with_overlap"
                    )
            elif scores[key] < min_score:
                stage = "score_threshold"
            elif key not in shown:
                stage = "ranking_below_top5"
            else:
                stage = "shown"
            item = {
                "case_id": case.id,
                "profession": case.profession,
                "split": case.split,
                "vacancy_key": key,
                "relevance": label.relevance,
                "origin": label.origin,
                "stage": stage,
                "active": active,
                "eligible": eligibility.eligible,
                "exclusion_codes": [issue.code for issue in eligibility.exclusions],
                "lexical_score": evidence.lexical_score,
                "baseline_score": scores.get(key),
                "baseline_rank": position.get(key),
            }
            case_rows.append(item)
        eligible = [r for r in case_rows if r["active"] and r["eligible"]]
        recalled = [
            r
            for r in eligible
            if r["stage"] not in {"retrieval_no_overlap", "retrieval_with_overlap"}
        ]
        metrics = ranking_metrics(
            [vacancy_key(v) for v, _ in admitted],
            {key: label.relevance for key, label in labels[case.id].items()},
            top=5,
        )
        entries.append(
            {
                "case_id": case.id,
                "profession": case.profession,
                "split": case.split,
                "selected": len(selection.items),
                "relevant": len(case_rows),
                "eligible_relevant": len(eligible),
                "recalled_or_equivalent": len(recalled),
                "eligible_recall": len(recalled) / len(eligible) if eligible else None,
                "positive_stages": [r["stage"] for r in case_rows],
                "serving_metrics": metrics,
                "seconds_selection_ranking_audit": time.perf_counter() - started,
            }
        )
        positives.extend(case_rows)
    validation = [c for c in entries if c["split"] == "validation"]
    validation_professions = {c["profession"] for c in validation}
    uncovered = sorted(
        p
        for p in validation_professions
        if not any(c["eligible_relevant"] for c in validation if c["profession"] == p)
    )
    missing = [
        r for r in positives if r["stage"] in {"retrieval_no_overlap", "retrieval_with_overlap"}
    ]
    # A zero-score miss is a review candidate, not proof of semantic causation or model benefit.
    reasons = []
    if origins != {"human": len(judgments)}:
        reasons.append("labels_not_independently_confirmed")
    if not missing:
        reasons.append("no_eligible_retrieval_losses_in_this_catalogue")
    if (
        uncovered
        or len(validation_professions) - len(uncovered)
        < CRITERIA["minimum_validation_professions_with_positives"]
    ):
        reasons.append("validation_positive_coverage_missing")
    reasons.append("server_budget_not_confirmed_by_this_audit")
    return {
        "version": "semantic-need-audit-v1",
        "catalogue_rows": len(catalog.vacancies),
        "catalogue_rows_with_snapshot_analysis": sum(
            v.analysis is not None for v in catalog.vacancies
        ),
        "reference_time": now.isoformat(),
        "parameters": params.model_dump(),
        "min_score": min_score,
        "top": 5,
        "manifest_hash": payload_hash(manifest),
        "labels_hash": hashlib.sha256((bundle / "labels.jsonl").read_bytes()).hexdigest(),
        "implementation_hash": _implementation_hash(),
        "label_origins": dict(origins),
        "human_quality_ready": not any(row.origin != "human" for row in judgments),
        "criteria": CRITERIA.copy(),
        "criteria_status": "engineering targets for a future comparison; not user acceptance",
        "all": _summary(entries),
        "training": _summary([c for c in entries if c["split"] == "training"]),
        "validation": _summary(validation),
        "validation_without_eligible_positives": uncovered,
        "by_profession": {
            p: _summary([c for c in entries if c["profession"] == p])
            for p in sorted({c["profession"] for c in entries})
        },
        "cases": entries,
        "positive_pairs": positives,
        "retrieval_misses_for_review": missing,
        "decision": "review_retrieval_losses" if missing else "defer_semantic_runtime",
        "reasons": reasons,
        "runtime_changed": False,
        "external_calls": 0,
        "scope": "Fixed catalogue and any analysis already in its snapshots, same VA-05 "
        "selector and rule scorer; no model comparison, history, live analysis, "
        "memory benchmark or live-catalogue guarantee.",
    }


def write_semantic_audit(report, output: Path):
    markdown = output.with_suffix(".md")
    if output == markdown or output.exists() or markdown.exists():
        raise ValueError("Выберите новые JSON и Markdown файлы отчёта")
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# VA-08: проверка необходимости смыслового отбора",
        "",
        f"Решение: `{report['decision']}`. Рабочая выдача не изменена.",
        "",
        "| Часть | Подходящих допустимых пар | Recall отбора | Потери на пороге |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in ("training", "validation", "all"):
        group = report[name]
        lines.append(
            f"| {name} | {group['eligible_positive_pairs']} | {group['eligible_recall']} | "
            f"{group['positive_stages'].get('score_threshold', 0)} |"
        )
    lines.extend(
        [
            "",
            "Отсутствие потерь на этом срезе не доказывает универсальную полноту.",
            "Профессии validation без положительных примеров: "
            + ", ".join(report["validation_without_eligible_positives"]),
            "Происхождение оценок: " + json.dumps(report["label_origins"], ensure_ascii=False),
            "",
            "Потеря ниже порога или Top-5 относится к ранжированию; дополнительный "
            "канал отбора сам по себе её не исправляет. Пары без совпадений требуют "
            "проверки человеком; это ещё не измеренный эффект языковой модели.",
            "",
            "## Причины и ограничения",
            "",
            *["- `" + reason + "`" for reason in report["reasons"]],
            "",
            report["scope"],
        ]
    )
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with markdown.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
