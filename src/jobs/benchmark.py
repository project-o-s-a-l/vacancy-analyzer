"""Reproducible baseline comparison on complete frozen catalogs and explicit labels."""

import json
import platform
import statistics
import sys
import time
import tracemalloc
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

from domain.benchmark import BenchmarkCases, BenchmarkCatalog
from domain.vacancy_analysis import is_fresh
from jobs.benchmark_data import (
    blank_judgments,
    digest,
    judgments_text,
    load_bundle,
    load_cases,
    snapshot_html,
    vacancy_key,
    validate_cases,
)
from nlp.titles import title_tokens
from scoring import rank_vacancies
from scoring.matching_evaluation import ranking_metrics


def implementation_hash() -> str:
    root = Path(__file__).parents[1]
    sources = {
        path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest()
        for module in ("domain", "scoring", "nlp", "processing")
        for path in sorted((root / module).glob("*.py"))
    }
    for name in ("benchmark.py", "benchmark_data.py", "rank.py"):
        sources["jobs/" + name] = sha256((root / "jobs" / name).read_bytes()).hexdigest()
    return digest(sources)


def run_baseline(catalog: BenchmarkCatalog, cases: BenchmarkCases, *, limit=500, min_score=50):
    if not 1 <= limit <= 5000 or not 0 <= min_score <= 100:
        raise ValueError("Некорректный лимит или порог")
    # Frozen legacy comparator, before candidate filtering: the date limit precedes admissibility.
    # Current retrieval is evaluated separately by jobs.retrieval_evaluation.
    selected = sorted(
        catalog.vacancies,
        key=lambda v: (
            -v.published_at.timestamp(),
            v.source_id,
        ),
    )[:limit]
    active = [v for v in selected if is_fresh(v, now=catalog.reference_time)]
    result = {}
    for case in cases.cases:
        ranking = rank_vacancies(active, case.profile, now=catalog.reference_time)
        result[case.id] = {
            "retrieved": [vacancy_key(v) for v in active],
            "eligible": [vacancy_key(v) for v, _ in ranking],
            "displayed": [vacancy_key(v) for v, match in ranking if match.score >= min_score],
            "scores": {vacancy_key(v): match.score for v, match in ranking},
        }
    return result


def prepare_benchmark(
    html_directory: Path,
    cases_path: Path,
    output: Path,
    *,
    now,
    limit: int = 200,
) -> dict:
    if output.exists():
        raise ValueError("Выберите новый каталог: существующая разметка не перезаписывается")
    cases = load_cases(cases_path)
    catalog = snapshot_html(html_directory, now=now, limit=limit)
    validate_cases(cases)
    baseline, resources = measure_baseline(catalog, cases, repeats=1, limit=500, min_score=50)
    output.mkdir(parents=True, exist_ok=False)
    files = {
        "catalog.json": catalog.model_dump(mode="json"),
        "cases.json": cases.model_dump(mode="json"),
        "baseline.json": baseline,
        "stand.json": resources,
    }
    for name, value in files.items():
        (output / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    rows = blank_judgments(catalog, cases)
    (output / "labels.jsonl").write_text(judgments_text(rows), encoding="utf-8")
    manifest = {
        "version": "matching-benchmark-v1",
        "reference_time": now.isoformat(),
        "sha256": {name: sha256((output / name).read_bytes()).hexdigest() for name in files},
        "baseline_implementation_sha256": implementation_hash(),
        "eligibility_version": "eligibility-v1",
        "candidate_limit": 500,
        "min_score": 50,
        "top": 5,
        "professions": sorted({case.profession for case in cases.cases}),
        "resume_source": "synthetic",
        "catalog_source": "saved_hh_html_without_llm_analysis",
        "vacancy_split": "shared_deduplicated_catalog",
        "pair_count": len(rows),
        "acceptance_criteria": {
            "minimum_validation_professions": 5,
            "minimum_relevant_per_validation_case": 1,
            "maximum_constraint_violations": 0,
            "quality_regression_tolerance": 0,
            "latency_ratio": 2.0,
            "latency_slack_seconds": 0.05,
            "python_peak_growth_bytes": 64 * 1024 * 1024,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # Scores are only in baseline.json; the labeling interface never reads them.
    (output / "REVIEW.md").write_text(
        "# Разметка подбора\n\n"
        "Все профили вымышлены. cases.json — резюме и подтверждённые поля; "
        "catalog.json — исходные сведения вакансий. Не смотрите baseline.json до разметки.\n\n"
        "0 — не подходит; 1 — частично, значимые пробелы; 2 — подходит; "
        "3 — особенно хорошо. Неизвестное не означает отсутствующее. "
        "Нарушение обязательного условия: relevance=0, violates_required=true, "
        "причина и цитата. Неоднозначную пару оставьте без оценки.\n\n"
        "Используйте benchmark-label с именем проверяющего. Машинный черновик можно "
        "сохранить с origin=assistant_draft; он не считается проверкой человеком. "
        "Официальные показатели появляются только после подтверждения всего каталога "
        "для конкретного профиля. Подробности: docs/BENCHMARK.md.\n",
        encoding="utf-8",
    )
    return manifest


def _average(records: list[dict]) -> dict:
    keys = ("ndcg_at_k", "precision_at_k", "recall_at_k", "violation_rate")
    result = {}
    for metric in keys:
        values = [r[metric] for r in records if r.get(metric) is not None]
        result[metric] = {"mean": statistics.mean(values) if values else None, "cases": len(values)}
    return result


def process_memory() -> dict:
    """Best-effort OS process memory; no services or additional dependencies."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "peak_working_set",
                    "working_set",
                    "peak_paged",
                    "paged",
                    "peak_nonpaged",
                    "nonpaged",
                    "pagefile",
                    "peak_pagefile",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        if psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return {
                "working_set_bytes": counters.working_set,
                "process_peak_working_set_bytes": counters.peak_working_set,
            }
    return {"working_set_bytes": None, "process_peak_working_set_bytes": None}


def measure_baseline(catalog, cases, *, repeats: int, limit: int, min_score: float):
    if not 1 <= repeats <= 5:
        raise ValueError("Повторы измерения: 1..5")
    run_baseline(catalog, cases, limit=limit, min_score=min_score)  # One warm-up, no I/O.
    durations, result = [], None
    for _ in range(repeats):
        start = time.perf_counter()
        result = run_baseline(catalog, cases, limit=limit, min_score=min_score)
        durations.append(time.perf_counter() - start)
    tracing = tracemalloc.is_tracing()
    if not tracing:
        tracemalloc.start()
    try:
        run_baseline(catalog, cases, limit=limit, min_score=min_score)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if not tracing:
            tracemalloc.stop()
    return result, {
        "repeats": repeats,
        "warmup_runs": 1,
        "batch_seconds": durations,
        "median_seconds_per_profile": statistics.median(durations) / len(cases.cases),
        "peak_python_allocated_bytes": peak,
        "memory_scope": "tracemalloc allocations, separate run; not RSS or server memory",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": {name: version(name) for name in ("pydantic", "pymorphy3")},
        "os_memory": process_memory(),
    }


def acceptance(case_reports: list[dict], current: dict, reference: dict, criteria: dict) -> dict:
    validation = [r for r in case_reports if r["split"] == "validation"]
    pending = []
    failures = []
    if any(not r["complete_human_catalog"] for r in case_reports):
        pending.append("Не весь каталог подтверждён человеком для каждого профиля")
    if len({r["profession"] for r in validation}) < criteria["minimum_validation_professions"]:
        pending.append("Недостаточно профессий в validation")
    for record in case_reports:
        result = record["methods"]["current_rules"]
        if len(result["human_constraint_violations"]) > criteria["maximum_constraint_violations"]:
            failures.append(f"Нарушены обязательные условия: {record['case_id']}")
    for record in validation:
        if not record["complete_human_catalog"]:
            continue
        result = record["methods"]["current_rules"]
        baseline = record["methods"]["frozen_baseline"]
        if result["relevant_catalog"] < criteria["minimum_relevant_per_validation_case"]:
            pending.append(f"Нет релевантных вакансий для {record['case_id']}")
        for metric in ("ndcg_at_k", "precision_at_k", "recall_at_k"):
            if (
                result.get(metric) is not None
                and baseline.get(metric) is not None
                and (result[metric] + criteria["quality_regression_tolerance"] < baseline[metric])
            ):
                failures.append(f"Ухудшение {metric}: {record['case_id']}")
        for k, value in result["retrieval_recall_at_k"].items():
            previous = baseline["retrieval_recall_at_k"][k]
            if value is not None and previous is not None and value < previous:
                failures.append(f"Ухудшение Recall@{k}: {record['case_id']}")
    same_stand = all(
        current[key] == reference[key]
        for key in (
            "python",
            "platform",
            "dependencies",
        )
    )
    if not same_stand:
        pending.append("Изменилось окружение измерений; нужен тот же стенд")
    else:
        if current["median_seconds_per_profile"] > (
            reference["median_seconds_per_profile"] * criteria["latency_ratio"]
            + criteria["latency_slack_seconds"]
        ):
            failures.append("Превышен бюджет времени ответа")
        if current["peak_python_allocated_bytes"] > (
            reference["peak_python_allocated_bytes"] + criteria["python_peak_growth_bytes"]
        ):
            failures.append("Превышен бюджет Python-памяти")
    return {
        "status": "failed" if failures else "pending" if pending else "passed",
        "failures": failures,
        "pending": pending,
        "same_environment": same_stand,
        "criteria": criteria,
    }


def evaluate_benchmark(directory: Path, *, repeats: int = 1) -> dict:
    catalog, cases, judgments, manifest = load_bundle(directory)
    frozen = json.loads((directory / "baseline.json").read_text(encoding="utf-8"))
    current, resources = measure_baseline(
        catalog,
        cases,
        repeats=repeats,
        limit=manifest["candidate_limit"],
        min_score=manifest["min_score"],
    )
    runs = {"frozen_baseline": frozen, "current_rules": current}
    case_reports = []
    for case in cases.cases:
        rows = [row for row in judgments if row.case_id == case.id]
        human = [row for row in rows if row.origin == "human"]
        complete = len(human) == len(catalog.vacancies)
        labels = {row.vacancy_key: row.relevance for row in human}
        violations = {row.vacancy_key for row in human if row.violates_required}
        experience = case.profile.experience_years
        record = {
            "case_id": case.id,
            "profession": case.profession,
            "split": case.split,
            "experience_group": "unknown"
            if experience is None
            else ("zero" if experience == 0 else "experienced"),
            "catalog_pairs": len(rows),
            "human_judged": len(human),
            "draft_judged": sum(row.origin == "assistant_draft" for row in rows),
            "complete_human_catalog": complete,
            "quality_status": "confirmed" if complete else "awaiting_human_review",
            "title_coverage_hint": sum(
                any(
                    (query := title_tokens(position))
                    and len(query & title_tokens(v.title)) / len(query) >= 0.6
                    for position in case.profile.preferences.positions
                )
                for v in catalog.vacancies
            ),
            "methods": {},
        }
        for method, run in runs.items():
            value = run[case.id]
            displayed = value["displayed"][: manifest["top"]]
            metrics = (
                ranking_metrics(displayed, labels, top=manifest["top"])
                if complete
                else {
                    key: None
                    for key in (
                        "ndcg_at_k",
                        "precision_at_k",
                        "precision_among_shown",
                        "recall_at_k",
                        "relevant_catalog",
                        "shown",
                    )
                }
            )
            metrics["violation_rate"] = (
                len(set(displayed) & violations) / len(displayed)
                if complete and displayed
                else None
            )
            retrieval = {}
            for k in (5, 20, 100, 500):
                retrieval[str(k)] = (
                    ranking_metrics(value["retrieved"], labels, top=k)["recall_at_k"]
                    if complete
                    else None
                )
            record["methods"][method] = {
                **metrics,
                "retrieval_recall_at_k": retrieval,
                "retrieved_count": len(value["retrieved"]),
                "eligible_count": len(value["eligible"]),
                "displayed_ids": displayed,
                "human_constraint_violations": sorted(set(displayed) & violations),
                "violation_labels_complete": complete,
            }
        if complete and not any(grade >= 2 for grade in labels.values()):
            record["quality_status"] = "no_relevant_vacancies_in_catalog"
        case_reports.append(record)
    aggregates = {}
    for method in runs:
        aggregates[method] = {}
        for dimension in ("all", "profession", "experience_group", "split"):
            groups = {"all"} if dimension == "all" else {r[dimension] for r in case_reports}
            aggregates[method][dimension] = {
                group: _average(
                    [
                        r["methods"][method]
                        for r in case_reports
                        if dimension == "all" or r[dimension] == group
                    ]
                )
                for group in sorted(groups)
            }
    ready = all(row.origin == "human" for row in judgments)
    reference_resources = json.loads((directory / "stand.json").read_text(encoding="utf-8"))
    return {
        "version": "benchmark-report-v1",
        "reference_time": catalog.reference_time.isoformat(),
        "input_sha256": {
            **manifest["sha256"],
            "manifest.json": sha256((directory / "manifest.json").read_bytes()).hexdigest(),
            "labels.jsonl": sha256((directory / "labels.jsonl").read_bytes()).hexdigest(),
        },
        "current_implementation_sha256": implementation_hash(),
        "quality_ready": ready,
        "catalog_count": len(catalog.vacancies),
        "catalog_exceeds_candidate_limit": len(catalog.vacancies) > manifest["candidate_limit"],
        "duplicates_removed": len(catalog.duplicates),
        "skipped_html": len(catalog.skipped),
        "professions": manifest["professions"],
        "cases": case_reports,
        "macro_averages": aggregates,
        "resources": resources,
        "baseline_resources": reference_resources,
        "acceptance": acceptance(
            case_reports, resources, reference_resources, manifest["acceptance_criteria"]
        ),
        "limitations": [
            "Нет разметки — нет подтверждённой метрики; черновики не являются эталоном.",
            "Каталог общий для training/validation, дубли удалены; это проверка новых профилей, "
            "а не полностью новых вакансий. Похожие тексты проверяются по Jaccard >= 0.9.",
            "Recall считается по всему размеченному каталогу, а не только результатам отбора.",
            "Если каталог не превышает лимит 500, Recall@500 не проверяет потери на большом "
            "корпусе; Recall на меньших K всё равно учитывает неотобранные вакансии.",
            "Сохранённые HH HTML не содержат актуального LLM-разбора из рабочей БД.",
            "Ресурсы измерены локально, без сети, БД и LLM; это не нагрузка рабочего сервера.",
            "История показов и feedback не применяются: измеряется подбор нового профиля.",
            "title_coverage_hint — совпадение >=60% слов должности, а не оценка пригодности. "
            "Отсутствие таких заголовков указывает на возможный пробел каталога.",
        ],
    }


def render_benchmark_report(report: dict) -> str:
    def value(item):
        return "ожидает разметки" if item is None else f"{item:.3f}"

    lines = [
        "# Проверка качества подбора",
        "",
        f"Момент оценки: {report['reference_time']}. "
        f"Каталог: {report['catalog_count']} вакансий; "
        f"удалено дублей: {report['duplicates_removed']}.",
        "",
        f"**Статус приёмки: {report['acceptance']['status']}.** "
        "Показатели относятся только к парам, подтверждённым человеком; "
        "пропуски не считаются отказами.",
        "",
        "| Профиль | Часть | Подтверждено / всего | Подсказка покрытия названий | "
        "NDCG@5 | Precision@5 | Recall@5 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in report["cases"]:
        metrics = record["methods"]["current_rules"]
        lines.append(
            f"| {record['case_id']} | {record['split']} | "
            f"{record['human_judged']} / {record['catalog_pairs']} | "
            f"{record['title_coverage_hint']} | {value(metrics['ndcg_at_k'])} | "
            f"{value(metrics['precision_at_k'])} | {value(metrics['recall_at_k'])} |"
        )
    resources = report["resources"]
    lines += [
        "",
        "## Измерения",
        "",
        f"Среднее время на профиль в измеренном пакетном прогоне: "
        f"{resources['median_seconds_per_profile']:.3f} с. "
        f"Пик Python-выделений: {resources['peak_python_allocated_bytes'] / 1024**2:.1f} МиБ.",
        "Это локальный последовательный расчёт без БД, сети и LLM. "
        "Время измерено отдельно от отслеживания памяти; общая память процесса "
        "приведена в JSON.",
        "",
        "## Что осталось",
        "",
    ]
    remaining = report["acceptance"]["pending"] + report["acceptance"]["failures"]
    lines += ["- " + item for item in remaining] or ["Критерии выполнены."]
    lines += ["", "## Границы результата", ""]
    lines += ["- " + item for item in report["limitations"]]
    return "\n".join(lines) + "\n"
