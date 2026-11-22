"""Unified working-path evaluator. No network, delivery, live data, or implicit labels."""

import asyncio
import importlib.metadata
import platform
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from domain import CandidateProfile
from domain.resume import RESUME_VERSION, ResumeDraft
from domain.vacancy_analysis import is_fresh
from jobs.admission import admit
from jobs.evaluation_data import digest, load_data, write_new
from jobs.evaluation_labels import resolve_labels
from jobs.rank import rank_recent
from jobs.resumes import prepared_profile, profile_hash
from jobs.retrieval_evaluation import MemoryCandidateSource
from nlp.professional_facts import (
    candidate_facts,
    resume_facts,
    vacancy_facts,
)
from nlp.resume_document import extract_document
from processing.resume import normalize_resume_text
from scoring.evaluation_metrics import catalog_metrics, fact_metrics, interval, wilson
from scoring.retrieval import retrieval_eligibility


def _fact(field, value, *, importance="unspecified", alternative_group=None):
    return {
        "field": field,
        "value": value,
        "importance": importance,
        "alternative_group": alternative_group,
        "status": "known",
    }


def extract_facts(row, *, owner_id=None, reference_time=None):
    """Adapters call current rule extractors. Unsupported runtime fields remain misses."""
    if hasattr(row, "profile"):
        text = normalize_resume_text(row.text)
        extraction = extract_document(text, reference_time=reference_time)
        owner_id = owner_id or int(digest(row.owner)[:12], 16) + 1
        professional = resume_facts(text, extraction, owner_id, version=RESUME_VERSION)
        facts = fact_atoms(professional)
        initial = CandidateProfile()
        initial.professional_facts = candidate_facts(initial, owner_id)
        draft = ResumeDraft(
            token="0" * 32,
            text=text,
            format="text",
            extraction=extraction,
            profile_hash=profile_hash(initial),
            professional_facts=professional,
        )
        return facts, prepared_profile(initial, draft, owner_id)
    vacancy = row.vacancy
    return fact_atoms(vacancy.professional_facts or vacancy_facts(vacancy)), None


def fact_atoms(professional):
    """wire adapter: unknown stays unknown; alternatives keep their OR identity."""
    requirements = defaultdict(list)
    for requirement in professional.requirements:
        if not requirement.id.startswith(("req:experience:", "req:source:experience")):
            requirements[requirement.fact_id].append(requirement)
    groups = {key: a.id for a in professional.alternatives for key in a.requirement_ids}
    atoms = []
    for fact in professional.facts:
        if fact.kind == "experience_total" and fact.id != "experience_total":
            continue  # Intermediate stated/partial interval totals are not resolved totals.
        if fact.assertion.status != "present":
            continue
        field = {
            "role": "position",
            "skill": "skill",
            "task": "task",
            "experience_total": "experience_total",
            "qualification": "qualification",
            "seniority": "seniority",
        }.get(fact.kind)
        value = (
            fact.name
            if fact.kind in {"role", "skill", "task", "qualification"}
            else fact.assertion.value
        )
        if fact.kind == "condition" and fact.name in {"remote", "location", "archived"}:
            field = fact.name
        if field:
            for requirement in requirements.get(fact.id, [None]):
                atom = _fact(
                    field,
                    value,
                    importance=(requirement.importance.value or "unspecified")
                    if requirement
                    else "unspecified",
                    alternative_group=groups.get(requirement.id) if requirement else None,
                )
                if requirement and requirement.importance.status == "conflict":
                    atom["status"] = "conflicting"
                atoms.append(atom)
    preferences = {
        f.name: f.assertion.value
        for f in professional.facts
        if f.kind == "preference" and f.assertion.status == "present"
    }
    for location in preferences.get("locations", []):
        atoms.append(
            _fact(
                "location",
                location,
                importance="required" if preferences.get("locations_required") else "preferred",
            )
        )
    if preferences.get("salary_min") is not None:
        atoms.append(
            _fact(
                "salary",
                {
                    "minimum": preferences["salary_min"],
                    "currency": preferences["currency"],
                    "period": "month",
                    "gross": preferences["salary_gross"],
                },
                importance="required" if preferences.get("salary_required") else "preferred",
            )
        )
    if preferences.get("remote") in {"required", "preferred"}:
        atoms.append(_fact("remote", True, importance=preferences["remote"]))
    return atoms


def stand():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "dependencies": {
            d.metadata["Name"]: d.version
            for d in sorted(
                importlib.metadata.distributions(), key=lambda d: d.metadata["Name"].lower()
            )
        },
    }


def source_revision():
    # Editable and wheel installs must identify the same shipped code. Protocol/data
    # have their own bundle hashes; unrelated site-packages are never part of this tree.
    root = Path(__file__).resolve().parents[1]
    packages = (
        "bot",
        "collectors",
        "domain",
        "jobs",
        "nlp",
        "observability",
        "processing",
        "scoring",
        "storage",
    )
    inventory = {
        "src/" + p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest()
        for directory in (root / package for package in packages)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.suffix in {".py", ".json", ".yaml", ".html"}
    }
    if "src/jobs/evaluation.py" not in inventory:
        raise ValueError("Installed runtime source inventory is unavailable")
    return {"inventory_sha256": digest(inventory), "files": inventory}


def _canonical(order, groups):
    return list(dict.fromkeys(groups[key] for key in order))


def _labels(case, vacancies, resolved):
    by_group = defaultdict(list)
    for v in vacancies:
        label = resolved.get(f"pair:{case.id}:{v.id}")
        by_group[v.duplicate_group].append(label)
    labels, violations = {}, {}
    for group, rows in by_group.items():
        if all(rows) and len({(r["relevance"], r["violates_required"]) for r in rows}) == 1:
            labels[group] = rows[0]["relevance"]
            violations[group] = rows[0]["violates_required"]
    return labels, violations


async def evaluate(
    bundle,
    *,
    split="development",
    ledger=None,
    repeats=None,
    clients=1,
    k=None,
    min_score=None,
    final_test=False,
    release_id=None,
    ranker_policy=None,
    experimental_ranker=False,
    pair_reranker=None,
):
    data, protocol, bundle_hash = load_data(bundle)
    if pair_reranker is not None and split == "test":
        raise ValueError("Experimental pair policy cannot evaluate the final test")
    if split == "test" and (not final_test or not release_id):
        raise ValueError(
            "Final test requires --final-test and a frozen --release-id; no phase tuning"
        )
    repeats = protocol["repeats"] if repeats is None else repeats
    if repeats < 2 or clients not in (1, 2, 4):
        raise ValueError("At least two warm repeats; clients must follow frozen matrix 1/2/4")
    selected_k = protocol["retrieval_k"] if k is None else k
    threshold = protocol["min_score"] if min_score is None else min_score
    profiles = [p for p in data.profiles if p.split == split]
    vacancies = [v for v in data.vacancies if v.split == split]
    if not profiles or not vacancies:
        raise ValueError("Empty evaluation split")
    now = data.reference_times[split]
    groups = {v.id: v.duplicate_group for v in vacancies}
    catalog_ids = sorted(set(groups.values()))
    jobs = {v.id: v.vacancy for v in vacancies}
    resolved, provenance = resolve_labels(ledger, bundle_hash)
    fact_rows, extraction = [], {}
    for row in [*profiles, *vacancies]:
        start = perf_counter()
        facts, extracted_profile = extract_facts(row, reference_time=now)
        elapsed = perf_counter() - start
        extraction[row.id] = (extracted_profile, elapsed)
        human = resolved.get("facts:" + row.id)
        fact_rows.append(
            {
                "id": row.id,
                "kind": "profile" if extracted_profile else "vacancy",
                "cluster": row.owner if extracted_profile else row.duplicate_group,
                "predicted": facts,
                "seconds": elapsed,
                "human": fact_metrics(facts, human["facts"]) if human else None,
                "assistant_draft_diagnostic": fact_metrics(
                    facts, [f.model_dump() for f in row.draft_facts]
                ),
            }
        )

    async def run_case(case, mode, source):
        profile = case.profile if mode == "confirmed_profile" else extraction[case.id][0]
        trace = {}
        started = perf_counter()
        ranking = await rank_recent(
            None,
            profile,
            now=now,
            limit=selected_k,
            candidate_source=source,
            trace=trace,
            ranker_policy=ranker_policy,
            experimental_ranker=experimental_ranker,
            pair_reranker=pair_reranker,
            pair_owner=case.owner if pair_reranker is not None else None,
        )
        admission_start = perf_counter()
        _, admitted, shown = admit(ranking, min_score=threshold, top=protocol["top"])
        trace["admission_seconds"] = perf_counter() - admission_start
        trace["selection_total_seconds"] = perf_counter() - started
        trace["admitted"] = [f"{v.source}:{v.source_id}" for v, _ in admitted]
        trace["shown"] = [f"{v.source}:{v.source_id}" for v, _ in shown]
        trace["selected_order"] = trace["shown"]
        trace["admission_decisions"] = {
            f"{v.source}:{v.source_id}": m.admission for v, m in ranking
        }
        trace["profile_extraction_seconds"] = (
            extraction[case.id][1] if mode == "resume_rules" else 0
        )
        trace["total_with_profile_extraction_seconds"] = (
            trace["selection_total_seconds"] + trace["profile_extraction_seconds"]
        )
        return {
            "case_id": case.id,
            "owner": case.owner,
            "profession": case.profession,
            "scenarios": case.scenarios,
            "mode": mode,
            "trace": trace,
        }

    runs, projection, batch_seconds = [], [], []
    # First pass is reported separately. Rebuilding projection is timed separately
    # every pass; warm means process/NLP caches, not a user selection cache.
    for _repeat in range(repeats + 1):
        if _repeat:
            for row, fact_row in zip([*profiles, *vacancies], fact_rows, strict=True):
                started = perf_counter()
                predicted, extracted_profile = extract_facts(row, reference_time=now)
                elapsed = perf_counter() - started
                if predicted != fact_row["predicted"]:
                    raise ValueError("Frozen extraction changed between repetitions")
                extraction[row.id] = (extracted_profile, elapsed)
                fact_row.setdefault("warm_seconds", []).append(elapsed)
        started = perf_counter()
        source = MemoryCandidateSource([v.vacancy for v in vacancies])
        projection.append(perf_counter() - started)
        semaphore = asyncio.Semaphore(clients)

        async def bounded(case, mode, semaphore=semaphore, source=source):
            async with semaphore:
                return await run_case(case, mode, source)

        started = perf_counter()
        rows = await asyncio.gather(
            *(bounded(p, mode) for mode in protocol["modes"] for p in profiles)
        )
        batch_seconds.append(perf_counter() - started)
        runs.append(rows)

    def fingerprint(rows):
        return digest(
            [
                {
                    "case_id": r["case_id"],
                    "mode": r["mode"],
                    "stages": {
                        key: value
                        for key, value in r["trace"].items()
                        if not key.endswith("seconds") and key != "pair_reranker"
                    },
                }
                for r in rows
            ]
        )

    if len({fingerprint(rows) for rows in runs}) != 1:
        raise ValueError("Frozen runtime changed ranking between repetitions")
    records = runs[-1]
    for record_index, row in enumerate(records):
        case = next(p for p in profiles if p.id == row["case_id"])
        labels, violations = _labels(case, vacancies, resolved)
        trace = row["trace"]
        shown = _canonical(trace["shown"], groups)
        row["display"] = catalog_metrics(shown, labels, catalog_ids, top=protocol["top"])
        row["order_before_admission"] = catalog_metrics(
            _canonical(trace["ranked"], groups), labels, catalog_ids, top=protocol["top"]
        )
        row["recall_by_stage"] = {}
        for stage in ("retrieved", "filtered", "ranked", "admitted", "shown"):
            order = _canonical(trace[stage], groups)
            row["recall_by_stage"][stage] = catalog_metrics(
                order, labels, catalog_ids, top=max(1, len(order))
            )
        row["retrieval_recall_at_k"] = {}
        # K truncates the same stable selector order. Smaller K must also be run
        # explicitly for latency/policy experiments; this reports recall only.
        for cutoff in protocol["recall_ks"]:
            if cutoff <= selected_k:
                metric = catalog_metrics(
                    _canonical(trace["retrieved"], groups), labels, catalog_ids, top=cutoff
                )
                row["retrieval_recall_at_k"][str(cutoff)] = {
                    "recall": metric["recall"],
                    "found": metric["relevant_found"],
                    "denominator": metric["recall_denominator"],
                    "pooled_recall": metric["pooled_recall"],
                    "unjudged_groups": metric["unjudged_groups"],
                    "lower_bound": metric["recall_lower_bound"],
                    "upper_bound": metric["recall_upper_bound"],
                }
        profile = case.profile if row["mode"] == "confirmed_profile" else extraction[case.id][0]
        eligible = {
            v.duplicate_group
            for v in vacancies
            if retrieval_eligibility(v.vacancy, profile, now).eligible
            and is_fresh(v.vacancy, now=now)
        }
        positives = {g for g, grade in labels.items() if grade >= 2}
        recalled = set(_canonical(trace["retrieved"], groups)) & positives
        row["loss_audit"] = {
            "human_positive_catalog": len(positives),
            "positive_rejected_by_runtime_conditions": sorted(positives - eligible),
            "eligible_positive_catalog": len(positives & eligible),
            "eligible_recalled": len(recalled & eligible),
            "eligible_recall_diagnostic": len(recalled & eligible) / len(positives & eligible)
            if positives & eligible and len(labels) == len(catalog_ids)
            else None,
            "warning": "runtime eligibility never reduces main full-catalog recall denominator",
        }
        judged_shown = set(shown) & violations.keys()
        row["safety"] = {
            "mandatory": wilson(sum(violations[g] for g in judged_shown), len(judged_shown)),
            "mandatory_unjudged_shown": len(set(shown) - violations.keys()),
            "stale": wilson(
                sum(not is_fresh(jobs[key], now=now) for key in trace["shown"]), len(trace["shown"])
            ),
            "duplicate_displays": len(trace["shown"]) - len(shown),
            "publication_age_days": [
                (now - jobs[key].published_at).total_seconds() / 86400 for key in trace["shown"]
            ],
        }
        row["latency"] = {
            key: [run[record_index]["trace"][key] for run in runs[1:]]
            for key in trace
            if key.endswith("seconds")
        }
    summary = {}
    for mode in protocol["modes"]:
        rows = [r for r in records if r["mode"] == mode]
        summary[mode] = {
            "queries": len(rows),
            "shown": sum(len(r["trace"]["shown"]) for r in rows),
            "slots": len(rows) * protocol["top"],
            "empty_queries": wilson(sum(not r["trace"]["shown"] for r in rows), len(rows)),
            "human_complete_queries": sum(r["display"]["unjudged_groups"] == 0 for r in rows),
            "human_no_relevant_queries": sum(
                r["display"]["unjudged_groups"] == 0 and r["display"]["relevant_catalog"] == 0
                for r in rows
            ),
            "precision_among_shown_micro": (
                sum(r["display"]["relevant_found"] for r in rows)
                / sum(len(r["trace"]["shown"]) for r in rows)
                if all(r["display"]["unjudged_groups"] == 0 for r in rows)
                and any(r["trace"]["shown"] for r in rows)
                else None
            ),
            "macro": {
                key: _cluster_interval([(r["owner"], r["display"][key]) for r in rows], protocol)
                for key in ("ndcg_at_5", "precision_at_5", "precision_among_shown", "recall")
            },
            "retrieval_recall_at_k": {
                str(cutoff): _cluster_interval(
                    [(r["owner"], r["retrieval_recall_at_k"][str(cutoff)]["recall"]) for r in rows],
                    protocol,
                )
                for cutoff in protocol["recall_ks"]
                if cutoff <= selected_k
            },
            "latency_seconds": {
                stage: _distribution([value for r in rows for value in r["latency"][stage]])
                for stage in rows[0]["latency"]
            },
            "mandatory": wilson(
                sum(r["safety"]["mandatory"]["numerator"] for r in rows),
                sum(r["safety"]["mandatory"]["denominator"] for r in rows),
            ),
            "mandatory_unjudged_shown": sum(r["safety"]["mandatory_unjudged_shown"] for r in rows),
            "by_profession": {
                p: {
                    "queries": sum(r["profession"] == p for r in rows),
                    "shown": sum(len(r["trace"]["shown"]) for r in rows if r["profession"] == p),
                }
                for p in sorted({r["profession"] for r in rows})
            },
        }
    return {
        "version": "working-evaluation-report-v1",
        "evaluated_at": datetime.now(UTC).isoformat(),
        "reference_time": now.isoformat(),
        "split": split,
        "release_id": release_id,
        "ranker_policy": ranker_policy.model_dump(mode="json")
        if ranker_policy
        else {"mode": "rules"},
        "experimental_ranker": experimental_ranker,
        "pair_policy": pair_reranker.policy.model_dump(mode="json") if pair_reranker else None,
        "bundle_hash": bundle_hash,
        "protocol_hash": digest(protocol),
        "protocol": protocol,
        "parameters": {"k": selected_k, "min_score": threshold, "top": protocol["top"]},
        "seed": data.seed,
        "repeats": repeats,
        "clients": clients,
        "quality": "ready_for_review"
        if all(v["human_complete_queries"] == len(profiles) for v in summary.values())
        and all(f["human"] is not None for f in fact_rows)
        else "waiting_external",
        "provenance": provenance,
        "stand": stand(),
        "source": source_revision(),
        "deterministic_result_sha256": digest(
            {
                "ranking": fingerprint(records),
                "facts": [{"id": f["id"], "predicted": f["predicted"]} for f in fact_rows],
            }
        ),
        "catalog_cards": len(vacancies),
        "catalog_duplicate_groups": len(catalog_ids),
        "summary": summary,
        "cases": records,
        "facts": fact_rows,
        "fact_summary": {
            kind: {
                "documents": sum(f["kind"] == kind for f in fact_rows),
                "human_documents": sum(
                    f["kind"] == kind and f["human"] is not None for f in fact_rows
                ),
                "human_fact_recall": _cluster_interval(
                    [
                        (f["cluster"], f["human"]["fact_recall"])
                        for f in fact_rows
                        if f["kind"] == kind and f["human"] is not None
                    ],
                    protocol,
                ),
                "human_fact_precision": _cluster_interval(
                    [
                        (f["cluster"], f["human"]["fact_precision"])
                        for f in fact_rows
                        if f["kind"] == kind and f["human"] is not None
                    ],
                    protocol,
                ),
                "extraction_seconds": _distribution(
                    [t for f in fact_rows if f["kind"] == kind for t in f["warm_seconds"]]
                ),
            }
            for kind in ("profile", "vacancy")
        },
        "measurement": {
            "projection_seconds": projection,
            "batch_seconds": batch_seconds,
            "first_pass": "first process pass, not guaranteed OS/model cold",
            "scope": (
                "actual rank_recent + selector + rules + apply_ranker + admission; "
                "memory adapter; no SQL/history/feedback/delivery/network/LLM; "
                "vacancy extraction measured separately; profile rules before selection; "
                + (
                    "pair adapter/cache included; inspect per-query traces"
                    if pair_reranker
                    else "no pair adapter or pair inference cache"
                )
            ),
            "rss_bytes": _rss(),
        },
    }


def _cluster_interval(observations, protocol):
    grouped = defaultdict(list)
    for cluster, value in observations:
        if value is not None:
            grouped[cluster].append(value)
    return interval(
        [statistics.mean(v) for v in grouped.values()],
        seed=protocol["seed"],
        repeats=protocol["bootstrap_repeats"],
        confidence=protocol["confidence"],
    )


def _distribution(values):
    values = sorted(values)
    return {
        "n": len(values),
        "median": statistics.median(values),
        "p95": values[min(len(values) - 1, int(0.95 * len(values)))],
        "max": max(values),
    }


def _rss():
    try:
        import psutil

        return psutil.Process().memory_info().rss
    except ImportError:
        return None


def save_report(report, output):
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "report.json", report)
    write_new(
        output / "manifest.json",
        {
            "report_sha256": sha256((output / "report.json").read_bytes()).hexdigest(),
            "bundle_hash": report["bundle_hash"],
            "source_sha256": report["source"]["inventory_sha256"],
            "deterministic_result_sha256": report["deterministic_result_sha256"],
        },
    )
    lines = [
        "# Рабочий протокол оценки",
        "",
        f"Quality: **{report['quality']}**. Split: {report['split']}. "
        f"Каталог: {report['catalog_cards']} карточек / "
        f"{report['catalog_duplicate_groups']} групп.",
        "",
        "Метрики без полной независимой разметки остаются null; partial pool "
        "и границы указаны в JSON. Черновые факты — отдельная диагностика.",
        "",
        "| Вход | Запросов | Показов / слотов | Пустых | Human complete |",
        "| --- | --- | --- | --- | --- |",
    ]
    for mode, values in report["summary"].items():
        lines.append(
            f"| {mode} | {values['queries']} | {values['shown']}/{values['slots']} | "
            f"{values['empty_queries']['numerator']} | {values['human_complete_queries']} |"
        )
    lines += [
        "",
        report["measurement"]["scope"],
        "",
        "Стадии и каждый знаменатель, интервалы, источники и hashes: report.json.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
