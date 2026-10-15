"""Predeclared paired comparison; uncertainty is never hidden by a fixed gain."""

import json
import statistics
from collections import defaultdict
from hashlib import sha256

from scoring.evaluation_metrics import interval, repeated_interval


def read_report(directory):
    path = directory / "report.json"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if sha256(path.read_bytes()).hexdigest() != manifest["report_sha256"]:
        raise ValueError("Frozen evaluation report changed")
    return json.loads(path.read_text(encoding="utf-8"))


def compare(reference, candidate, *, claim_improvement=False):
    for key in (
        "version",
        "bundle_hash",
        "protocol_hash",
        "split",
        "release_id",
        "clients",
        "repeats",
    ):
        if reference[key] != candidate[key]:
            raise ValueError(f"Comparison inputs differ: {key}")
    if reference["provenance"]["imports"] != candidate["provenance"]["imports"]:
        raise ValueError("Labels/provenance changed; rerun both versions on the same ledger")
    if reference["parameters"]["top"] != candidate["parameters"]["top"]:
        raise ValueError("Display denominator changed")
    old = {(r["case_id"], r["mode"]): r for r in reference["cases"]}
    new = {(r["case_id"], r["mode"]): r for r in candidate["cases"]}
    if old.keys() != new.keys():
        raise ValueError("Query/mode population changed")
    protocol = reference["protocol"]
    deltas, failures, pending = {}, [], []

    def assess(name, observations, *, latency=False):
        clusters = [statistics.mean(v) for v in observations.values()]
        result = interval(
            clusters,
            seed=protocol["seed"],
            repeats=protocol["bootstrap_repeats"],
            confidence=protocol["confidence"],
        )
        if latency:
            result = repeated_interval(
                observations,
                seed=protocol["seed"],
                repeats=protocol["bootstrap_repeats"],
                confidence=protocol["confidence"],
            )
        deltas[name] = result
        if result["low"] is None:
            pending.append(name + ": insufficient independent judged clusters")
        elif latency:
            if result["low"] > 1:
                failures.append(name + ": latency regression beyond measured uncertainty")
            elif result["high"] > 1:
                pending.append(name + ": latency non-regression inconclusive")
        elif result["high"] < 0:
            failures.append(name + ": quality/coverage regression beyond measured uncertainty")
        elif result["low"] < 0:
            pending.append(name + ": quality/coverage non-regression inconclusive")

    if reference["quality"] != "ready_for_review" or candidate["quality"] != "ready_for_review":
        pending.append("Full independent human pair and fact review is required")
    for mode in protocol["modes"]:
        rows = [key for key in old if key[1] == mode]
        for metric in ("ndcg_at_5", "precision_at_5", "recall_at_64", "shown", "nonempty"):
            grouped = defaultdict(list)
            for key in rows:

                def value(row, metric=metric):
                    if metric == "shown":
                        return len(row["trace"]["shown"])
                    if metric == "nonempty":
                        return int(bool(row["trace"]["shown"]))
                    if metric == "recall_at_64":
                        return row["retrieval_recall_at_k"].get("64", {}).get("recall")
                    return row["display"][metric]

                before, after = value(old[key]), value(new[key])
                if before is not None and after is not None:
                    grouped[old[key]["owner"]].append(after - before)
            assess(mode + "/" + metric, grouped)
        grouped = defaultdict(list)
        for key in rows:
            before = old[key]["latency"]["selection_total_seconds"]
            after = new[key]["latency"]["selection_total_seconds"]
            grouped[old[key]["owner"]].extend(b / a for a, b in zip(before, after, strict=True))
            safety = new[key]["safety"]
            if (
                safety["mandatory"]["numerator"]
                or safety["stale"]["numerator"]
                or safety["duplicate_displays"]
            ):
                failures.append(f"{key}: observed mandatory/stale/duplicate display")
            if safety["mandatory_unjudged_shown"]:
                pending.append(f"{key}: displayed mandatory conditions not fully judged")
        assess(mode + "/warm_selection_latency_ratio", grouped, latency=True)
    old_facts = {r["id"]: r for r in reference["facts"]}
    new_facts = {r["id"]: r for r in candidate["facts"]}
    if old_facts.keys() != new_facts.keys():
        raise ValueError("Fact population changed")
    for kind in ("profile", "vacancy"):
        for metric in ("fact_recall", "fact_precision"):
            grouped = defaultdict(list)
            for key, row in old_facts.items():
                other = new_facts[key]
                if row["kind"] != kind or not row["human"] or not other["human"]:
                    continue
                a, b = row["human"][metric], other["human"][metric]
                if (a is None) != (b is None):
                    pending.append(kind + "/" + metric + ": metric coverage changed")
                if a is not None and b is not None:
                    grouped[row["cluster"]].append(b - a)
            assess(kind + "/" + metric, grouped)
    if reference["stand"] != candidate["stand"]:
        pending.append("Stand changed; performance requires same runtime environment")
    improved = [
        name
        for name, value in deltas.items()
        if "latency" not in name and value["low"] is not None and value["low"] > 0
    ]
    if claim_improvement and not improved:
        pending.append("No strictly positive lower confidence bound supports an improvement claim")
    return {
        "version": "working-evaluation-comparison-v1",
        "status": "failed" if failures else "inconclusive" if pending else "passed",
        "failures": sorted(set(failures)),
        "pending": sorted(set(pending)),
        "paired_cluster_deltas": deltas,
        "supported_improvements": improved,
        "reference_source": reference["source"]["inventory_sha256"],
        "candidate_source": candidate["source"]["inventory_sha256"],
        "protocol_hash": reference["protocol_hash"],
        "limitations": "Per-metric 95% intervals, not a joint 95% guarantee; synthetic "
        "development population; test use is restricted by frozen release ID",
    }
