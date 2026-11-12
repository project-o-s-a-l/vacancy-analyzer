"""Explicit NEW-041 derivation; no human-ledger rewrite or fresh holdout claim."""

import json
from hashlib import sha256

from domain.evaluation import EvaluationData
from jobs.evaluation_data import DATA_VERSION, digest, freeze_bundle, load_data, write_new


def migrate_units(bundle, output, *, draft_labels=None):
    if output.resolve().is_relative_to(bundle.resolve()):
        raise ValueError("Migration output must be outside the immutable source bundle")
    if output.exists():
        raise FileExistsError("Choose a new immutable migration directory")
    original, protocol, parent_hash = load_data(bundle, allow_legacy=True)
    if original.version != "working-evaluation-data-v1":
        raise ValueError("Unit migration requires an archival v1 bundle")
    raw = json.loads((bundle / "data.json").read_text(encoding="utf-8"))
    raw["version"] = DATA_VERSION
    changes = []
    for row in raw["vacancies"]:
        if row["vacancy"].get("salary_period") == "month":
            row["vacancy"]["salary_period"] = "MONTH"
            changes.append(
                {
                    "id": row["id"],
                    "field": "vacancy.salary_period",
                    "before": "month",
                    "after": "MONTH",
                }
            )
    # No other aliases/defaults are repaired. Unsupported input requires separate review.
    data = EvaluationData.model_validate(raw)
    before = original.model_dump(mode="json")
    after = data.model_dump(mode="json")
    after["version"] = before["version"]
    old_periods = {r["id"]: r["vacancy"]["salary_period"] for r in before["vacancies"]}
    for row in after["vacancies"]:
        row["vacancy"]["salary_period"] = old_periods[row["id"]]
    if after != before:
        raise ValueError("Migration changed fields beyond version and salary_period case")

    draft = None
    if draft_labels is not None:
        draft = json.loads(draft_labels.read_text(encoding="utf-8"))
        if (
            draft.get("origin") != "assistant_draft"
            or draft.get("bundle_hash") != parent_hash
            or draft.get("human_reviewed") is not False
            or draft.get("acceptance_eligible") is not False
        ):
            raise ValueError("Only explicitly machine-only draft labels for the source bundle")
        expected = {
            f"pair:{p.id}:{v.id}"
            for p in data.profiles
            if p.split != "test"
            for v in data.vacancies
            if v.split == p.split
        }
        if set(draft.get("records", {})) != expected:
            raise ValueError("Draft must contain exactly all non-test pairs, no test records")
        for record in draft["records"].values():
            if (
                set(record) - {"relevance", "violates_required", "rationale"}
                or type(record.get("relevance")) is not int
                or record["relevance"] not in range(4)
                or type(record.get("violates_required")) is not bool
                or record["violates_required"]
                and record["relevance"] != 0
            ):
                raise ValueError(
                    "Unsupported draft record; do not silently rebind structured evidence"
                )

    output.mkdir(parents=True, exist_ok=False)
    _, _, new_hash = freeze_bundle(
        output / "bundle",
        data,
        protocol,
        purpose="legacy_units_diagnostic",
        parent_bundle_hash=parent_hash,
    )
    report = {
        "version": "evaluation-unit-migration-v1",
        "source_bundle_hash": parent_hash,
        "bundle_hash": new_hash,
        "source_files_sha256": {
            name: sha256((bundle / name).read_bytes()).hexdigest()
            for name in ("data.json", "protocol.json", "manifest.json")
        },
        "changes": changes,
        "data_version": DATA_VERSION,
        "other_semantic_fields_unchanged": True,
        "independent_holdout": False,
        "test": "previously opened diagnostic only; no labels or release markers copied",
        "training_executed": False,
        "quality": "waiting_external",
        "transferred_draft_pairs": 0,
    }
    if draft is not None:
        # The grade/violation/rationale records stay identical; only explicit lineage changes.
        draft.update(
            bundle_hash=new_hash,
            unit_contract_blocker=None,
            training_authorized=False,
            experimental_draft=True,
            parent_bundle_hash=parent_hash,
            parent_draft_sha256=sha256(draft_labels.read_bytes()).hexdigest(),
            derivation="evaluation-unit-migration-v1; see migration.json; not reannotation",
        )
        write_new(output / "draft-labels.json", draft)
        report.update(
            transferred_draft_pairs=len(draft["records"]),
            records_sha256=digest(draft["records"]),
            source_draft_sha256=draft["parent_draft_sha256"],
        )
    write_new(output / "migration.json", report)
    return report
