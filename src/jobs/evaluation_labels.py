"""Blind review packages and immutable, evidence-checked human submission ledger."""

import json
import os
from contextlib import contextmanager
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from domain.evaluation import Fact, Strict
from jobs.evaluation_data import digest, load_data, tasks, write_new
from scoring.evaluation_metrics import normalize_fact_groups


class Evidence(Strict):
    source: Literal["profile_text", "vacancy_text"]
    quote: str = Field(min_length=1)


class PairReview(Strict):
    relevance: int = Field(ge=0, le=3, strict=True)
    violates_required: bool = Field(strict=True)
    reasons: list[str] = Field(min_length=1)
    unknowns: list[str]
    evidence: list[Evidence] = Field(min_length=1)
    rationale: str = Field(min_length=10)

    @model_validator(mode="after")
    def violation(self):
        if not self.rationale.strip() or any(not e.quote.strip() for e in self.evidence):
            raise ValueError("Nonempty human rationale and evidence required")
        if self.violates_required and self.relevance != 0:
            raise ValueError("Mandatory violation requires grade 0")
        return self


class FactReview(Strict):
    complete: Literal[True]
    facts: list[Fact]
    unknowns: list[str]
    rationale: str = Field(min_length=10)


class Submission(Strict):
    version: Literal["evaluation-review-v1"]
    assignment_hash: str
    reviewer: str = Field(min_length=2)
    reviewed_at: AwareDatetime
    attestation: Literal["I personally reviewed the frozen sources; no automatic labels"]
    records: list[dict]


def export_review(bundle, output, *, split, reviewer, role="reviewer"):
    data, _, bundle_hash = load_data(bundle)
    reviewer = reviewer.strip()
    if not reviewer.strip() or reviewer.casefold() in {"assistant", "llm", "auto"}:
        raise ValueError("Human reviewer identity required")
    cards = tasks(data, split)
    assignment = {
        "version": "evaluation-assignment-v1",
        "bundle_hash": bundle_hash,
        "split": split,
        "reviewer": reviewer,
        "role": role,
        "tasks_hash": digest(cards),
        "task_ids": sorted(cards),
    }
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "assignment.json", assignment)
    write_new(
        output / "review.json",
        {
            "version": "evaluation-review-v1",
            "assignment_hash": digest(assignment),
            "reviewer": reviewer,
            "reviewed_at": None,
            "attestation": None,
            "records": [{"task_id": key, "payload": None, "resolves": []} for key in sorted(cards)],
        },
    )
    # Frozen source cards are separate from editable labels; no algorithm scores,
    # draft labels, split-target grades or model predictions are exposed.
    write_new(output / "cards.json", cards)
    template = Path(__file__).resolve().parents[2] / "config/evaluation/review.html"
    payload = json.dumps(
        {"cards": cards, "assignment": assignment, "assignment_hash": digest(assignment)},
        ensure_ascii=False,
    )
    payload = payload.replace("<", "\\u003c")
    with (output / "cards.html").open("x", encoding="utf-8") as stream:
        stream.write(template.read_text(encoding="utf-8").replace("__REVIEW_PACKAGE__", payload))
    instructions = Path(__file__).resolve().parents[2] / "docs/evaluation/REVIEW.md"
    with (output / "INSTRUCTIONS.md").open("x", encoding="utf-8") as stream:
        stream.write(instructions.read_text(encoding="utf-8"))
    return {"tasks": len(cards), "assignment_hash": digest(assignment)}


def _ledger(directory, bundle_hash):
    entries = []
    if not directory.exists():
        return entries
    for path in sorted(directory.glob("*.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        seal = entry.pop("sha256")
        if seal != digest(entry) or path.stem != seal or entry["bundle_hash"] != bundle_hash:
            raise ValueError("Label provenance ledger changed or belongs to another bundle")
        if entry["origin"] != "human" or not entry["attested_human"]:
            raise ValueError("Invalid label provenance")
        if (
            entry["assignment_hash"] != digest(entry["assignment"])
            or entry["reviewer"] != entry["assignment"]["reviewer"]
            or entry["assignment"]["bundle_hash"] != bundle_hash
            or entry["role"] != entry["assignment"]["role"]
            or entry["role"] not in {"reviewer", "adjudicator"}
            or entry["attestation"]
            != "I personally reviewed the frozen sources; no automatic labels"
        ):
            raise ValueError("Imported label provenance mismatch")
        entries.append(entry | {"sha256": seal})
    return entries


def _signature(record):
    payload = record["payload"]
    if "relevance" in payload:
        return digest({k: payload[k] for k in ("relevance", "violates_required", "unknowns")})
    # Different evidence spans for the same fact are allowed; field/value,
    # importance, alternatives and unknown/conflicting status must agree.
    atoms = [
        dict((k, v) for k, v in f.items() if k not in {"start", "end", "quote"})
        for f in normalize_fact_groups(payload["facts"])
    ]
    return digest({"facts": sorted(atoms, key=digest), "unknowns": sorted(payload["unknowns"])})


@contextmanager
def _import_lock(path):
    stream = path.open("x")
    try:
        yield
    finally:
        stream.close()
        path.unlink()


def _commit(ledger, entry):
    seal = digest(entry)
    target = ledger / f"{seal}.json"
    if target.exists():
        raise ValueError("Import already exists")
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=ledger,
            prefix=".import-",
            suffix=".pending",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(
                entry | {"sha256": seal}, stream, ensure_ascii=False, indent=2, allow_nan=False
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return seal


def import_review(bundle, assignment_path, submission_path, ledger, *, attest_human=False):
    if not attest_human:
        raise ValueError("Explicit --attest-human required; never convert assistant_draft")
    data, _, bundle_hash = load_data(bundle)
    assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
    submission = Submission.model_validate_json(submission_path.read_bytes())
    cards = tasks(data, assignment["split"])
    if (
        assignment["bundle_hash"] != bundle_hash
        or assignment["tasks_hash"] != digest(cards)
        or assignment["task_ids"] != sorted(cards)
        or submission.assignment_hash != digest(assignment)
        or submission.reviewer != assignment["reviewer"]
        or submission.reviewer.casefold() in {"assistant", "llm", "auto"}
        or submission.reviewed_at > datetime.now(UTC)
        or assignment["role"] not in {"reviewer", "adjudicator"}
    ):
        raise ValueError("Assignment/source/reviewer provenance mismatch")
    validated, seen = [], set()
    for record in submission.records:
        if set(record) != {"task_id", "payload", "resolves"}:
            raise ValueError("Unknown record/provenance fields")
        key = record["task_id"]
        if key not in cards or key in seen:
            raise ValueError("Unknown or repeated review task")
        seen.add(key)
        if record["payload"] is None:
            continue
        card = cards[key]
        if card["kind"] == "facts":
            payload = FactReview.model_validate(record["payload"])
            if len({digest(f.model_dump()) for f in payload.facts}) != len(payload.facts):
                raise ValueError("Repeated fact")
            for fact in payload.facts:
                fact.validate_source(card["document"])
        else:
            payload = PairReview.model_validate(record["payload"])
            for evidence in payload.evidence:
                if evidence.quote not in card[evidence.source]:
                    raise ValueError("Pair evidence does not match frozen source")
        validated.append(record | {"payload": payload.model_dump(mode="json")})
    if not validated:
        raise ValueError("No completed human reviews to import")
    ledger.mkdir(parents=True, exist_ok=True)
    lock = ledger / ".import.lock"
    # Serializes validation + exclusive commit, including concurrent imports.
    with _import_lock(lock):
        previous = _ledger(ledger, bundle_hash)
        fresh = []
        for record in validated:
            votes = [
                e for e in previous if any(r["task_id"] == record["task_id"] for r in e["records"])
            ]
            own = [
                r
                for e in votes
                if e["reviewer"] == submission.reviewer
                for r in e["records"]
                if r["task_id"] == record["task_id"]
            ]
            if own:
                if own == [record]:
                    continue  # Cumulative browser exports are idempotent.
                raise ValueError("Existing human review cannot be overwritten")
            refs = record["resolves"]
            if assignment["role"] == "adjudicator":
                independent = [e for e in votes if e["role"] == "reviewer"]
                if (
                    len(independent) != 2
                    or set(refs) != {e["sha256"] for e in independent}
                    or any(e["role"] == "adjudicator" for e in votes)
                ):
                    raise ValueError("Adjudication must reference exactly two independent votes")
                signatures = {
                    _signature(next(r for r in e["records"] if r["task_id"] == record["task_id"]))
                    for e in independent
                }
                if len(signatures) != 2:
                    raise ValueError("No disagreement to adjudicate")
            elif refs or len(votes) >= 2:
                raise ValueError("Two independent reviews only; use adjudication for conflicts")
            fresh.append(record)
        if not fresh:
            return {"imported": 0, "sha256": None, "status": "already_imported"}
        entry = {
            "version": "evaluation-import-v1",
            "bundle_hash": bundle_hash,
            "assignment": assignment,
            "assignment_hash": digest(assignment),
            "submission_hash": sha256(submission_path.read_bytes()).hexdigest(),
            "reviewer": submission.reviewer,
            "reviewed_at": submission.reviewed_at.isoformat(),
            "attestation": submission.attestation,
            "origin": "human",
            "attested_human": True,
            "role": assignment["role"],
            "records": fresh,
        }
        seal = _commit(ledger, entry)
    return {"imported": len(fresh), "sha256": seal}


def resolve_labels(directory, bundle_hash):
    entries = _ledger(directory, bundle_hash) if directory else []
    votes = {}
    for entry in entries:
        for record in entry["records"]:
            votes.setdefault(record["task_id"], []).append((entry, record))
    resolved, conflicts, single = {}, [], []
    for key, rows in votes.items():
        adjudicated = [r for e, r in rows if e["role"] == "adjudicator"]
        if adjudicated:
            resolved[key] = adjudicated[0]["payload"]
        elif len(rows) == 2 and _signature(rows[0][1]) == _signature(rows[1][1]):
            resolved[key] = rows[0][1]["payload"]
        elif len(rows) >= 2:
            conflicts.append(key)
        else:
            single.append(key)
    return resolved, {
        "imports": [e["sha256"] for e in entries],
        "resolved": len(resolved),
        "disagreements": conflicts,
        "single_review": single,
        "human_identity": "operator-attested; local hashes detect accidental edits",
    }
