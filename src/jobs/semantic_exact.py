"""frozen-catalog experiment. Nothing here is imported by the serving path.

Exact search is the reference: every current eligible vector is considered before K.
Private query vectors stay in an owner-isolated, bounded volatile cache, never on disk.
"""

import json
import math
import os
import threading
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

from domain.vacancy_analysis import is_fresh
from domain.vacancy_duplicates import vacancy_duplicate_key
from jobs.benchmark_data import vacancy_key
from jobs.hybrid_candidates import reciprocal_rank_fusion
from jobs.semantic_documents import vacancy_document
from scoring.retrieval import retrieval_eligibility
from scoring.serving_features import digest

VERSION = "f11-exact-v1"
MAX_BYTES = 64 * 1024 * 1024
MAX_ENTRIES = 5000


def valid_vector(value):
    return (
        isinstance(value, list)
        and len(value) == 384
        and all(type(x) in (int, float) and math.isfinite(x) for x in value)
        and abs(sum(x * x for x in value) - 1) < 0.001
    )


def seal_index(entries, identity):
    value = {"version": VERSION, "identity": identity, "entries": entries}
    return value | {"sha256": digest(value)}


def validate_index(value, identity):
    if not isinstance(value, dict):
        raise ValueError("invalid_index")
    payload = {k: v for k, v in value.items() if k != "sha256"}
    if (
        value.get("version") != VERSION
        or value.get("identity") != identity
        or digest(payload) != value.get("sha256")
        or not isinstance(value.get("entries"), dict)
        or len(value["entries"]) > MAX_ENTRIES
    ):
        raise ValueError("incompatible_or_corrupt_index")
    for entry in value["entries"].values():
        if not isinstance(entry, dict) or not all(
            valid_vector(entry.get(key)) for key in ("vector", "section_vector")
        ):
            raise ValueError("invalid_index_vector")
    return value


def read_index(path, identity):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("oversized_index")
    return validate_index(json.loads(raw), identity)


def write_index(path, value):
    validate_index(value, value["identity"])
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    if len(raw) > MAX_BYTES:
        raise ValueError("oversized_index")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def refresh_index(vacancies, identity, encode, *, previous=None, document=vacancy_document):
    """Rebuild membership; reuse only meaningful text+encoder matches; fence revisions.

    Callers publish this immutable result only for their frozen catalog snapshot.
    A future live external index requires a transactional outbox/CAS, not this helper.
    """
    if len(vacancies) > MAX_ENTRIES:
        raise ValueError("catalog_budget_exceeded")
    prior = validate_index(previous, identity)["entries"] if previous else {}
    entries, tasks, keys = {}, [], []
    omitted = 0
    for vacancy in vacancies:
        key = vacancy_key(vacancy)
        if key in entries:
            raise ValueError("duplicate_catalog_id")
        doc = document(vacancy)
        if not doc.sections:
            omitted += 1
            continue
        old = prior.get(key)
        entry = {
            "input_revision": doc.input_revision,
            "source_revision": doc.source_revision,
            "facts_revision": doc.facts_revision,
        }
        if old and old["input_revision"] == doc.input_revision:
            entry |= {k: old[k] for k in ("vector", "section_vector", "chunks", "tokens")}
        else:
            keys.append(key)
            tasks.append({"kind": "passage", "sections": list(doc.sections)})
        entries[key] = entry
    measured = encode(tasks) if tasks else {"values": [], "wall_seconds": 0}
    if len(measured["values"]) != len(keys):
        raise ValueError("incomplete_encoder_batch")
    for key, value in zip(keys, measured["values"], strict=True):
        if value.get("truncated"):
            raise ValueError("truncated_document")
        entries[key] |= {k: value[k] for k in ("vector", "section_vector", "chunks", "tokens")}
    result = seal_index(entries, identity)
    validate_index(result, identity)
    return result, {k: v for k, v in measured.items() if k != "values"} | {
        "computed": len(keys),
        "reused": len(entries) - len(keys),
        "omitted_empty": omitted,
        "removed": len(prior.keys() - entries.keys()),
    }


def exact_candidates(
    vacancies,
    profile,
    index,
    query_vector,
    *,
    identity,
    now,
    k=64,
    pooling="vector",
    excluded_ids=(),
    excluded_keys=(),
    days=None,
    category=None,
    document=vacancy_document,
):
    if k < 1 or pooling not in {"vector", "section_vector"}:
        raise ValueError("invalid_search_parameters")
    stats = dict(catalog=len(vacancies), filtered=0, stale_or_missing=0, duplicates=0, scanned=0)
    try:
        validate_index(index, identity)
    except (ValueError, TypeError, KeyError):
        return [], stats | {"fallback_reason": "missing_or_corrupt_index"}
    if not valid_vector(query_vector):
        return [], stats | {"fallback_reason": "invalid_query_vector"}
    ordered = []
    vector_seconds = 0.0
    for vacancy in vacancies:
        group = vacancy_duplicate_key(vacancy) or str(vacancy.id)
        if (
            vacancy.id in excluded_ids
            or group in excluded_keys
            or not is_fresh(vacancy, now=now, days=days)
            or category is not None
            and (not vacancy.analysis or vacancy.analysis.extraction.category != category)
            or not retrieval_eligibility(vacancy, profile, now).eligible
        ):
            stats["filtered"] += 1
            continue
        doc = document(vacancy)
        entry = index["entries"].get(vacancy_key(vacancy))
        if not entry or any(
            entry.get(key) != getattr(doc, key)
            for key in ("input_revision", "source_revision", "facts_revision")
        ):
            stats["stale_or_missing"] += 1
            continue
        stats["scanned"] += 1
        started = perf_counter()
        similarity = sum(a * b for a, b in zip(query_vector, entry[pooling], strict=True))
        vector_seconds += perf_counter() - started
        ordered.append(
            (
                -similarity,
                -vacancy.published_at.timestamp(),
                vacancy.source_id,
                str(vacancy.id),
                group,
                vacancy,
            )
        )
    selected, seen = [], set()
    examined = 0
    for *_, group, vacancy in sorted(ordered, key=lambda row: row[:4]):
        examined += 1
        if group in seen:
            stats["duplicates"] += 1
            continue
        seen.add(group)
        selected.append(vacancy)
        if len(selected) == k:
            break
    return selected, stats | {
        "selected": len(selected),
        "ordered_examined": examined,
        "vector_dot_seconds": vector_seconds,
        "exhausted": len(selected) < k,
        "fallback_reason": None if selected else "empty_current_dense_pool",
        "refill": "all_current_eligible_vectors_before_limit",
    }


def fuse(lexical, dense, *, k=64, method="rrf", quota=0.25):
    if k < 1 or not 0 <= quota <= 1 or method not in {"rrf", "novelty"}:
        raise ValueError("invalid_fusion")

    def group(v):
        return vacancy_duplicate_key(v) or str(v.id)

    representatives = {}
    for vacancy in [*lexical, *dense]:
        representatives.setdefault(group(vacancy), vacancy)
    if method == "rrf":
        order = [
            representatives[key]
            for key in reciprocal_rank_fusion(
                list(dict.fromkeys(group(v) for v in lexical)),
                list(dict.fromkeys(group(v) for v in dense)),
            )
        ]
    else:
        lex_groups = {group(v) for v in lexical}
        novel, novel_seen = [], set()
        for vacancy in dense:
            key = group(vacancy)
            if key not in lex_groups and key not in novel_seen:
                novel.append(vacancy)
                novel_seen.add(key)
        count = min(int(k * quota), len(novel))
        order = [*lexical[: k - count], *novel[:count], *lexical, *dense]
    result, seen = [], set()
    for vacancy in order:
        key = group(vacancy)
        if key not in seen:
            seen.add(key)
            result.append(vacancy)
            if len(result) == k:
                break
    return result


class QueryCache:
    """No persisted vectors, no cross-owner reuse, bounded LRU and deletion fencing."""

    def __init__(self, capacity=128):
        if capacity < 1:
            raise ValueError("invalid_cache_capacity")
        self.capacity = capacity
        self._values = OrderedDict()
        self._owners = OrderedDict()
        self._lock = threading.RLock()

    def delete(self, owner):
        with self._lock:
            self._owners[owner] = (object(), None)
            self._owners.move_to_end(owner)
            while len(self._owners) > 2 * self.capacity:
                self._owners.popitem(last=False)
            for key in list(self._values):
                if key[0] == owner:
                    del self._values[key]

    def get(self, owner, document, identity, encode):
        if owner is None or not str(owner):
            raise ValueError("owner_required")
        revision = (document.input_revision, digest(identity))
        key = (owner, *revision)
        with self._lock:
            previous = self._owners.get(owner)
            if previous is None or previous[1] != revision:
                self.delete(owner)
                self._owners[owner] = (object(), revision)
            token = self._owners[owner][0]
            if key in self._values:
                self._values.move_to_end(key)
                return deepcopy(self._values[key]), True
        if not document.sections:
            raise ValueError("empty_query_document")
        result = encode([{"kind": "query", "sections": list(document.sections)}])
        value = result["values"][0]
        if value.get("truncated") or not all(
            valid_vector(value.get(k)) for k in ("vector", "section_vector")
        ):
            raise ValueError("invalid_query_result")
        with self._lock:
            if self._owners.get(owner, (None,))[0] is not token:
                raise ValueError("profile_changed_during_encoding")
            self._values[key] = deepcopy(value)
            self._values.move_to_end(key)
            while len(self._values) > self.capacity:
                self._values.popitem(last=False)
            # Bound owner fence metadata too. Evict only owners with no live values.
            active = {item[0] for item in self._values}
            if len(self._owners) > 2 * self.capacity:
                self._owners = OrderedDict((k, v) for k, v in self._owners.items() if k in active)
        return deepcopy(value), False
