"""Offline-gated adapter on the shared rank/admission/snapshot path.

No settings switch exists: this experiment has not passed the human/resource gates.
Fallback keeps structured decisions, with the same unresolved-document guard used
before successful CE scoring. No partial reranking or default positive margin.
"""

import math
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from domain.serving_ranker import ModelAdmission, RankingDecision
from jobs.cross_encoder import MODEL_ID, PAIR_SPEC, score_bounded, validate_model, validate_values
from jobs.semantic_documents import DOCUMENT_SPEC, profile_document, vacancy_document
from scoring.admission_policy import model_admission
from scoring.serving_features import digest


class PairPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["pair-reranker-f12-v1"] = "pair-reranker-f12-v1"
    policy_id: str
    encoder_id: Literal[MODEL_ID] = MODEL_ID
    structured_model_id: str
    pool: Literal[8, 16, 32]
    alpha: Literal[0.25, 0.5, 0.75, 1.0]
    ce_mean: FiniteFloat
    ce_scale: float = Field(gt=0, allow_inf_nan=False)
    linear_mean: FiniteFloat
    linear_scale: float = Field(gt=0, allow_inf_nan=False)
    admission: ModelAdmission | None = None
    origin: Literal["assistant_draft"] = "assistant_draft"
    serving_allowed: Literal[False] = False

    @classmethod
    def seal(cls, **values):
        draft = cls.model_construct(policy_id="", **values)
        content = draft.model_dump(mode="json", exclude={"policy_id"})
        return cls.model_validate(content | {"policy_id": digest(content)})

    @model_validator(mode="after")
    def intact(self):
        if self.policy_id != digest(self.model_dump(mode="json", exclude={"policy_id"})):
            raise ValueError("pair_policy_checksum")
        return self

    def score(self, ce, linear):
        value = self.alpha * (ce - self.ce_mean) / self.ce_scale
        value += (1 - self.alpha) * (linear - self.linear_mean) / self.linear_scale
        if not math.isfinite(value):
            raise ValueError("nonfinite_combined_score")
        return value


def pair_task(profile, vacancy, *, owner):
    candidate = profile_document(profile)
    document = vacancy_document(vacancy)
    task = {
        "candidate": list(candidate.sections)
        + [
            "Unlisted or unresolved experience is not confirmed; "
            "desired role is not held experience."
        ],
        "vacancy": list(document.sections),
    }
    key = digest(
        {
            "owner": owner,
            "model": MODEL_ID,
            "pair_spec": PAIR_SPEC,
            "documents": DOCUMENT_SPEC,
            "candidate": [
                candidate.input_revision,
                candidate.source_revision,
                candidate.facts_revision,
            ],
            "vacancy": [document.input_revision, document.source_revision, document.facts_revision],
            "input": task,
        }
    )
    return key, task


class PairCache:
    """Bounded volatile cache; owner generations fence deletion/in-flight late puts."""

    def __init__(self, capacity=4096):
        if not 1 <= capacity <= 4096:
            raise ValueError("cache_budget")
        self.capacity = capacity
        self.rows = OrderedDict()
        self.lock = threading.Lock()
        self.generation = 0

    def clear_owner(self, owner):
        with self.lock:
            self.generation += 1
            for key in [k for k in self.rows if k[0] == owner]:
                del self.rows[key]

    def epoch(self):
        with self.lock:
            return self.generation

    def get(self, owner, key):
        with self.lock:
            record = self.rows.get((owner, key))
            if record is None:
                return None
            value, checksum = record
            try:
                validate_values([value], 1)
                if checksum != digest([owner, key, value]):
                    raise ValueError("cache_checksum")
            except (ValueError, TypeError):
                del self.rows[(owner, key)]
                raise ValueError("corrupt_pair_cache") from None
            self.rows.move_to_end((owner, key))
            return dict(value)

    def put(self, owner, key, value, epoch):
        validate_values([value], 1)
        with self.lock:
            if epoch != self.generation:
                raise ValueError("stale_pair_generation")
            self.rows[(owner, key)] = (dict(value), digest([owner, key, value]))
            self.rows.move_to_end((owner, key))
            while len(self.rows) > self.capacity:
                self.rows.popitem(last=False)


def required_guard(match):
    """Unknown required credential/condition cannot be made true by a neural logit."""
    return bool(
        match.comparison
        and any(
            item.importance == "required"
            and item.kind in {"qualification", "condition"}
            and item.state != "match"
            for item in match.comparison.items
        )
    )


class PairReranker:
    def __init__(self, policy, directory, *, cache=None, deadline=2.0):
        self.policy = policy
        self.directory = Path(directory)
        self.cache = cache if cache is not None else PairCache()
        self.deadline = deadline
        self.lock = threading.Lock()

    def apply(self, ranking, profile, *, owner, trace=None):
        start = time.monotonic()
        stats = {"policy_id": self.policy.policy_id, "cache_hits": 0, "fallback": None}
        # Never mutate cached baseline match objects or accept partially applied decisions.
        result = [(v, m.model_copy(deep=True)) for v, m in ranking]
        for index, (_, match) in enumerate(result, 1):
            if required_guard(match):
                if match.ranking is None or not match.ranking.applied:
                    match.ranking = RankingDecision(
                        requested_mode="pair",
                        applied=True,
                        baseline_rank=index,
                        admission_policy="required-evidence-f12-v1",
                        experimental=True,
                    )
                match.ranking.admission_passed = False
                match.ranking.admission_reason = "required_document_or_condition_unconfirmed"
        acquired = self.lock.acquire(blocking=False)
        try:
            if not acquired:
                raise ValueError("pair_request_busy")
            if not result:
                return result
            if any(
                not m.ranking
                or not m.ranking.applied
                or m.ranking.model_id != self.policy.structured_model_id
                for _, m in result
            ):
                raise ValueError("structured_model_unavailable")
            validate_model(
                self.directory
            )  # Also on cache hits: corrupt weights cannot hide behind cache.
            epoch = self.cache.epoch()
            chosen = result[: self.policy.pool]
            pairs = [pair_task(profile, v, owner=owner) for v, _ in chosen]
            values = [self.cache.get(owner, key) for key, _ in pairs]
            missing = [i for i, value in enumerate(values) if value is None]
            stats["cache_hits"] = len(values) - len(missing)
            if missing:
                remaining = self.deadline - (time.monotonic() - start)
                if remaining <= 0:
                    raise TimeoutError("pair_deadline")
                measured = score_bounded(
                    [pairs[i][1] for i in missing], self.directory, deadline=remaining
                )
                for i, value in zip(missing, measured["values"], strict=True):
                    values[i] = value
                stats["worker"] = {k: v for k, v in measured.items() if k != "values"}
            if self.cache.epoch() != epoch:
                raise ValueError("stale_pair_generation")
            if any(
                pair_task(profile, v, owner=owner)[0] != key
                for (v, _), (key, _) in zip(chosen, pairs, strict=True)
            ):
                raise ValueError("stale_pair_inputs")
            if time.monotonic() - start > self.deadline:
                raise TimeoutError("pair_deadline")
            for i in missing:
                self.cache.put(owner, pairs[i][0], values[i], epoch)
            return apply_pair_values(result, values, self.policy, [key for key, _ in pairs])
        except (ValueError, OSError, TimeoutError, KeyError, TypeError) as exc:
            # Payload-free fixed exception category; model input never appears in diagnostics.
            known = {
                "pair_request_busy",
                "structured_model_unavailable",
                "corrupt_pair_cache",
                "stale_pair_generation",
                "stale_pair_inputs",
                "invalid_pair_output",
                "invalid_pair_output_count",
                "pair_worker_failed",
                "pair_memory_budget",
                "insufficient_memory",
                "inference_slot_busy",
                "worker_resource_access_denied",
            }
            message = str(exc)
            stats["fallback"] = (
                "timeout"
                if isinstance(exc, TimeoutError)
                else "model_invalid"
                if message == "Missing or incompatible pinned encoder file"
                else message
                if message in known
                else type(exc).__name__
            )
            if message in {"stale_pair_generation", "stale_pair_inputs"}:
                # The structured baseline is stale too; never display it after deletion/edit.
                return []
            for _, match in result:
                if match.ranking:
                    match.ranking.reranker = {
                        "policy_id": self.policy.policy_id,
                        "fallback": stats["fallback"],
                        "mode": "structured_with_required_guard",
                    }
            return result
        finally:
            if acquired:
                self.lock.release()
            stats["seconds"] = time.monotonic() - start
            if trace is not None:
                trace["pair_reranker"] = stats


def apply_pair_values(result, values, policy, keys):
    """Shared deterministic transform for calibration and the real-inference adapter.

    Caller owns match copies; the structured tail retains its own admission policy.
    """
    chosen = result[: policy.pool]
    validate_values(values, len(chosen))
    for _, match in result:
        if match.ranking and required_guard(match):
            match.ranking.admission_passed = False
            match.ranking.admission_reason = "required_document_or_condition_unconfirmed"
    decorated = []
    for index, ((v, match), value) in enumerate(zip(chosen, values, strict=True)):
        decision = match.ranking.model_copy(deep=True)
        combined = policy.score(value["score"], decision.model_score)
        decision.reranker = {
            "policy_id": policy.policy_id,
            "encoder_id": MODEL_ID,
            "input_key": keys[index],
            "raw_logit": value["score"],
            "combined_score": combined,
            "windows": value["windows"],
            "pool": policy.pool,
            "alpha": policy.alpha,
            "score_meaning": "relevance_margin_not_hiring_probability",
            "structured_model_id": policy.structured_model_id,
            "mode": "recalibrated" if policy.admission else "order_only",
        }
        if policy.admission:
            decision.model_id = policy.policy_id
            decision.model_score = combined
            decision.contributions = {
                "cross_encoder": policy.alpha * (value["score"] - policy.ce_mean) / policy.ce_scale,
                "structured": (1 - policy.alpha)
                * (match.ranking.model_score - policy.linear_mean)
                / policy.linear_scale,
            }
            passed, reason, support = model_admission(match, combined, policy.admission)
            decision.admission_threshold = policy.admission.threshold
            decision.evidence_floor = policy.admission.evidence_floor
            decision.evidence_support = support
            decision.admission_passed = passed and not required_guard(match)
            decision.admission_reason = (
                "required_document_or_condition_unconfirmed" if required_guard(match) else reason
            )
        decorated.append((combined, index, v, match, decision))
    decorated.sort(key=lambda row: (-row[0], row[1]))
    for rank, (_, _, _, match, decision) in enumerate(decorated, 1):
        decision.model_rank = rank
        match.ranking = decision
    return [(v, m) for _, _, v, m, _ in decorated] + result[len(chosen) :]
