"""train/development/calibration on the frozen, user-visible retrieval path.

Final test is never selected. The fitted margin has no probabilistic interpretation.
"""

import json
import math
import random
import statistics
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from domain.contracts import normalization_contract
from domain.matching_dataset import MatchingDataset, ReviewPair
from domain.retrieval import RetrievalParameters
from domain.serving_ranker import (
    ModelAdmission,
    RankerPolicy,
    RankingDecision,
    ServingRankerArtifact,
)
from jobs.admission import admit
from jobs.evaluation import _labels, extract_facts, source_revision, stand
from jobs.evaluation_data import load_data
from jobs.evaluation_labels import resolve_labels
from jobs.rank import default_retrieval_parameters
from jobs.recommendations import recommend
from jobs.retrieval_evaluation import MemoryCandidateSource
from scoring.admission_policy import PROTOCOL, model_admission, policy_contract
from scoring.evaluation_metrics import catalog_metrics, interval, wilson
from scoring.linear_ranker import (
    FEATURE_NAMES,
    LabeledQuery,
    feature_vector,
    fit_pairwise_ranker,
    ranker_scores,
)
from scoring.serving_features import (
    FEATURE_SPEC,
    digest,
    feature_contract,
    pair_features,
    retrieval_contract,
)
from scoring.serving_ranker import activation_reasons, load_artifact

SPLITS = ("train", "development", "calibration")


def _experience(profile):
    years = profile.experience_years
    return "unknown" if years is None else "0" if years == 0 else "1-3" if years <= 3 else "3+"


async def prepare(bundle, *, ledger=None, draft_labels=None, allow_assistant_draft=False):
    if not (bundle / "data.json").is_file():
        raise ValueError(
            "Нужна проверенная разметка с train/development/calibration; legacy bundle запрещён"
        )
    data, protocol, bundle_hash = load_data(bundle)
    resolved, imports = resolve_labels(ledger, bundle_hash)
    origin = "human"
    if draft_labels is not None:
        if not allow_assistant_draft or ledger is not None:
            raise ValueError("Draft requires explicit flag and cannot mix with a human ledger")
        draft = json.loads(draft_labels.read_text(encoding="utf-8"))
        if draft.get("bundle_hash") != bundle_hash or draft.get("origin") != "assistant_draft":
            raise ValueError("Draft provenance mismatch")
        resolved, origin = draft["records"], "assistant_draft"
    required = {
        f"pair:{p.id}:{v.id}"
        for p in data.profiles
        if p.split in SPLITS
        for v in data.vacancies
        if v.split == p.split
    }
    if not required <= resolved.keys():
        raise ValueError(
            "Нужна полная проверенная разметка train/development/calibration; unknown не negative"
        )
    # Validate only non-test labels, never inspect or score final-test judgments.
    for key in required:
        row = resolved[key]
        if (
            type(row.get("relevance")) is not int
            or row["relevance"] not in range(4)
            or type(row.get("violates_required")) is not bool
        ):
            raise ValueError("Invalid complete pair label")
    prepared = {s: [] for s in SPLITS}
    parameters = default_retrieval_parameters()
    for split in SPLITS:
        vacancies = [v for v in data.vacancies if v.split == split]
        groups = {v.id: v.duplicate_group for v in vacancies}
        source = MemoryCandidateSource([v.vacancy for v in vacancies])
        now = data.reference_times[split]
        for case in (p for p in data.profiles if p.split == split):
            labels, violations = _labels(case, vacancies, resolved)
            if len(labels) != len(set(groups.values())):
                raise ValueError("Duplicate judgments disagree; adjudicate before fitting")
            for mode in protocol["modes"]:
                profile = (
                    case.profile
                    if mode == "confirmed_profile"
                    else extract_facts(case, reference_time=now)[1]
                )
                trace = {}
                started = perf_counter()
                result = await recommend(
                    None,
                    profile,
                    now=now,
                    candidate_source=source,
                    retrieval_parameters=parameters,
                    trace=trace,
                )
                ranking = result.ranking
                query_id = digest([case.id, mode])
                pairs = [
                    ReviewPair(
                        pair_id=digest([query_id, str(v.id)]),
                        vacancy=v,
                        baseline_rank=i,
                        features=pair_features(v, profile, m),
                    )
                    for i, (v, m) in enumerate(ranking, 1)
                ]
                dataset = (
                    MatchingDataset(
                        created_at=now,
                        query_id=query_id,
                        query_source="profile",
                        profile=profile,
                        feature_spec=FEATURE_SPEC,
                        corpus_size=len(pairs),
                        word_known_fraction=1,
                        warnings=[],
                        pairs=pairs,
                    )
                    if pairs
                    else None
                )
                prepared[split].append(
                    SimpleNamespace(
                        case=case,
                        mode=mode,
                        profile=profile,
                        ranking=ranking,
                        dataset=dataset,
                        labels=labels,
                        violations=violations,
                        groups=groups,
                        source=source,
                        now=now,
                        trace=trace,
                        seconds=perf_counter() - started,
                    )
                )
    provenance = {
        "bundle_hash": bundle_hash,
        "label_origins": [origin],
        "label_imports": imports["imports"],
        "labels_sha256": digest({k: resolved[k] for k in sorted(required)}),
        "draft_file_sha256": sha256(draft_labels.read_bytes()).hexdigest()
        if draft_labels
        else None,
        "splits": {
            s: {
                "owners": sorted({p.owner for p in data.profiles if p.split == s}),
                "profiles": sorted(
                    {
                        digest(p.profile.model_dump(mode="json"))
                        for p in data.profiles
                        if p.split == s
                    }
                ),
                "vacancy_groups": sorted(
                    {v.duplicate_group for v in data.vacancies if v.split == s}
                ),
                "reference_time": data.reference_times[s].isoformat(),
            }
            for s in SPLITS
        },
        "test_used": False,
        "retrieval_parameters": parameters.model_dump(mode="json"),
        "training_parameters": PROTOCOL,
        "scope": "Synthetic data; retrieval, both profile modes; no market generalization",
    }
    return prepared, provenance


def _canonical(ranking, groups):
    return list(dict.fromkeys(groups[f"{v.source}:{v.source_id}"] for v, _ in ranking))


def predictions(query, model):
    if not query.dataset:
        return []
    scores = ranker_scores(model, query.dataset)
    by_id = {p.vacancy.id: scores[p.pair_id] for p in query.dataset.pairs}
    return sorted([(v, m, by_id[v.id]) for v, m in query.ranking], key=lambda x: -x[2])


def measured(query, predicted, policy):
    ranking = []
    for i, (vacancy, original, score) in enumerate(predicted, 1):
        match = original.model_copy(deep=True)
        passed, reason, support = model_admission(match, score, policy)
        match.ranking = RankingDecision(
            requested_mode="model",
            applied=True,
            baseline_rank=i,
            model_score=score,
            admission_policy=policy.version,
            admission_threshold=policy.threshold,
            evidence_floor=policy.evidence_floor,
            evidence_support=support,
            admission_passed=passed,
            admission_reason=reason,
            experimental=True,
        )
        ranking.append((vacancy, match))
    _, admitted, displayed = admit(ranking)
    _, base_admitted, base_displayed = admit(query.ranking)
    ids = sorted(set(query.groups.values()))
    result = {
        "owner": query.case.owner,
        "case": query.case.id,
        "mode": query.mode,
        "profession": query.case.profession,
        "experience": _experience(query.profile),
    }
    for name, rows, accepted in (
        ("baseline", base_displayed, base_admitted),
        ("model", displayed, admitted),
    ):
        order = _canonical(rows, query.groups)
        eligible = _canonical(accepted, query.groups)
        metrics = catalog_metrics(order, query.labels, ids)
        metrics.update(
            slots=len(order),
            nonempty=int(bool(order)),
            retained_positives=sum(query.labels[g] >= 2 for g in eligible),
        )
        result[name] = metrics
        result[name + "_order"] = order
        result[name + "_admitted"] = eligible
    result["rank_delta"] = (
        catalog_metrics(_canonical(ranking, query.groups), query.labels, ids)["ndcg_at_5"] or 0
    ) - (
        catalog_metrics(_canonical(query.ranking, query.groups), query.labels, ids)["ndcg_at_5"]
        or 0
    )
    positives = {g for g, grade in query.labels.items() if grade >= 2}
    retrieved = {query.groups[k] for k in query.trace["retrieved"]}
    result["losses"] = {
        "positive_catalog": len(positives),
        "retrieval": sorted(positives - retrieved),
        "admission": sorted(positives & retrieved - set(result["model_admitted"])),
        "top": sorted(positives & set(result["model_admitted"]) - set(result["model_order"])),
    }
    result["violations"] = sum(query.violations[g] for g in result["model_order"])
    result["baseline_violations"] = sum(query.violations[g] for g in result["baseline_order"])
    return result


def _interval(rows, value):
    owners = defaultdict(list)
    for row in rows:
        item = value(row)
        if item is not None:
            owners[row["owner"]].append(item)
    return interval(
        [statistics.mean(v) for v in owners.values()],
        seed=PROTOCOL["seed"],
        repeats=PROTOCOL["bootstrap_repeats"],
        confidence=PROTOCOL["confidence"],
    )


def precision_interval(rows, method):
    owners = defaultdict(lambda: [0, 0])
    for row in rows:
        counts = owners[row["owner"]]
        counts[0] += row[method]["relevant_found"]
        counts[1] += row[method]["slots"]
    groups = list(owners.values())
    total = sum(d for _, d in groups)
    mean = sum(n for n, _ in groups) / total if total else None
    result = {"mean": mean, "low": None, "high": None, "clusters": len(groups)}
    if len(groups) < 2 or not total:
        return result
    rng = random.Random(PROTOCOL["seed"])
    samples = []
    for _ in range(PROTOCOL["bootstrap_repeats"]):
        sample = rng.choices(groups, k=len(groups))
        denominator = sum(d for _, d in sample)
        # Empty bootstrap samples cannot establish a precision floor.
        samples.append(sum(n for n, _ in sample) / denominator if denominator else 0.0)
    samples.sort()
    tail = (1 - PROTOCOL["confidence"]) / 2
    result.update(
        low=samples[int(tail * len(samples))],
        high=samples[math.ceil((1 - tail) * len(samples)) - 1],
    )
    return result


def summarize(rows, *, strata=True):
    result = {
        "owners": len({r["owner"] for r in rows}),
        "queries": len(rows),
        "violations": sum(r["violations"] for r in rows),
        "baseline_violations": sum(r["baseline_violations"] for r in rows),
        "rank_delta": _interval(rows, lambda r: r["rank_delta"]),
        "precision_micro": wilson(
            sum(r["model"]["relevant_found"] for r in rows), sum(r["model"]["slots"] for r in rows)
        ),
        "precision_cluster": precision_interval(rows, "model"),
        "deltas": {},
    }
    for name in (
        "ndcg_at_5",
        "precision_at_5",
        "recall",
        "slots",
        "nonempty",
        "retained_positives",
    ):
        result["deltas"][name] = _interval(
            rows,
            lambda r, name=name: (
                r["model"][name] - r["baseline"][name]
                if r["model"][name] is not None and r["baseline"][name] is not None
                else None
            ),
        )
    for method in ("baseline", "model"):
        result[method] = {
            "shown": sum(r[method]["slots"] for r in rows),
            "slots": 5 * len(rows),
            "empty": wilson(sum(not r[method]["nonempty"] for r in rows), len(rows)),
            "precision_micro": wilson(
                sum(r[method]["relevant_found"] for r in rows),
                sum(r[method]["slots"] for r in rows),
            ),
            "precision_cluster": precision_interval(rows, method),
            "macro": {
                key: _interval(rows, lambda r, method=method, key=key: r[method][key])
                for key in ("ndcg_at_5", "precision_at_5", "recall", "retained_positives")
            },
        }
    result["positive_losses"] = {
        k: sum(len(r["losses"][k]) for r in rows) for k in ("retrieval", "admission", "top")
    }
    if strata:
        for dimension in ("profession", "experience", "mode"):
            result["by_" + dimension] = {
                g: summarize([r for r in rows if r[dimension] == g], strata=False)
                for g in sorted({r[dimension] for r in rows})
            }
    return result


def calibrate(queries, model):
    predicted = [predictions(q, model) for q in queries]
    scores = sorted(s for p in predicted for _, _, s in p)
    if not scores:
        raise ValueError("Calibration has no retrieved candidates")
    thresholds = sorted({scores[int(q * (len(scores) - 1))] for q in PROTOCOL["score_quantiles"]})
    thresholds.append(
        scores[-1] + 1.0
    )  # Explicit empty policy is measured, never awarded fake precision=1.
    sweep = []
    for floor in PROTOCOL["evidence_floors"]:
        for threshold in thresholds:
            policy = ModelAdmission(threshold=threshold, evidence_floor=floor)
            rows = [measured(q, p, policy) for q, p in zip(queries, predicted, strict=True)]
            shown = sum(r["model"]["slots"] for r in rows)
            found = sum(r["model"]["relevant_found"] for r in rows)
            feasible = bool(shown) and found / shown >= PROTOCOL["precision_floor"]
            # Do not optimize precision by quietly shrinking coverage or losing positives.
            feasible = feasible and all(
                sum(r["model"][m] for r in rows) >= sum(r["baseline"][m] for r in rows)
                for m in ("slots", "nonempty", "retained_positives")
            )
            sweep.append(
                {
                    "policy": policy.model_dump(),
                    "precision": found / shown if shown else None,
                    "shown": shown,
                    "found": found,
                    "feasible_point_estimate": feasible,
                    "empty_queries": sum(not r["model"]["nonempty"] for r in rows),
                    "admission_positive_losses": sum(len(r["losses"]["admission"]) for r in rows),
                }
            )
    # Infeasible best candidate remains diagnostic and fails the numerical release gates.
    winner = max(
        sweep,
        key=lambda r: (
            r["feasible_point_estimate"],
            r["found"],
            r["precision"] or 0,
            r["shown"],
            -r["policy"]["evidence_floor"],
            -r["policy"]["threshold"],
        ),
    )
    return ModelAdmission.model_validate(winner["policy"]), sweep


async def replay(queries, artifact):
    """End-to-end parity: runtime features, candidates, ordered admission and displays."""
    with tempfile.TemporaryDirectory(prefix="f10-replay-") as directory:
        path = Path(directory) / "model.json"
        path.write_text(artifact.model_dump_json(), encoding="utf-8")
        for query in queries:
            trace = {}
            result = await recommend(
                None,
                query.profile,
                now=query.now,
                candidate_source=query.source,
                ranker_policy=RankerPolicy(mode="model", model_path=path),
                experimental_ranker=True,
                retrieval_parameters=artifact.retrieval_parameters,
                trace=trace,
            )
            expected = measured(query, predictions(query, artifact.model), artifact.admission)
            if (
                trace["retrieved"] != query.trace["retrieved"]
                or _canonical(result.displayed, query.groups) != expected["model_order"]
                or _canonical(result.admitted, query.groups) != expected["model_admitted"]
            ):
                raise ValueError("Offline/runtime candidate/order/admission mismatch")
            offline = {v.id: (m, score) for v, m, score in predictions(query, artifact.model)}
            for vacancy, match in result.ranking:
                original, score = offline[vacancy.id]
                if (
                    not match.ranking
                    or not match.ranking.applied
                    or match.ranking.model_score != score
                    or pair_features(vacancy, query.profile, match)
                    != pair_features(vacancy, query.profile, original)
                ):
                    raise ValueError("Offline/runtime feature/score mismatch")


def fit_candidates(prepared):
    training = []
    for q in prepared["train"]:
        if q.dataset:
            labels = {
                p.pair_id: q.labels[q.groups[f"{p.vacancy.source}:{p.vacancy.source_id}"]]
                for p in q.dataset.pairs
            }
            if len(set(labels.values())) > 1:
                training.append(LabeledQuery(q.dataset, labels, q.case.owner))
    from threadpoolctl import threadpool_limits

    candidates = []
    for c in PROTOCOL["C"]:
        with threadpool_limits(limits=1):
            model = fit_pairwise_ranker(training, c=c, professional_only=True)
        rows = [
            measured(q, predictions(q, model), ModelAdmission(threshold=-1e100, evidence_floor=0))
            for q in prepared["development"]
        ]
        candidates.append((statistics.mean(r["rank_delta"] for r in rows), c, model, rows))
    _, selected_c, model, _ = max(candidates, key=lambda v: (v[0], -v[1]))
    return selected_c, model, candidates, training


async def train(bundle, **options):
    started = perf_counter()
    prepared, provenance = await prepare(bundle, **options)
    selected_c, model, candidates, training = fit_candidates(prepared)
    admission, sweep = calibrate(prepared["calibration"], model)
    development = [measured(q, predictions(q, model), admission) for q in prepared["development"]]
    calibration = [measured(q, predictions(q, model), admission) for q in prepared["calibration"]]
    proof = {
        "protocol_hash": digest(PROTOCOL),
        "development": summarize(development),
        "calibration": summarize(calibration),
        "selected_C": selected_c,
    }
    artifact = ServingRankerArtifact.seal(
        created_at=datetime.now(UTC),
        feature_contract=feature_contract(),
        normalization_contract=normalization_contract(),
        policy_contract=policy_contract(),
        retrieval_contract=retrieval_contract(),
        retrieval_parameters=RetrievalParameters.model_validate(provenance["retrieval_parameters"]),
        model=model,
        admission=admission,
        provenance=provenance,
        validation=proof,
    )
    await replay(prepared["development"] + prepared["calibration"], artifact)
    return artifact, {
        "version": "f10-ranker-report-v1",
        "model_id": artifact.model_id,
        "protocol": PROTOCOL,
        "provenance": provenance,
        "validation": proof,
        "development_candidates": [
            {"C": c, "rank_delta": summarize(rows)["rank_delta"]} for _, c, _, rows in candidates
        ],
        "calibration_sweep": sweep,
        "cases": {"development": development, "calibration": calibration},
        "activation_blockers": activation_reasons(artifact),
        "activated": False,
        "quality": "experimental"
        if provenance["label_origins"] != ["human"]
        else "candidate_for_F14",
        "parity_queries": len(prepared["development"]) + len(prepared["calibration"]),
        "training_comparisons": model.training_comparisons,
        "feature_support": {
            name: {
                "distinct_train_values": len(
                    {feature_vector(p.features)[i] for q in training for p in q.dataset.pairs}
                ),
                "coefficient": model.coefficients[i],
            }
            for i, name in enumerate(FEATURE_NAMES)
        },
        "source": source_revision(),
        "stand": stand(),
        "elapsed_seconds": perf_counter() - started,
        "final_test_used": False,
    }


async def check(bundle, model_path, **options):
    artifact, reason = load_artifact(model_path, default_retrieval_parameters())
    if artifact is None:
        raise ValueError("Artifact unavailable: " + reason)
    prepared, provenance = await prepare(bundle, **options)
    if artifact.provenance != provenance:
        raise ValueError("Training data/labels provenance mismatch; retraining required")
    selected_c, fitted, _, _ = fit_candidates(prepared)
    for key in (
        "coefficients",
        "means",
        "scales",
        "training_query_ids",
        "training_groups",
        "training_comparisons",
    ):
        if getattr(fitted, key) != getattr(artifact.model, key):
            raise ValueError("Model does not reproduce train-only fitting")
    if selected_c != artifact.validation["selected_C"]:
        raise ValueError("Model does not reproduce development selection")
    admission, _ = calibrate(prepared["calibration"], artifact.model)
    if admission != artifact.admission:
        raise ValueError("Admission does not match measured calibration selection")
    rows = {
        s: [measured(q, predictions(q, artifact.model), artifact.admission) for q in prepared[s]]
        for s in ("development", "calibration")
    }
    proof = {
        "protocol_hash": digest(PROTOCOL),
        "development": summarize(rows["development"]),
        "calibration": summarize(rows["calibration"]),
        "selected_C": artifact.validation["selected_C"],
    }
    if (
        proof["development"] != artifact.validation["development"]
        or proof["calibration"] != artifact.validation["calibration"]
    ):
        raise ValueError("Measured proof differs from artifact")
    await replay(prepared["development"] + prepared["calibration"], artifact)
    return {
        "version": "f10-check-v1",
        "model_id": artifact.model_id,
        "recomputed_from_data": True,
        "activation_blockers": activation_reasons(artifact),
        "final_test_used": False,
        "validation": proof,
    }
