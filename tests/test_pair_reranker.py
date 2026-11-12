"""behavioural contracts. Neural quality is measured separately with real weights."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from uuid import UUID

import numpy as np
import pytest

from domain.comparison import ComparisonItem
from domain.serving_ranker import ModelAdmission, RankingDecision
from jobs import cross_encoder as encoder
from jobs import pair_reranker as module
from jobs.admission import admit
from jobs.recommendations import recommend
from jobs.retrieval_evaluation import MemoryCandidateSource
from scoring import rank_vacancies
from scoring.serving_features import shown_context


class Tokens:
    def __init__(self):
        self.words = {}

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return SimpleNamespace(
            ids=[self.words.setdefault(w, len(self.words) + 3) for w in text.split()]
        )

    def token_to_id(self, value):
        return {"<s>": 0, "</s>": 2}[value]


def policy(**kwargs):
    return module.PairPolicy.seal(
        **(
            dict(
                structured_model_id="linear",
                pool=8,
                alpha=1.0,
                ce_mean=0,
                ce_scale=1,
                linear_mean=0,
                linear_scale=1,
                admission=ModelAdmission(threshold=1, evidence_floor=0),
            )
            | kwargs
        )
    )


def ranking(vacancy, profile, count=2):
    cards = [
        vacancy.model_copy(
            update={
                "id": UUID(int=i + 1),
                "source_id": str(i + 1),
                "title": f"Python developer {i}",
            }
        )
        for i in range(count)
    ]
    rows = rank_vacancies(cards, profile, include_ineligible=True)
    for i, (_, match) in enumerate(rows):
        match.eligible = True
        match.comparison.items = [
            ComparisonItem(
                id="skill",
                name="python",
                kind="skill",
                importance="required",
                state="match",
                detail="evidence",
            )
        ]
        match.ranking = RankingDecision(
            requested_mode="model",
            applied=True,
            model_id="linear",
            baseline_rank=i + 1,
            model_rank=i + 1,
            model_score=2 - i,
            admission_policy="rank-evidence-admission-v1",
            admission_threshold=1,
            admission_passed=i == 0,
            admission_reason="model_rank_and_evidence",
            experimental=True,
        )
    return rows


def value(score):
    return {"score": score, "windows": 1, "truncated": False}


def test_pair_windows_preserve_long_tail_and_required_label():
    tokenizer = Tokens()
    task = {
        "candidate": ["Candidate performed task: " + "build " * 280 + "CANDIDATE_TAIL"],
        "vacancy": [
            "Vacancy requirement: Required qualification: " + "document " * 280 + "REQUIRED_TAIL"
        ],
    }
    windows = encoder.pair_windows(tokenizer, task)
    assert len(windows) == 4
    assert max(len(ids) for ids, _ in windows) <= 512
    assert any(tokenizer.words["CANDIDATE_TAIL"] in ids for ids, _ in windows)
    assert all(tokenizer.words["Required"] in ids for ids, _ in windows)
    assert any(tokenizer.words["REQUIRED_TAIL"] in ids for ids, _ in windows)
    assert all(ids[0] == 0 and ids[-1] == 2 for ids, _ in windows)


def test_or_group_and_extreme_long_inputs_fail_without_partial_coverage():
    with pytest.raises(ValueError, match="pair_input_budget"):
        encoder.pair_windows(Tokens(), {"candidate": ["x" * 100001], "vacancy": ["y" * 100000]})
    with pytest.raises(ValueError, match="alternative_group_overflow"):
        encoder.document_windows(Tokens(), ["Vacancy alternatives (any one): " + "A OR " * 200])
    with pytest.raises(ValueError, match="pair_window_budget"):
        encoder.pair_windows(Tokens(), {"candidate": ["x " * 1500], "vacancy": ["y " * 1500]})


def test_batch_mask_uses_pair_boundary_and_raw_finite_logits():
    local = encoder.LocalCrossEncoder.__new__(encoder.LocalCrossEncoder)
    local.np, local.tokenizer = np, Tokens()
    feeds = []

    def run(_, feed):
        feeds.append(feed)
        return [np.arange(feed["input_ids"].shape[0], dtype=float).reshape(-1, 1)]

    local.session = SimpleNamespace(run=run)
    result = local.score([{"candidate": ["one"], "vacancy": ["two " * (i + 1)]} for i in range(10)])
    assert len(result) == 10 and max(f["input_ids"].shape[0] for f in feeds) == 1
    assert feeds[0]["input_ids"][0, :6].tolist() == [0, 3, 2, 2, 4, 2]
    assert feeds[0]["attention_mask"].sum() == 6
    assert result[0]["score"] == 0  # Not sigmoid .5 or a hiring probability.
    local.session.run = lambda *_: [np.array([[float("nan")]])]
    with pytest.raises(ValueError, match="invalid_logits"):
        local.score([{"candidate": ["x"], "vacancy": ["y"]}])


def test_corrupt_weights_and_policy_are_rejected(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="pinned"):
        encoder.validate_model(tmp_path)
    raw = policy().model_dump()
    raw["alpha"] = 0.5
    with pytest.raises(ValueError, match="checksum"):
        module.PairPolicy.model_validate(raw)
    with pytest.raises(ValueError):
        policy(serving_allowed=True)


def test_cache_owner_integrity_capacity_and_late_deletion():
    cache = module.PairCache(capacity=2)
    epoch = cache.epoch()
    cache.put("one", "key", value(2), epoch)
    assert cache.get("two", "key") is None
    got = cache.get("one", "key")
    got["score"] = 99
    assert cache.get("one", "key")["score"] == 2
    cache.rows[("one", "key")][0]["score"] = 99
    with pytest.raises(ValueError, match="corrupt_pair_cache"):
        cache.get("one", "key")
    cache.put("one", "a", value(1), epoch)
    cache.put("two", "b", value(1), epoch)
    cache.put("three", "c", value(1), epoch)
    assert cache.get("one", "a") is None
    cache.clear_owner("two")
    assert cache.get("two", "b") is None
    with pytest.raises(ValueError, match="stale_pair_generation"):
        cache.put("two", "late", value(1), epoch)


def test_pair_key_versions_inputs_owner_and_no_contacts(vacancy, profile):
    a, task = module.pair_task(profile, vacancy, owner="one")
    assert a != module.pair_task(profile, vacancy, owner="two")[0]
    profile.experience_years = 50
    assert a != module.pair_task(profile, vacancy, owner="one")[0]
    old = module.pair_task(profile, vacancy, owner="one")[0]
    vacancy.description = "Обязанности:\nПродавать банковские продукты.\nКонтакты: x@example.com"
    assert old != module.pair_task(profile, vacancy, owner="one")[0]
    assert "x@example.com" not in str(module.pair_task(profile, vacancy, owner="one")[1])
    assert "not confirmed" in str(task)


@pytest.mark.parametrize("state", ["unknown", "conflict", "contradiction"])
def test_required_document_never_bought_by_logit_or_timeout(vacancy, profile, monkeypatch, state):
    rows = ranking(vacancy, profile, count=1)
    rows[0][1].comparison.items.append(
        ComparisonItem(
            id="doc",
            name="licence",
            kind="qualification",
            importance="required",
            state=state,
            detail="missing",
        )
    )
    adapter = module.PairReranker(policy(), ".")
    monkeypatch.setattr(module, "validate_model", lambda *_: None)
    monkeypatch.setattr(module, "score_bounded", lambda *_args, **_kwargs: {"values": [value(1e8)]})
    assert admit(adapter.apply(rows, profile, owner="one"), min_score=0)[2] == []
    adapter.cache.clear_owner("one")

    def timeout(*args, **kwargs):
        raise TimeoutError()

    monkeypatch.setattr(module, "score_bounded", timeout)
    assert admit(adapter.apply(rows, profile, owner="one"), min_score=0)[2] == []
    assert rows[0][1].ranking.admission_passed  # Does not mutate the baseline.


def test_timeout_keeps_bad_structured_card_rejected_and_order(vacancy, profile, monkeypatch):
    rows = ranking(vacancy, profile)
    monkeypatch.setattr(module, "validate_model", lambda *_: None)

    def timeout(*args, **kwargs):
        raise TimeoutError()

    monkeypatch.setattr(module, "score_bounded", timeout)
    trace = {}
    result = module.PairReranker(policy(), ".").apply(rows, profile, owner="one", trace=trace)
    assert [v.id for v, _ in result] == [v.id for v, _ in rows]
    assert [v.id for v, _ in admit(result)[2]] == [rows[0][0].id]
    assert trace["pair_reranker"]["fallback"] == "timeout"


def test_rerank_updates_calibrated_admission_and_exact_snapshot(vacancy, profile, monkeypatch):
    rows = ranking(vacancy, profile)
    monkeypatch.setattr(module, "validate_model", lambda *_: None)
    monkeypatch.setattr(
        module, "score_bounded", lambda *_args, **_kwargs: {"values": [value(-3), value(5)]}
    )
    p = policy()
    result = module.PairReranker(p, ".").apply(rows, profile, owner="one")
    shown = admit(result)[2]
    assert [v.id for v, _ in shown] == [rows[1][0].id]
    v, m = shown[0]
    snapshot = shown_context(v, profile, m, 1)
    assert snapshot["decision"]["model_id"] == p.policy_id
    assert snapshot["decision"]["reranker"]["raw_logit"] == 5
    assert snapshot["admission"]["threshold"] == 1
    assert snapshot["display_rank"] == 1
    assert rows[0][1].ranking.model_score == 2


def test_outside_pool_keeps_structured_admission(vacancy, profile):
    rows = ranking(vacancy, profile, count=10)
    rows[9][1].ranking.admission_passed = True
    result = module.apply_pair_values(rows, [value(-100)] * 8, policy(), ["k"] * 8)
    assert result[-1][1].ranking.model_id == "linear"
    assert [v.id for v, _ in admit(result)[2]] == [rows[9][0].id]


def test_concurrent_request_falls_back_and_late_delete_discards_result(
    vacancy, profile, monkeypatch
):
    rows = ranking(vacancy, profile)
    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr(module, "validate_model", lambda *_: None)

    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return {"values": [value(0), value(9)]}

    monkeypatch.setattr(module, "score_bounded", delayed)
    adapter = module.PairReranker(policy(), ".", deadline=10)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(adapter.apply, rows, profile, owner="one")
        assert entered.wait(5)
        second = adapter.apply(rows, profile, owner="two")
        assert [v.id for v, _ in admit(second)[2]] == [rows[0][0].id]
        adapter.cache.clear_owner("one")
        release.set()
        late = first.result(5)
    assert late == []
    assert not adapter.cache.rows


def test_edit_during_inference_discards_stale_baseline_too(vacancy, profile, monkeypatch):
    rows = ranking(vacancy, profile)
    monkeypatch.setattr(module, "validate_model", lambda *_: None)

    def edited(*args, **kwargs):
        rows[0][0].title = "Changed role after inference started"
        return {"values": [value(0), value(9)]}

    monkeypatch.setattr(module, "score_bounded", edited)
    trace = {}
    result = module.PairReranker(policy(), ".").apply(rows, profile, owner="one", trace=trace)
    assert result == []
    assert trace["pair_reranker"]["fallback"] == "stale_pair_inputs"


async def test_evaluator_rejects_pair_policy_on_final_test(monkeypatch):
    from jobs import evaluation

    monkeypatch.setattr(evaluation, "load_data", lambda _: (None, None, None))
    with pytest.raises(ValueError, match="final test"):
        await evaluation.evaluate(None, split="test", pair_reranker=object())


async def test_offline_gate_prevents_live_or_implicit_use(vacancy, profile):
    adapter = module.PairReranker(policy(), ".")
    with pytest.raises(ValueError, match="offline experiment"):
        await recommend(
            None,
            profile,
            candidate_source=MemoryCandidateSource([vacancy]),
            pair_reranker=adapter,
            pair_owner="one",
        )
    with pytest.raises(ValueError, match="frozen offline"):
        await recommend(
            object(), profile, experimental_ranker=True, pair_reranker=adapter, pair_owner="one"
        )
