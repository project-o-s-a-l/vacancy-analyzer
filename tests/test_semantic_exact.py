"""Behavioral lifecycle/owner/refill checks for the isolated reference."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from threading import Event

import pytest
from test_candidates import candidate
from test_candidates import job as base_job

from domain.vacancy_duplicates import vacancy_duplicate_key
from jobs.semantic_documents import profile_document, vacancy_document
from jobs.semantic_exact import (
    QueryCache,
    exact_candidates,
    fuse,
    read_index,
    refresh_index,
    seal_index,
    write_index,
)

IDENTITY = {"model": "test-unit-fixture-not-real-inference"}


def job(vacancy, **updates):
    return base_job(vacancy, **({"published_at": datetime(2026, 10, 15, tzinfo=UTC)} | updates))


def embedding():
    vector = [1.0] + [0.0] * 383
    return dict(vector=vector, section_vector=vector.copy(), chunks=1, tokens=4, truncated=False)


def encode(tasks):
    return {"values": [embedding() for _ in tasks]}


def test_incremental_text_facts_model_and_deleted_membership(vacancy, now, tmp_path):
    first, second = job(vacancy), job(vacancy, title="Python engineer")
    index, counts = refresh_index([first, second], IDENTITY, encode)
    assert counts["computed"] == 2
    same, counts = refresh_index([first, second], IDENTITY, encode, previous=index)
    assert same == index and counts["computed"] == 0 and counts["reused"] == 2
    changed = first.model_copy(update={"title": "Senior Java developer"})
    selected, diagnostics = exact_candidates(
        [changed], candidate(), index, embedding()["vector"], identity=IDENTITY, now=now
    )
    assert not selected and diagnostics["stale_or_missing"] == 1
    updated, counts = refresh_index([changed], IDENTITY, encode, previous=index)
    assert counts["computed"] == 1 and counts["removed"] == 1
    assert len(updated["entries"]) == 1
    path = tmp_path / "index.json"
    write_index(path, updated)
    assert read_index(path, IDENTITY) == updated
    with pytest.raises(ValueError, match="incompatible"):
        read_index(path, {"model": "changed"})
    path.write_text("{}")
    with pytest.raises(ValueError):
        read_index(path, IDENTITY)


def test_fact_fence_refresh_reuses_unchanged_text_but_rejects_stale(vacancy, now):
    card = job(vacancy)
    document = vacancy_document(card)
    index, _ = refresh_index([card], IDENTITY, encode, document=lambda _: document)
    changed = replace(document, facts_revision="new_fact_revision")
    rows, stats = exact_candidates(
        [card],
        candidate(),
        index,
        embedding()["vector"],
        identity=IDENTITY,
        now=now,
        document=lambda _: changed,
    )
    assert rows == [] and stats["stale_or_missing"] == 1
    index, stats = refresh_index(
        [card], IDENTITY, encode, previous=index, document=lambda _: changed
    )
    assert stats["reused"] == 1 and stats["computed"] == 0
    rows, _ = exact_candidates(
        [card],
        candidate(),
        index,
        embedding()["vector"],
        identity=IDENTITY,
        now=now,
        document=lambda _: changed,
    )
    assert rows == [card]


@pytest.mark.parametrize("bad_index", [None, {}, {"version": "wrong"}])
def test_missing_corrupt_index_and_empty_dense_fallback(vacancy, now, bad_index):
    rows, stats = exact_candidates(
        [job(vacancy)], candidate(), bad_index, embedding()["vector"], identity=IDENTITY, now=now
    )
    assert not rows and stats["fallback_reason"] == "missing_or_corrupt_index"
    empty = seal_index({}, IDENTITY)
    rows, stats = exact_candidates(
        [job(vacancy)], candidate(), empty, embedding()["vector"], identity=IDENTITY, now=now
    )
    assert not rows and stats["fallback_reason"] == "empty_current_dense_pool"


@pytest.mark.parametrize("restriction", ["archived", "future", "expired", "experience", "remote"])
def test_required_conditions_are_not_overridden_by_perfect_similarity(vacancy, now, restriction):
    card = job(vacancy)
    profile = candidate(experience_years=0)
    if restriction == "archived":
        card.archived = True
    elif restriction == "future":
        card.published_at = now + timedelta(days=1)
    elif restriction == "expired":
        card.valid_until = now - timedelta(days=1)
    elif restriction == "experience":
        card.experience = "6+"
    else:
        card.remote = False
        profile.preferences.remote = "required"
    index, _ = refresh_index([card], IDENTITY, encode)
    rows, stats = exact_candidates(
        [card], profile, index, embedding()["vector"], identity=IDENTITY, now=now
    )
    assert not rows and stats["filtered"] == 1


def test_full_exact_refills_after_history_filter_and_duplicate_groups(vacancy, now):
    cards = [job(vacancy, title=f"Java role {i}", company=f"Company{i}") for i in range(80)]
    duplicate = job(cards[-1])
    cards.append(duplicate)
    index, _ = refresh_index(cards, IDENTITY, encode)
    rows, stats = exact_candidates(
        cards,
        candidate(),
        index,
        embedding()["vector"],
        identity=IDENTITY,
        now=now,
        k=10,
        excluded_ids={v.id for v in cards[:65]},
        excluded_keys={vacancy_duplicate_key(cards[65])},
    )
    assert len(rows) == 10
    assert len({vacancy_duplicate_key(v) for v in rows}) == 10
    assert stats["filtered"] == 66 and stats["scanned"] == 15
    assert stats["exhausted"] is False


def test_fusion_reserve_keeps_actual_new_groups_without_increasing_budget(vacancy):
    cards = [job(vacancy, title=f"Distinct role {i}") for i in range(120)]
    lex, dense = cards[:64], cards[56:120]
    for method, quota in (("rrf", 0), ("novelty", 0.125), ("novelty", 0.25)):
        result = fuse(lex, dense, method=method, quota=quota)
        assert len(result) == len({v.id for v in result}) == 64
        if method == "novelty":
            assert len({v.id for v in result} - {v.id for v in lex}) == int(64 * quota)
    assert fuse(lex, []) == lex


def test_private_cache_owner_revision_model_and_deletion():
    cache = QueryCache(capacity=2)
    doc = profile_document(candidate())
    value, hit = cache.get("owner1", doc, IDENTITY, encode)
    assert not hit
    value["vector"][0] = 0
    value, hit = cache.get("owner1", doc, IDENTITY, encode)
    assert hit and value["vector"][0] == 1
    assert cache.get("owner2", doc, IDENTITY, encode)[1] is False
    assert cache.get("owner1", replace(doc, input_revision="new"), IDENTITY, encode)[1] is False
    assert cache.get("owner1", doc, {"model": "new"}, encode)[1] is False
    cache.delete("owner1")
    assert cache.get("owner1", doc, IDENTITY, encode)[1] is False
    for i in range(20):
        cache.delete(str(i))
    assert len(cache._owners) <= 4 and len(cache._values) <= 2


def test_inflight_query_cannot_resurrect_deleted_profile():
    cache = QueryCache()
    started, release = Event(), Event()

    def delayed(tasks):
        started.set()
        assert release.wait(5)
        return encode(tasks)

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(cache.get, "owner", profile_document(candidate()), IDENTITY, delayed)
        assert started.wait(5)
        cache.delete("owner")
        release.set()
        with pytest.raises(ValueError, match="changed_during"):
            pending.result(timeout=5)
    assert not cache._values


def test_close_technology_names_have_no_dense_only_eligibility_heuristic(vacancy, now):
    cards = [
        job(vacancy, title="Java Developer"),
        job(
            vacancy,
            title="JavaScript Developer",
            skills=["JavaScript", "React"],
            description="Обязанности:\nСоздание интерфейсов на JavaScript React",
        ),
    ]
    index, _ = refresh_index(cards, IDENTITY, encode)
    selected, _ = exact_candidates(
        cards, candidate(), index, embedding()["vector"], identity=IDENTITY, now=now, k=2
    )
    assert {v.id for v in selected} == {v.id for v in cards}
    # Dense similarity is not evidence of skill equality. The ranker gets the exact current facts.
    from scoring import rank_vacancies

    ranked = rank_vacancies(selected, candidate(), now=now)
    assert ranked[0][0].title == "Java Developer"
    assert ranked[0][1].score > ranked[1][1].score


def test_rrf_combines_rank_evidence_across_duplicate_representatives(vacancy):
    a = job(vacancy, title="Same role")
    copy = job(a, title=a.title)
    b = job(vacancy, title="Different role")
    assert vacancy_duplicate_key(a) == vacancy_duplicate_key(copy)
    result = fuse([a, b], [copy], k=2)
    assert result == [a, b]
