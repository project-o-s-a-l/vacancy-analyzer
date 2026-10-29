import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_candidates import candidate, job

from domain.retrieval import RetrievalParameters
from domain.vacancy_duplicates import vacancy_duplicate_key
from jobs.hybrid_candidates import hybrid_candidates, reciprocal_rank_fusion
from jobs.semantic_index import build_index, load_index, save_index, valid_vector
from jobs.semantic_model import MODEL_ID, MODEL_SPEC, download_model
from jobs.semantic_text import TEXT_SPEC, profile_text, text_revision, vacancy_text


def vector(index=0):
    v = [0.0] * 384
    v[index] = 1.0
    return v


def index_for(cards):
    return {
        "version": "semantic-index-v1",
        "model_id": MODEL_ID,
        "text_spec": TEXT_SPEC,
        "entries": {
            f"{v.source}:{v.source_id}": {
                "input_revision": text_revision(vacancy_text(v)),
                "vector": vector(),
            }
            for v in cards
        },
    }


def test_professional_text_uses_confirmed_fields_and_strips_contacts(vacancy):
    p = candidate(
        skills={"java": {}, "me@example.com": {}},
        preferences={"positions": ["Java Developer"], "locations": ["SecretCity"]},
    )
    assert "SecretCity" not in profile_text(p) and "example.com" not in profile_text(p)
    v = job(
        vacancy,
        title="Java Developer",
        company="SecretCompany",
        location="SecretCity",
        address="SecretStreet 19",
        description=(
            "Имя: Иван Иванов\nОбязанности:\nРазработка Java для SecretCompany.\n"
            "Адрес: SecretStreet 19\nТелефон: +7 (999) 123-45-67\n"
            "Почта: recruiter@example.com\nhttps://example.com/person\n"
            "Требования:\nJava Spring SQL\nУсловия:\nСекретные бонусы компании"
        ),
    )
    text = vacancy_text(v)
    assert "Java Spring SQL" in text
    for forbidden in (
        "Иван",
        "SecretCompany",
        "SecretCity",
        "SecretStreet",
        "999",
        "example.com",
        "бонусы",
    ):
        assert forbidden not in text
    assert text_revision(text) == text_revision(vacancy_text(v))


def test_free_unstructured_resume_or_vacancy_text_is_not_copied(vacancy):
    v = job(vacancy, description="Иван Петров, квартира 20, адрес: SecretStreet 19")
    assert "Петров" not in vacancy_text(v) and "SecretStreet" not in vacancy_text(v)


def test_index_incremental_update_removal_and_version_validation(tmp_path, vacancy):
    a, b = job(vacancy), job(vacancy, title="Another Engineer")
    calls = []

    def encode(tasks):
        calls.append(len(tasks))
        return {"values": [{"vector": vector(), "truncated": False} for _ in tasks]}

    path = tmp_path / "index.json"
    build_index([a, b], path, encode)
    _, stats = build_index([a, b], path, encode)
    assert stats["reused"] == 2 and calls == [2, 0]
    a.title = "Changed Engineer"
    _, stats = build_index([a, b], path, encode)
    assert stats["computed"] == 1
    build_index([a], path, encode)
    assert len(load_index(path)["entries"]) == 1
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["model_id"] = "wrong"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="Несовместимый"):
        load_index(path)


@pytest.mark.parametrize("corruption", ["hash", "vector", "dimension"])
def test_corrupt_index_rejected(tmp_path, vacancy, corruption):
    data = index_for([job(vacancy)])
    entry = next(iter(data["entries"].values()))
    if corruption == "vector":
        entry["vector"][0] = float("nan")
    if corruption == "dimension":
        entry["vector"] = [1.0]
    path = tmp_path / "index.json"
    save_index(data, path)
    if corruption == "hash":
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["sha256"] = "0" * 64
        path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_index(path)


def test_rrf_is_rank_based_stable_and_does_not_double_count_repeats():
    assert reciprocal_rank_fusion(["a", "b"], ["b", "c"])[0] == "b"
    assert reciprocal_rank_fusion(["a"], ["b"]) == ["a", "b"]
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(["a", "a"])


@pytest.mark.parametrize("change", ["archived", "expired", "future", "experience", "document"])
async def test_semantics_cannot_bypass_full_constraints(vacancy, now, change):
    p = candidate(experience_years=0, qualifications=[{"name": "Медкнижка", "status": "absent"}])
    good = job(
        vacancy,
        title="Java Junior",
        experience="no_experience",
        published_at=now - timedelta(days=1),
    )
    bad = job(good, title="Java", description="Java Spring\nОбязанности:\nРазработка приложений")
    if change == "archived":
        bad.archived = True
    if change == "expired":
        bad.valid_until = now - timedelta(seconds=1)
    if change == "future":
        bad.published_at = now + timedelta(days=1)
    if change == "experience":
        bad.description += "\n" + "x" * 8000 + "\nТребуется коммерческий опыт от 6 лет"
    if change == "document":
        bad.description += "\nОбязательна медицинская книжка"
    result = await hybrid_candidates(
        [good, bad],
        p,
        index_for([good, bad]),
        vector(),
        now=now,
        parameters=RetrievalParameters(k=1),
    )
    assert [v.id for v in result.semantic] == [good.id]
    assert [v.id for v in result.fused] == [good.id]


async def test_history_feedback_exclusions_and_duplicate_groups_precede_topk(vacancy, now):
    a = job(vacancy, published_at=now - timedelta(days=1))
    duplicate = job(a, published_at=a.published_at - timedelta(days=1))
    b = job(vacancy, title="Another Java Engineer", published_at=now - timedelta(days=1))
    cards = [a, duplicate, b]
    idx = index_for(cards)
    result = await hybrid_candidates(cards, candidate(), idx, vector(), now=now)
    assert len(result.semantic) == 2 and len(result.fused) == 2
    result = await hybrid_candidates(
        cards,
        candidate(),
        idx,
        vector(),
        now=now,
        excluded_keys={vacancy_duplicate_key(a)},
        parameters=RetrievalParameters(k=1),
    )
    assert [v.id for v in result.fused] == [b.id]
    result = await hybrid_candidates(
        cards, candidate(), idx, vector(), now=now, excluded_ids={a.id, duplicate.id, b.id}
    )
    assert not result.fused


@pytest.mark.parametrize("reason", ["missing", "version", "stale", "query"])
async def test_fallback_keeps_exact_lexical_selection(vacancy, now, reason):
    a = job(vacancy, published_at=now - timedelta(days=1))
    idx = index_for([a])
    query = vector()
    if reason == "missing":
        idx = None
    if reason == "version":
        idx["model_id"] = "other"
    if reason == "stale":
        next(iter(idx["entries"].values()))["input_revision"] = "other"
    if reason == "query":
        query = [float("nan")] * 384
    result = await hybrid_candidates([a], candidate(), idx, query, now=now)
    assert result.fused == result.lexical and result.fallback_reason is not None


@pytest.mark.parametrize("wanted,other", [("Java", "JavaScript"), ("C", "C++"), ("C++", "C#")])
async def test_semantic_channel_preserves_explicit_technology_identity(vacancy, now, wanted, other):
    p = candidate(skills={wanted: {}}, preferences={"positions": [wanted + " Developer"]})
    a = job(vacancy, title=wanted + " Developer", published_at=now - timedelta(days=1))
    b = job(vacancy, title=other + " Developer", published_at=now - timedelta(days=1))
    result = await hybrid_candidates([a, b], p, index_for([a, b]), vector(), now=now)
    assert [v.id for v in result.semantic] == [a.id]


def test_no_automatic_model_download_or_neural_import_at_module_import():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import jobs.hybrid_candidates,jobs.semantic_model,jobs.semantic_index; "
            "assert not any(x in sys.modules for x in ['onnxruntime','torch','tokenizers'])",
        ],
        check=True,
        capture_output=True,
    )


def test_downloader_checks_hash_and_does_not_redownload_valid_files(tmp_path, monkeypatch):
    import hashlib
    import sys

    import requests

    from jobs import semantic_model

    payload = b"test-model"
    monkeypatch.setitem(sys.modules, "truststore", SimpleNamespace(inject_into_ssl=lambda: None))
    monkeypatch.setitem(
        MODEL_SPEC,
        "files",
        {
            "model.onnx": {
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "source": "model.onnx",
            }
        },
    )
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, size):
            yield payload

    monkeypatch.setattr(requests, "get", lambda *a, **kw: calls.append(a[0]) or Response())
    folder = tmp_path / "model"
    download_model(folder)
    download_model(folder)
    assert len(calls) == 1
    (folder / "model.onnx").write_bytes(b"wrong")
    with pytest.raises(ValueError):
        semantic_model.validate_model(folder)


def test_vectors_require_finite_normalized_exact_dimension():
    assert valid_vector(vector())
    assert not valid_vector([0.0] * 384) and not valid_vector([1.0] * 383)


@pytest.mark.parametrize("exceeds_limit", [False, True])
def test_watchdog_accounts_for_windows_launcher_children_and_kills_tree(
    tmp_path, monkeypatch, exceeds_limit
):
    import sys

    from jobs import semantic_process as module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "_release_retained_ollama", lambda: None)
    events = []

    class Member:
        def __init__(self, name, resident):
            self.name, self.resident = name, resident

        def nice(self, value):
            events.append(("nice", self.name))

        def cpu_affinity(self, value=None):
            if value is not None:
                events.append(("affinity", self.name, value))
            return [0, 1]

        def memory_info(self):
            return SimpleNamespace(rss=self.resident)

        def children(self, recursive=True):
            return [descendant] if self is root else []

        def kill(self):
            events.append(("kill", self.name))

    descendant = Member("actual-python", module.MAX_RSS if exceeds_limit else 500_000_000)
    root = Member("venv-launcher", 4_000_000)
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(
            Process=lambda pid: root,
            virtual_memory=lambda: SimpleNamespace(available=module.MIN_AVAILABLE * 2),
            BELOW_NORMAL_PRIORITY_CLASS=1,
            NoSuchProcess=ProcessLookupError,
        ),
    )

    class Process:
        pid = 42

        def __init__(self, args, **kwargs):
            self.count = 0
            self.args = args
            self.stdin = SimpleNamespace(write=lambda value: None, close=self.release)

        def release(self):
            Path(self.args[-2]).write_text(json.dumps({"values": []}), encoding="utf-8")

        def poll(self):
            self.count += 1
            return None if exceeds_limit or self.count == 1 else 0

        def wait(self):
            return 0

        def kill(self):
            events.append(("kill", "launcher"))

    monkeypatch.setattr(module.subprocess, "Popen", Process)
    monkeypatch.setattr(module.time, "sleep", lambda value: None)
    if exceeds_limit:
        with pytest.raises(ValueError, match="лимиту"):
            module.encode_bounded([{"text": "test", "kind": "query"}], "model")
        assert ("kill", "actual-python") in events and ("kill", "launcher") in events
    else:
        result = module.encode_bounded([{"text": "test", "kind": "query"}], "model")
        assert result["peak_rss_bytes"] == 504_000_000
    assert ("affinity", "actual-python", [0]) in events


def test_low_available_ram_prevents_model_process_launch(monkeypatch):
    import sys

    from jobs import semantic_process as module

    monkeypatch.setattr(module, "_release_retained_ollama", lambda: None)

    monkeypatch.setitem(
        sys.modules, "psutil", SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=1))
    )
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **kw: pytest.fail("must not launch"))
    with pytest.raises(ValueError, match="4 ГиБ"):
        module.encode_bounded([{"text": "test", "kind": "query"}], "model")


def test_report_checks_constraints_before_ranker_can_hide_invalid_rows(vacancy, now):
    from jobs.hybrid_candidates import HybridResult
    from jobs.semantic_experiment import _case_metrics

    v = job(vacancy, archived=True, published_at=now - timedelta(days=1))
    r = HybridResult(lexical=[v], semantic=[v], fused=[v], novelty=[v])
    metrics = _case_metrics(r, candidate(), now, {f"{v.source}:{v.source_id}": 0})
    assert all(m["eligibility_violations"] == 1 for m in metrics.values())


def test_busy_existing_model_slot_prevents_parallel_inference(monkeypatch):
    from jobs import semantic_process
    from nlp import ollama_slot

    monkeypatch.setattr(
        semantic_process, "_encode_bounded", lambda *args: pytest.fail("must not launch")
    )
    with ollama_slot.LOCK_PATH.open("a+b") as stream:
        if stream.seek(0, 2) == 0:
            stream.write(b"0")
            stream.flush()
        assert ollama_slot._lock(stream)
        try:
            with pytest.raises(ValueError, match="Слот модели занят"):
                semantic_process.encode_bounded([{"text": "test", "kind": "query"}], "model")
        finally:
            ollama_slot._unlock(stream)
