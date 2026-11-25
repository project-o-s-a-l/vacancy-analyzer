"""encoder contract checks, without downloads or neural inference."""

from types import SimpleNamespace

import numpy as np
import pytest

from jobs import semantic_encoder as module


class Tokenizer:
    def __init__(self):
        self.words = {}

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return SimpleNamespace(
            ids=[self.words.setdefault(word, len(self.words) + 1) for word in text.split()]
        )

    def token_to_id(self, token):
        return {"<s>": 0, "</s>": 0}.get(token)


def test_chunks_keep_sentences_sections_prefix_and_long_tail():
    tokenizer = Tokenizer()
    chunks = module.section_chunks(
        tokenizer,
        ["one two. three four.", "five six seven eight nine ten tail"],
        prefix="query: ",
        max_tokens=7,
    )
    assert [section for section, _, _ in chunks] == [0, 1, 1]
    assert [len(body) for _, body, _ in chunks] == [4, 4, 3]
    assert all(len(body) + len(prefix) + 2 <= 7 for _, body, prefix in chunks)
    assert all(prefix == [tokenizer.words["query:"]] for _, _, prefix in chunks)
    assert chunks[-1][1][-1] == tokenizer.words["tail"]
    assert sum(len(body) for _, body, _ in chunks) == 11


def test_long_document_fails_explicitly_instead_of_truncating():
    with pytest.raises(ValueError, match="chunk budget"):
        module.section_chunks(Tokenizer(), ["word " * 13], prefix="", max_tokens=4, max_chunks=6)
    with pytest.raises(ValueError, match="empty"):
        module.section_chunks(Tokenizer(), ["  "], prefix="", max_tokens=128)


def fake_encoder(model_name):
    encoder = module.LocalSectionEncoder.__new__(module.LocalSectionEncoder)
    encoder.np = np
    encoder.spec = module.ENCODERS[model_name]
    encoder.tokenizer = Tokenizer()
    encoder.inputs = {"input_ids", "attention_mask"}
    feeds = []

    def run(_, feed):
        feeds.append(feed)
        ids = feed["input_ids"]
        output = np.zeros((*ids.shape, 384), dtype=float)
        output[..., 0] = ids == 1
        output[..., 1] = ids >= 2
        return [output]

    encoder.session = SimpleNamespace(run=run)
    return encoder, feeds


def test_pooling_is_unit_finite_token_weighted_and_section_alternative_is_distinct():
    encoder, feeds = fake_encoder("minilm")
    result = encoder.encode(["one", "two two two"], "query")
    assert result["tokens"] == 4
    assert result["chunk_body_tokens"] == [1, 3]
    assert result["chunks"] == result["section_count"] == 2
    assert result["truncated"] is False
    assert np.isclose(np.linalg.norm(result["vector"]), 1)
    assert np.isclose(np.linalg.norm(result["section_vector"]), 1)
    assert np.allclose(result["vector"][:2], np.array([1, 3]) / np.sqrt(10))
    assert np.allclose(result["section_vector"][:2], np.array([1, 1]) / np.sqrt(2))
    assert [feed["input_ids"].shape[1] for feed in feeds] == [3, 5]
    assert all(np.all(feed["attention_mask"] == 1) for feed in feeds)


def test_e5_prefix_present_on_every_chunk_and_minilm_has_no_prefix():
    encoder, feeds = fake_encoder("e5-small")
    result = encoder.encode(["word " * 520], "passage")
    assert result["chunks"] == 2
    assert result["tokens"] == 520
    prefix = encoder.tokenizer.words["passage:"]
    assert all(feed["input_ids"][0, 1] == prefix for feed in feeds)
    assert max(feed["input_ids"].shape[1] for feed in feeds) == 512
    assert module.ENCODERS["minilm"]["max_tokens"] == 128
    assert module.ENCODERS["minilm"]["prefixes"] == {"query": "", "passage": ""}


def test_invalid_or_nonfinite_model_output_fails_closed():
    encoder, _ = fake_encoder("minilm")
    encoder.session.run = lambda *_: [np.full((1, 3, 384), np.nan)]
    with pytest.raises(ValueError, match="Invalid encoder vector"):
        encoder.encode(["word"], "query")
    encoder.session.run = lambda *_: [np.zeros((1, 384))]
    with pytest.raises(ValueError, match="output shape"):
        encoder.encode(["word"], "query")
    with pytest.raises(ValueError, match="input kind"):
        encoder.encode(["word"], "candidate")


def test_model_content_and_encoding_policy_are_part_of_identity(monkeypatch):
    original = module.encoder_id("minilm")
    monkeypatch.setitem(module.ENCODING_SPEC, "chunk_limit", 63)
    assert module.encoder_id("minilm") != original
    assert module.encoder_id("e5-small") != module.encoder_id("minilm")


def test_missing_encoder_is_rejected_without_download(tmp_path):
    with pytest.raises(ValueError, match="pinned encoder"):
        module.validate_encoder(tmp_path, "minilm")
    assert list(tmp_path.iterdir()) == []


def test_empty_batch_avoids_model_and_shared_slot(monkeypatch):
    monkeypatch.setattr(module, "_release_retained_ollama", lambda: pytest.fail("release called"))
    result = module.encode_sections_bounded([], "missing", "e5-small")
    assert result["values"] == []
    assert result["peak_rss_bytes"] == result["wall_seconds"] == 0


def test_concurrent_inference_fails_without_releasing_other_model(monkeypatch, tmp_path):
    from nlp import ollama_slot

    monkeypatch.setattr(ollama_slot, "LOCK_PATH", tmp_path / "slot.lock")
    monkeypatch.setattr(ollama_slot, "_lock", lambda _: False)
    monkeypatch.setattr(module, "_release_retained_ollama", lambda: pytest.fail("release called"))
    with pytest.raises(ValueError, match="occupied"):
        module.encode_sections_bounded([{"sections": ["query"], "kind": "query"}], "missing")


def test_absent_local_listener_is_proven_but_remote_timeout_is_not_ignored(monkeypatch):
    import psutil

    from jobs.settings import Settings

    monkeypatch.setattr(psutil, "net_connections", lambda **_: [])
    monkeypatch.setattr(Settings, "__init__", lambda self: None)
    monkeypatch.setattr(module, "_release_retained_ollama", lambda: pytest.fail("unexpected drain"))
    monkeypatch.setattr(
        "jobs.settings.Settings", lambda: SimpleNamespace(ollama_base_url="http://127.0.0.1:11434")
    )
    module._release_local_model()
    called = []
    monkeypatch.setattr(module, "_release_retained_ollama", lambda: called.append(True))
    monkeypatch.setattr(
        "jobs.settings.Settings",
        lambda: SimpleNamespace(ollama_base_url="http://remote.test:11434"),
    )
    module._release_local_model()
    assert called == [True]


def test_listener_and_uncertain_marker_preserve_f08_drain(monkeypatch):
    import psutil

    monkeypatch.setattr(
        "jobs.settings.Settings", lambda: SimpleNamespace(ollama_base_url="http://127.0.0.1:11434")
    )
    monkeypatch.setattr(
        psutil,
        "net_connections",
        lambda **_: [SimpleNamespace(status="LISTEN", laddr=SimpleNamespace(port=11434))],
    )
    called = []
    monkeypatch.setattr(module, "_release_retained_ollama", lambda: called.append(True))
    module._release_local_model()
    assert called == [True]


def test_worker_exit_during_rss_sampling_accepts_completed_output(monkeypatch):
    import io
    import json
    from pathlib import Path

    import psutil

    class FinishedChild:
        def children(self, **_):
            raise psutil.NoSuchProcess(123)

    class Process:
        pid = 123
        stdin = io.BytesIO()
        polls = 0

        def __init__(self, command, **_):
            Path(command[4]).write_text(json.dumps({"values": [], "elapsed_seconds": 0}))

        def poll(self):
            self.polls += 1
            return None if self.polls == 1 else 0

        def wait(self):
            return 0

    monkeypatch.setattr(module.subprocess, "Popen", Process)
    monkeypatch.setattr(psutil, "Process", lambda _: FinishedChild())
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=10**12))
    result = module._run_worker([{"kind": "query", "sections": ["text"]}], "unused", "minilm")
    assert result["values"] == [] and result["peak_rss_bytes"] == 0


def test_uncertain_marker_blocks_absent_listener(monkeypatch, tmp_path):
    from nlp import ollama_slot

    monkeypatch.setattr(ollama_slot, "LOCK_PATH", tmp_path / "slot.lock")
    ollama_slot.LOCK_PATH.with_suffix(".uncertain.json").write_text("{}")
    with pytest.raises(ValueError, match="uncertain"):
        module._release_local_model()
