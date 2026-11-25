"""Pinned local encoders for F11; explicit downloads and bounded offline execution."""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

from jobs.semantic_model import MODEL_SPEC as E5_FILES
from jobs.semantic_model import file_hash, open_cpu_session
from jobs.semantic_process import MAX_RSS, MAX_SECONDS, MIN_AVAILABLE, _release_retained_ollama
from scoring.serving_features import digest

ENCODING_SPEC = {
    "version": "f11-section-encoder-v1",
    "chunking": "section_then_sentence_token_windows_no_overlap",
    "chunk_limit": 64,
    "pooling": "masked_mean_l2_then_unique_body_token_weighted_l2",
    "alternative_pooling": "within_section_token_weighted_l2_then_equal_sections_l2",
    "device": "cpu_single_thread",
}
ENCODERS = {
    "e5-small": {
        "repo": E5_FILES["repo"],
        "revision": E5_FILES["revision"],
        "dimension": 384,
        "max_tokens": 512,
        "prefixes": {"query": "query: ", "passage": "passage: "},
        "files": E5_FILES["files"],
        "instructions": "https://huggingface.co/intfloat/multilingual-e5-small",
    },
    "minilm": {
        "repo": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "revision": "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
        "dimension": 384,
        "max_tokens": 128,
        "prefixes": {"query": "", "passage": ""},
        "files": {
            "model.onnx": {
                "source": "onnx/model_qint8_avx512_vnni.onnx",
                "size": 118412398,
                "sha256": "783fea82d71a58179b830a4dbd2d58447e640609e98eedf9ffa12622d375a672",
            },
            "tokenizer.json": {
                "source": "tokenizer.json",
                "size": 9081518,
                "sha256": "2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8",
            },
        },
        "instructions": (
            "https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ),
    },
}


def encoder_id(model_name):
    return digest({"encoder": ENCODERS[model_name], "encoding": ENCODING_SPEC})


def validate_encoder(directory, model_name):
    for name, meta in ENCODERS[model_name]["files"].items():
        path = Path(directory) / name
        if (
            not path.is_file()
            or path.stat().st_size != meta["size"]
            or file_hash(path) != meta["sha256"]
        ):
            raise ValueError("Missing or incompatible pinned encoder file")


def download_encoder(directory, model_name):
    """Explicit CLI only. Download two hash-pinned files; never run remote code."""
    import requests

    if os.name == "nt":
        import truststore

        truststore.inject_into_ssl()
    spec = ENCODERS[model_name]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, meta in spec["files"].items():
        path = directory / name
        if path.exists():
            if path.stat().st_size != meta["size"] or file_hash(path) != meta["sha256"]:
                raise ValueError("Existing encoder file is incompatible")
            continue
        temporary = directory / (name + ".partial")
        # Exclusive temporary ownership prevents another downloader's file being removed.
        with temporary.open("xb") as stream:
            try:
                url = (
                    f"https://huggingface.co/{spec['repo']}/resolve/"
                    f"{spec['revision']}/{meta['source']}"
                )
                with requests.get(url, stream=True, timeout=(15, 60)) as response:
                    response.raise_for_status()
                    size = 0
                    for block in response.iter_content(1024 * 1024):
                        size += len(block)
                        if size > meta["size"]:
                            raise ValueError("Pinned download size exceeded")
                        stream.write(block)
            except BaseException:
                stream.close()
                temporary.unlink(missing_ok=True)
                raise
        try:
            if size != meta["size"] or file_hash(temporary) != meta["sha256"]:
                raise ValueError("Pinned encoder checksum mismatch")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    validate_encoder(directory, model_name)


def section_chunks(tokenizer, sections, *, prefix, max_tokens, max_chunks=64):
    """Keep section/sentence boundaries where possible; every body token is retained once."""
    if (
        not isinstance(sections, (list, tuple))
        or not sections
        or any(not isinstance(section, str) for section in sections)
        or sum(map(len, sections)) > 200000
    ):
        raise ValueError("Invalid professional sections")
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False).ids if prefix else []
    capacity = max_tokens - len(prefix_ids) - 2
    if capacity < 1:
        raise ValueError("Encoder prefix leaves no document capacity")
    chunks = []

    def append(section_id, body):
        if len(chunks) >= max_chunks:
            raise ValueError("Professional document exceeds the explicit chunk budget")
        chunks.append((section_id, list(body), prefix_ids))

    for section_id, section in enumerate(sections):
        pending = []
        for sentence in re.split(r"(?<=[.!?。！？;])\s+|\n+", section.strip()):
            if not sentence.strip():
                continue
            body = tokenizer.encode(sentence, add_special_tokens=False).ids
            if pending and len(pending) + len(body) > capacity:
                append(section_id, pending)
                pending = []
            while len(body) > capacity:
                append(section_id, body[:capacity])
                body = body[capacity:]
            pending.extend(body)
        if pending:
            append(section_id, pending)
    if not chunks:
        raise ValueError("Professional document is empty")
    return chunks


class LocalSectionEncoder:
    """Only constructed inside the bounded worker; tests may inject deterministic components."""

    def __init__(self, directory, model_name):
        import numpy as np
        from tokenizers import Tokenizer

        validate_encoder(directory, model_name)
        self.spec = ENCODERS[model_name]
        self.np = np
        self.tokenizer = Tokenizer.from_file(str(Path(directory) / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.session = open_cpu_session(directory)
        self.inputs = {value.name for value in self.session.get_inputs()}

    def _normal(self, vector):
        norm = self.np.linalg.norm(vector)
        if not self.np.isfinite(vector).all() or not self.np.isfinite(norm) or norm <= 0:
            raise ValueError("Invalid encoder vector")
        return vector / norm

    def encode(self, sections, kind):
        if kind not in self.spec["prefixes"]:
            raise ValueError("Invalid encoder input kind")
        chunks = section_chunks(
            self.tokenizer,
            sections,
            prefix=self.spec["prefixes"][kind],
            max_tokens=self.spec["max_tokens"],
            max_chunks=ENCODING_SPEC["chunk_limit"],
        )
        vectors, weights, by_section = [], [], {}
        for section_id, body, prefix in chunks:
            ids = [
                self.tokenizer.token_to_id("<s>"),
                *prefix,
                *body,
                self.tokenizer.token_to_id("</s>"),
            ]
            if any(value is None for value in ids):
                raise ValueError("Missing encoder boundary tokens")
            array = self.np.array([ids], dtype=self.np.int64)
            feed = {
                "input_ids": array,
                "attention_mask": self.np.ones_like(array),
                "token_type_ids": self.np.zeros_like(array),
            }
            output = self.session.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0]
            if (
                output.ndim != 3
                or output.shape[:2] != array.shape
                or output.shape[-1] != self.spec["dimension"]
            ):
                raise ValueError("Invalid encoder output shape")
            # One sequence, no padding: its attention mask is all ones.
            vector = self._normal(output[0].mean(axis=0))
            vectors.append(vector)
            weights.append(len(body))
            by_section.setdefault(section_id, []).append((vector, len(body)))
        section_vectors = [
            self._normal(
                self.np.average([v for v, _ in rows], axis=0, weights=[n for _, n in rows])
            )
            for rows in by_section.values()
        ]
        return {
            "vector": self._normal(self.np.average(vectors, axis=0, weights=weights)).tolist(),
            "section_vector": self._normal(self.np.mean(section_vectors, axis=0)).tolist(),
            "chunks": len(chunks),
            "tokens": sum(weights),
            "chunk_body_tokens": weights,
            "section_count": len(by_section),
            "truncated": False,
        }


def encode_sections_bounded(tasks, model_directory, model_name="e5-small"):
    """One shared inference slot, a disposable process, explicit memory/time limits."""
    if model_name not in ENCODERS:
        raise ValueError("Unknown experimental encoder")
    if not tasks:
        return {
            "values": [],
            "peak_rss_bytes": 0,
            "elapsed_seconds": 0,
            "load_seconds": 0,
            "wall_seconds": 0,
            "model_id": encoder_id(model_name),
        }
    return _with_slot(lambda: _run_worker(tasks, model_directory, model_name))


def encode_legacy_bounded(tasks, model_directory):
    """Historical encoder math/text, same host guard for a fair control."""
    from jobs.semantic_process import _encode_bounded

    if not tasks:
        return _encode_bounded(tasks, model_directory)
    return _with_slot(lambda: _encode_bounded(tasks, model_directory))


def _release_local_model():
    """A closed local port can time out on Windows; prove absence instead of guessing.

    Unknown remote endpoints, listener enumeration failure, existing listeners or
    uncertain in-flight work still use the Inference drain and fail closed on error.
    """
    import psutil

    from jobs.settings import Settings
    from nlp import ollama_slot

    if ollama_slot.LOCK_PATH.with_suffix(".uncertain.json").exists():
        raise ValueError("Previous inference outcome is uncertain")
    endpoint = urlsplit(Settings().ollama_base_url)
    if endpoint.hostname in {"localhost", "127.0.0.1", "::1"}:
        port = endpoint.port or (443 if endpoint.scheme == "https" else 80)
        listeners = [
            c
            for c in psutil.net_connections(kind="tcp")
            if c.status == psutil.CONN_LISTEN and c.laddr.port == port
        ]
        if not listeners:
            return
    _release_retained_ollama()


def _with_slot(work):
    from nlp import ollama_slot

    with ollama_slot.LOCK_PATH.open("a+b") as stream:
        if stream.seek(0, 2) == 0:
            stream.write(b"0")
            stream.flush()
        if not ollama_slot._lock(stream):
            raise ValueError("Inference slot is occupied")
        try:
            _release_local_model()
            return work()
        finally:
            ollama_slot._unlock(stream)


def _run_worker(tasks, model_directory, model_name):
    import psutil

    if not isinstance(tasks, list) or len(tasks) > 5000:
        raise ValueError("Invalid encoder batch")
    if psutil.virtual_memory().available < MIN_AVAILABLE:
        raise ValueError("Insufficient available RAM for encoder")
    workspace = Path(".local-tmp")
    workspace.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="f11-encoder-", dir=workspace) as temporary:
        input_path, output_path = Path(temporary) / "input.json", Path(temporary) / "output.json"
        wire = json.dumps(tasks, ensure_ascii=False)
        if len(wire.encode("utf-8")) > 16 * 1024 * 1024:
            raise ValueError("Encoder batch exceeds serialized input budget")
        input_path.write_text(wire, encoding="utf-8")
        environment = os.environ.copy() | {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONIOENCODING": "utf-8",
        }
        started = time.perf_counter()
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "jobs.semantic_encoder_worker",
                str(input_path),
                str(output_path),
                str(model_directory),
                model_name,
            ],
            env=environment,
            stdin=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        child, peak = psutil.Process(process.pid), 0
        try:
            affinity = child.cpu_affinity()[:1] if hasattr(child, "cpu_affinity") else None

            def sample_tree():
                resident = 0
                try:
                    members = [child, *child.children(recursive=True)]
                except psutil.NoSuchProcess:
                    return 0
                for member in members:
                    try:
                        member.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
                        if affinity:
                            member.cpu_affinity(affinity)
                        resident += member.memory_info().rss
                    except psutil.NoSuchProcess:
                        continue
                return resident

            peak = sample_tree()
            process.stdin.write(b"1")
            process.stdin.close()
            while process.poll() is None:
                peak = max(peak, sample_tree())
                if (
                    peak > MAX_RSS
                    or psutil.virtual_memory().available < MIN_AVAILABLE
                    or time.perf_counter() - started > MAX_SECONDS
                ):
                    raise ValueError("Encoder exceeded memory or time budget")
                time.sleep(0.05)
            if process.wait() != 0 or not output_path.is_file():
                raise ValueError("Bounded encoder failed; no vectors accepted")
            result = json.loads(output_path.read_text(encoding="utf-8"))
            return result | {
                "peak_rss_bytes": peak,
                "wall_seconds": time.perf_counter() - started,
                "model_id": encoder_id(model_name),
            }
        finally:
            if process.poll() is None:
                try:
                    descendants = child.children(recursive=True)
                except psutil.NoSuchProcess:
                    descendants = []
                for member in reversed(descendants):
                    try:
                        member.kill()
                    except psutil.NoSuchProcess:
                        pass
                process.kill()
                process.wait()
