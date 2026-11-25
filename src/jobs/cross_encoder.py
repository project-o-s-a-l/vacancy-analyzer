"""Pinned pair inference. Local CPU, exclusive inference slot, disposable watchdog worker."""

import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from jobs.semantic_encoder import download_encoder, validate_encoder
from jobs.semantic_model import open_cpu_session
from jobs.semantic_process import MAX_RSS, MIN_AVAILABLE
from scoring.serving_features import digest

MODEL = {
    "repo": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "revision": "1427fd652930e4ba29e8149678df786c240d8825",
    "license": "Apache-2.0",
    "max_tokens": 512,
    "files": {
        "model.onnx": {
            "source": "onnx/model_quint8_avx2.onnx",
            "size": 118620016,
            "sha256": "6c2513767fb63d008a4377bef7a7a3555433d9436342bb53e35a3a72ffc52d4b",
        },
        "tokenizer.json": {
            "source": "tokenizer.json",
            "size": 17082660,
            "sha256": "62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626",
        },
    },
}
PAIR_SPEC = {
    "version": "professional-pair-f12-v1",
    "input": "Professional sections; candidate first; vacancy second; no prompt",
    "pair_tokens": "<s> candidate </s></s> vacancy </s>",
    "max_tokens": 512,
    "side_tokens": 254,
    "max_windows": 16,
    "batch": 1,
    "batch_reason": "dynamic int8 activation scale must not depend on other pairs",
    "pooling": "mean over vacancy windows of max over candidate windows; raw logits",
    "overflow": "reject whole request; no truncation",
}
MODEL_ID = digest({"model": MODEL, "pair": PAIR_SPEC})


def _register():
    # Reuse the shared pinned-file downloader/validator, with a distinct registered identity.
    from jobs.semantic_encoder import ENCODERS

    ENCODERS.setdefault("f12-cross", MODEL)


def download_model(directory):
    _register()
    download_encoder(directory, "f12-cross")


def validate_model(directory):
    _register()
    validate_encoder(directory, "f12-cross")


def document_windows(tokenizer, sections, *, capacity=254):
    """Pack complete labelled facts; long fact continuations repeat their labels.

    OR alternatives are atomic: splitting them could invent independent requirements.
    Every original token survives or the entire pair is rejected explicitly.
    """
    if not sections or any(not isinstance(s, str) for s in sections):
        raise ValueError("empty_or_invalid_sections")
    if sum(map(len, sections)) > 200000:
        raise ValueError("input_budget")
    windows, pending = [], []
    for section in sections:
        ids = tokenizer.encode(section, add_special_tokens=False).ids
        if not ids:
            continue
        pieces = [ids]
        if len(ids) > capacity:
            if "alternatives (any one)" in section:
                raise ValueError("alternative_group_overflow")
            boundary = section.rfind(":", 0, 120)
            label = tokenizer.encode(
                section[: boundary + 1] if boundary >= 0 else "Section continuation:",
                add_special_tokens=False,
            ).ids
            if len(label) >= capacity:
                raise ValueError("label_overflow")
            pieces = [ids[:capacity]]
            remaining = ids[capacity:]
            while remaining:
                pieces.append(label + remaining[: capacity - len(label)])
                remaining = remaining[capacity - len(label) :]
        for piece in pieces:
            if pending and len(pending) + len(piece) > capacity:
                windows.append(pending)
                pending = []
            pending.extend(piece)
            if len(windows) >= PAIR_SPEC["max_windows"]:
                raise ValueError("pair_window_budget")
    if pending:
        windows.append(pending)
    if not windows:
        raise ValueError("empty_professional_document")
    return windows


def pair_windows(tokenizer, task):
    if sum(len(s) for side in ("candidate", "vacancy") for s in task[side]) > 200000:
        raise ValueError("pair_input_budget")
    left = document_windows(tokenizer, task["candidate"])
    right = document_windows(tokenizer, task["vacancy"])
    if len(left) * len(right) > PAIR_SPEC["max_windows"]:
        raise ValueError("pair_window_budget")
    bos, eos = tokenizer.token_to_id("<s>"), tokenizer.token_to_id("</s>")
    if bos != 0 or eos != 2:
        raise ValueError("incompatible_special_tokens")
    return [([bos, *a, eos, eos, *b, eos], j) for j, b in enumerate(right) for a in left]


def validate_values(values, count):
    if not isinstance(values, list) or len(values) != count:
        raise ValueError("invalid_pair_output_count")
    for row in values:
        if (
            not isinstance(row, dict)
            or type(row.get("score")) not in {int, float}
            or not math.isfinite(row["score"])
            or type(row.get("windows")) is not int
            or not 1 <= row["windows"] <= PAIR_SPEC["max_windows"]
            or row.get("truncated") is not False
        ):
            raise ValueError("invalid_pair_output")


class LocalCrossEncoder:
    """Instantiate only in the watchdog child. No torch/pickle/remote code."""

    def __init__(self, directory):
        import numpy as np
        from tokenizers import Tokenizer

        validate_model(directory)
        self.np = np
        self.tokenizer = Tokenizer.from_file(str(Path(directory) / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        self.session = open_cpu_session(directory)
        if {i.name for i in self.session.get_inputs()} != {"input_ids", "attention_mask"}:
            raise ValueError("incompatible_onnx_inputs")

    def score(self, tasks):
        expanded, groups, counts = [], [], []
        for index, task in enumerate(tasks):
            windows = pair_windows(self.tokenizer, task)
            counts.append(len(windows))
            for ids, section in windows:
                expanded.append(ids)
                groups.append((index, section))
        margins = [dict() for _ in tasks]
        for start in range(0, len(expanded), PAIR_SPEC["batch"]):
            rows = expanded[start : start + PAIR_SPEC["batch"]]
            length = max(map(len, rows))
            ids = self.np.full((len(rows), length), 1, dtype=self.np.int64)
            mask = self.np.zeros_like(ids)
            for i, row in enumerate(rows):
                ids[i, : len(row)] = row
                mask[i, : len(row)] = 1
            output = self.session.run(None, {"input_ids": ids, "attention_mask": mask})[0]
            if output.shape != (len(rows), 1) or not self.np.isfinite(output).all():
                raise ValueError("invalid_logits")
            for offset, value in enumerate(output[:, 0]):
                index, section = groups[start + offset]
                margins[index].setdefault(section, []).append(float(value))
        values = [
            {
                "score": sum(max(v) for v in sections.values()) / len(sections),
                "windows": count,
                "truncated": False,
            }
            for sections, count in zip(margins, counts, strict=True)
        ]
        validate_values(values, len(tasks))
        return values


def score_bounded(tasks, directory, *, deadline=2.0, repeats=1):
    """Inference slot, one process, no live DB transaction; timeout accepts no partial values."""
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 32:
        raise ValueError("pair_request_budget")
    if not 0 < deadline <= 600 or not 1 <= repeats <= 6:
        raise ValueError("invalid_runtime_budget")
    from nlp import ollama_slot

    started = time.monotonic()
    with ollama_slot.LOCK_PATH.open("a+b") as stream:
        if stream.seek(0, 2) == 0:
            stream.write(b"0")
            stream.flush()
        if not ollama_slot._lock(stream):
            raise ValueError("inference_slot_busy")
        try:
            # Inference drain runs INSIDE the watchdog deadline, before loading any weights.
            return _run(tasks, directory, deadline, repeats, started)
        finally:
            ollama_slot._unlock(stream)


def _run(tasks, directory, deadline, repeats, started):
    import psutil

    if psutil.virtual_memory().available < MIN_AVAILABLE:
        raise ValueError("insufficient_memory")
    if time.monotonic() - started >= deadline:
        raise TimeoutError("pair_deadline")
    workspace = Path(".local-tmp")
    workspace.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="f12-pair-", dir=workspace) as temporary:
        source, target = Path(temporary) / "input.json", Path(temporary) / "output.json"
        wire = json.dumps({"tasks": tasks, "repeats": repeats}, ensure_ascii=False)
        if len(wire.encode()) > 8 * 1024 * 1024:
            raise ValueError("pair_wire_budget")
        source.write_text(wire, encoding="utf-8")
        env = os.environ.copy() | {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "PYTHONIOENCODING": "utf-8",
        }
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "jobs.cross_encoder_worker",
                str(source),
                str(target),
                str(directory),
            ],
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        child, peak, cpu_peak = psutil.Process(process.pid), 0, 0.0
        try:
            affinity = child.cpu_affinity()[:1] if hasattr(child, "cpu_affinity") else None
            restricted = set()
            process.stdin.write(b"1")
            process.stdin.close()
            while process.poll() is None:
                resident = cpu = 0
                try:
                    members = [child, *child.children(recursive=True)]
                except psutil.NoSuchProcess:
                    members = []
                for member in members:
                    try:
                        if member.pid not in restricted:
                            if affinity:
                                member.cpu_affinity(affinity)
                            member.nice(
                                psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10
                            )
                            restricted.add(member.pid)
                        resident += member.memory_info().rss
                        times = member.cpu_times()
                        cpu += times.user + times.system
                    except psutil.NoSuchProcess:
                        pass
                    except psutil.AccessDenied:
                        # Windows can deny access to an exiting process before psutil
                        # observes its disappearance. Never waive limits for a live one.
                        if process.poll() is None and member.is_running():
                            raise ValueError("worker_resource_access_denied") from None
                peak, cpu_peak = max(peak, resident), max(cpu_peak, cpu)
                if time.monotonic() - started > deadline:
                    raise TimeoutError("pair_deadline")
                if peak > MAX_RSS or psutil.virtual_memory().available < MIN_AVAILABLE:
                    raise ValueError("pair_memory_budget")
                time.sleep(0.025)
            if time.monotonic() - started > deadline:
                raise TimeoutError("pair_deadline")
            if process.wait() or not target.is_file():
                raise ValueError("pair_worker_failed")
            if target.stat().st_size > 1024 * 1024:
                raise ValueError("pair_output_budget")
            result = json.loads(target.read_text(encoding="utf-8"))
            validate_values(result.get("values"), len(tasks))
            return result | {
                "model_id": MODEL_ID,
                "wall_seconds": time.monotonic() - started,
                "peak_worker_tree_rss": peak,
                "sampled_worker_cpu_seconds": cpu_peak,
                "gpu_allocated_bytes": 0,
            }
        finally:
            if not process.stdin.closed:
                process.stdin.close()
            if process.poll() is None:
                try:
                    members = child.children(recursive=True)
                except psutil.NoSuchProcess:
                    members = []
                for member in reversed(members):
                    try:
                        member.kill()
                    except psutil.NoSuchProcess:
                        pass
                process.kill()
                process.wait()
