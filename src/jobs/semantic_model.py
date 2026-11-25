"""Explicit pinned download and local-only CPU ONNX inference, without remote code."""

import hashlib
import json
import os
from pathlib import Path

from scoring.serving_features import digest

MODEL_SPEC = {
    "repo": "intfloat/multilingual-e5-small",
    "revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
    "dimension": 384,
    "format": "onnx-int8-masked-mean-l2-v1",
    "max_tokens": 256,
    "max_chunks": 4,
    "chunk_stride": 32,
    "prefixes": {"query": "query: ", "passage": "passage: "},
    "files": {
        "model.onnx": {
            "source": "onnx/model_qint8_avx512_vnni.onnx",
            "size": 118346824,
            "sha256": "dd476dd0c2514e9b9be83aeb3853fac0763e0bdf4a71645407587d77c48a2d88",
        },
        "tokenizer.json": {
            "source": "tokenizer.json",
            "size": 17082730,
            "sha256": "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
        },
    },
}
MODEL_ID = digest(MODEL_SPEC)


def file_hash(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def validate_model(directory):
    for name, meta in MODEL_SPEC["files"].items():
        path = Path(directory) / name
        if not path.is_file() or path.stat().st_size != meta["size"]:
            raise ValueError("Локальная модель отсутствует или имеет неверный размер")
        if file_hash(path) != meta["sha256"]:
            raise ValueError("Не совпадает контрольная сумма локальной модели")


def download_model(directory):
    import requests

    if os.name == "nt":
        import truststore

        truststore.inject_into_ssl()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Only this explicit command downloads. Startup and inference never call it.
    for name, meta in MODEL_SPEC["files"].items():
        existing = directory / name
        if existing.exists():
            if existing.stat().st_size != meta["size"] or file_hash(existing) != meta["sha256"]:
                raise ValueError("Существующий файл модели не соответствует зафиксированной версии")
            continue
        url = (
            f"https://huggingface.co/{MODEL_SPEC['repo']}/resolve/"
            f"{MODEL_SPEC['revision']}/{meta['source']}"
        )
        temporary = directory / (name + ".partial")
        try:
            with requests.get(url, stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                size = 0
                with temporary.open("xb") as stream:
                    for block in response.iter_content(1024 * 1024):
                        size += len(block)
                        if size > meta["size"]:
                            raise ValueError("Размер загрузки превысил зафиксированный лимит")
                        stream.write(block)
            if size != meta["size"] or file_hash(temporary) != meta["sha256"]:
                raise ValueError("Не совпадает контрольная сумма загруженной модели")
            temporary.replace(directory / name)
        finally:
            temporary.unlink(missing_ok=True)
    (directory / "manifest.json").write_text(
        json.dumps({"model_id": MODEL_ID, "spec": MODEL_SPEC}, indent=2) + "\n", encoding="utf-8"
    )


def open_cpu_session(directory):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return ort.InferenceSession(
        str(Path(directory) / "model.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


class LocalEncoder:
    """Use only inside the bounded inference process. No Torch, CUDA or HTTP calls."""

    def __init__(self, directory):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import numpy as np
        from tokenizers import Tokenizer

        validate_model(directory)
        self.np = np
        self.tokenizer = Tokenizer.from_file(str(Path(directory) / "tokenizer.json"))
        self.session = open_cpu_session(directory)
        self.inputs = {v.name for v in self.session.get_inputs()}

    def encode(self, text, kind):
        if kind not in MODEL_SPEC["prefixes"] or not text or len(text) > 6000:
            raise ValueError("Некорректный профессиональный текст")
        prefix = self.tokenizer.encode(MODEL_SPEC["prefixes"][kind], add_special_tokens=False).ids
        body = self.tokenizer.encode(text, add_special_tokens=False).ids
        capacity = MODEL_SPEC["max_tokens"] - len(prefix) - 2
        vectors, consumed = [], 0
        for start in range(0, len(body), capacity - MODEL_SPEC["chunk_stride"]):
            part = body[start : start + capacity]
            ids = [
                self.tokenizer.token_to_id("<s>"),
                *prefix,
                *part,
                self.tokenizer.token_to_id("</s>"),
            ]
            array = self.np.array([ids], dtype=self.np.int64)
            feed = {
                "input_ids": array,
                "attention_mask": self.np.ones_like(array),
                "token_type_ids": self.np.zeros_like(array),
            }
            output = self.session.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0]
            if output.ndim != 3 or output.shape[-1] != MODEL_SPEC["dimension"]:
                raise ValueError("Неожиданный формат выхода ONNX")
            vectors.append(output[0].mean(axis=0))  # batch=1, no padding, mask is all ones.
            consumed = start + len(part)
            if consumed >= len(body) or len(vectors) >= MODEL_SPEC["max_chunks"]:
                break
        value = self.np.mean(vectors, axis=0)
        norm = self.np.linalg.norm(value)
        if not self.np.isfinite(value).all() or norm <= 0:
            raise ValueError("Некорректный вектор модели")
        return {
            "vector": (value / norm).tolist(),
            "chunks": len(vectors),
            "tokens": len(body),
            "truncated": consumed < len(body),
        }
