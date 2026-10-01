"""Small versioned exact index artifact, reusable by CLI and a future frontend."""

import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from jobs.benchmark_data import vacancy_key
from jobs.semantic_model import MODEL_ID, MODEL_SPEC
from jobs.semantic_text import TEXT_SPEC, text_revision, vacancy_text
from scoring.serving_features import digest

MAX_INDEX_BYTES = 64 * 1024 * 1024
MAX_CATALOGUE = 5000


def valid_vector(vector):
    return (
        isinstance(vector, list)
        and len(vector) == MODEL_SPEC["dimension"]
        and all(type(v) in {int, float} and math.isfinite(v) for v in vector)
        and abs(sum(v * v for v in vector) - 1) < 0.001
    )


def load_index(path):
    with Path(path).open("rb") as stream:
        content = stream.read(MAX_INDEX_BYTES + 1)
    if len(content) > MAX_INDEX_BYTES:
        raise ValueError("Индекс превышает допустимый размер")
    value = json.loads(content)
    if (
        value.get("version") != "semantic-index-v1"
        or value.get("model_id") != MODEL_ID
        or value.get("text_spec") != TEXT_SPEC
        or not isinstance(value.get("entries"), dict)
        or len(value["entries"]) > MAX_CATALOGUE
    ):
        raise ValueError("Несовместимый индекс представлений")
    seal = value.pop("sha256", None)
    if digest(value) != seal:
        raise ValueError("Нарушена целостность индекса")
    for entry in value["entries"].values():
        if not isinstance(entry, dict) or not valid_vector(entry.get("vector")):
            raise ValueError("Некорректный вектор индекса")
    return value


def save_index(value, path):
    value = value | {"sha256": digest(value)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
        if temporary.stat().st_size > MAX_INDEX_BYTES:
            raise ValueError("Индекс превышает допустимый размер")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_index(vacancies, path, encode):
    if len(vacancies) > MAX_CATALOGUE:
        raise ValueError("Каталог превышает лимит точного эксперимента")
    previous = {}
    if Path(path).exists():
        # Invalid/incompatible input is never silently accepted or overwritten.
        previous = load_index(path)["entries"]
    entries, tasks, keys = {}, [], []
    for vacancy in vacancies:
        key, text = vacancy_key(vacancy), vacancy_text(vacancy)
        revision = text_revision(text)
        if key in entries:
            raise ValueError("Повторный идентификатор в каталоге")
        old = previous.get(key)
        if old and old.get("input_revision") == revision:
            entries[key] = old
        else:
            entries[key] = {"input_revision": revision}
            tasks.append({"text": text, "kind": "passage"})
            keys.append(key)
    computed = encode(tasks)
    if len(computed["values"]) != len(keys):
        raise ValueError("Неполный результат вычисления индекса")
    for key, embedding in zip(keys, computed["values"], strict=True):
        if not valid_vector(embedding["vector"]):
            raise ValueError("Некорректный вектор модели")
        entries[key].update(embedding)
    value = {
        "version": "semantic-index-v1",
        "model_id": MODEL_ID,
        "text_spec": TEXT_SPEC,
        "entries": entries,
    }
    save_index(value, path)
    return value, computed | {"computed": len(keys), "reused": len(entries) - len(keys)}
