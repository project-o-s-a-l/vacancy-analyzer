"""Disposable encoder worker. Never exposes input text or vectors to logs."""

import json
import sys
import time
from pathlib import Path


def run(input_path, output_path, model_path, model_name):
    if sys.stdin.read(1) != "1":
        raise ValueError("Worker was not released")
    from jobs.semantic_encoder import LocalSectionEncoder

    if input_path.stat().st_size > 16 * 1024 * 1024 or output_path.exists():
        raise ValueError("Invalid worker paths")
    tasks = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 5000:
        raise ValueError("Invalid worker tasks")
    started = time.perf_counter()
    encoder = LocalSectionEncoder(model_path, model_name)
    load_seconds = time.perf_counter() - started
    values = [encoder.encode(task["sections"], task["kind"]) for task in tasks]
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "values": values,
                "load_seconds": load_seconds,
                "elapsed_seconds": time.perf_counter() - started,
            },
            stream,
        )


if __name__ == "__main__":
    try:
        run(*(Path(value) for value in sys.argv[1:4]), sys.argv[4])
    except Exception:
        print("Bounded semantic encoder failed; no input contents logged.", file=sys.stderr)
        raise SystemExit(1) from None
