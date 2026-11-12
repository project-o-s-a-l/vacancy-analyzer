"""Private worker protocol; no input or model exception contents in logs."""

import json
import sys
import time
from pathlib import Path


def run(source, target, directory):
    if sys.stdin.read(1) != "1" or source.stat().st_size > 8 * 1024 * 1024 or target.exists():
        raise ValueError("invalid_worker_input")
    from jobs.cross_encoder import LocalCrossEncoder
    from jobs.semantic_encoder import _release_local_model

    request = json.loads(source.read_text(encoding="utf-8"))
    tasks, repeats = request["tasks"], request["repeats"]
    if not 1 <= len(tasks) <= 32 or not 1 <= repeats <= 6:
        raise ValueError("invalid_worker_budget")
    start = time.perf_counter()
    _release_local_model()
    drain = time.perf_counter() - start
    start = time.perf_counter()
    encoder = LocalCrossEncoder(directory)
    load = time.perf_counter() - start
    durations, values = [], None
    for _ in range(repeats):
        start = time.perf_counter()
        current = encoder.score(tasks)
        durations.append(time.perf_counter() - start)
        if values is not None and values != current:
            raise ValueError("non_deterministic_logits")
        values = current
    with target.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "values": values,
                "drain_seconds": drain,
                "load_seconds": load,
                "inference_seconds": durations,
            },
            stream,
        )


if __name__ == "__main__":
    try:
        run(*(Path(arg) for arg in sys.argv[1:]))
    except Exception:
        raise SystemExit(1) from None
