"""Disposable single-thread CPU worker; launched only by an explicit experiment."""

import json
import logging
import sys
import time
from pathlib import Path

from observability import emit_event, logging_session, operation
from observability.settings import LoggingOptions

logger = logging.getLogger("jobs.semantic_worker")


def run(input_path, output_path, model_path):
    # Parent configures affinity/priority and starts monitoring before releasing the worker.
    if sys.stdin.read(1) != "1":
        raise ValueError("Worker launch was not released")
    from jobs.semantic_model import LocalEncoder

    if input_path.stat().st_size > 16 * 1024 * 1024 or output_path.exists():
        raise ValueError("Invalid worker files")
    tasks = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 5000:
        raise ValueError("Invalid worker batch")
    started = time.perf_counter()
    encoder = LocalEncoder(model_path)
    load_seconds = time.perf_counter() - started
    values = []
    for i, task in enumerate(tasks, 1):
        values.append(encoder.encode(task["text"], task["kind"]))
        if i % 25 == 0:
            emit_event(logger, "experiment.progress", count=i, total=len(tasks))
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
        options = LoggingOptions()
    except (OSError, ValueError):
        print("Неверные настройки журнала.", file=sys.stderr)
        raise SystemExit(1) from None
    with (
        logging_session(
            role="semantic_worker",
            level=options.level,
            console_format=options.console_format,
            file_options=options,
        ),
        operation("command"),
    ):
        try:
            run(*(Path(value) for value in sys.argv[1:4]))
        except KeyboardInterrupt:
            emit_event(logger, "application.cancelled", result="cancelled")
            raise SystemExit(130) from None
        except Exception as exc:
            emit_event(
                logger,
                "application.failed",
                level=logging.ERROR,
                error=exc,
                result="failed",
                code="unexpected_error",
            )
            raise SystemExit(1) from None
