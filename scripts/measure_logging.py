"""Small sequential logging measurement. No DB, network, model or working data."""

import argparse
import io
import json
import logging
import time
import tracemalloc
from contextlib import redirect_stderr
from pathlib import Path

from observability import emit_event, logging_session, operation
from observability.settings import LoggingOptions

MAX_EVENTS = 200


def measure(directory, events=MAX_EVENTS):
    if not 1 <= events <= MAX_EVENTS:
        raise ValueError("Разрешено от 1 до 200 искусственных событий")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    results = []
    logger = logging.getLogger("jobs.logging_measurement")
    # Sequential modes; file mode measures the default queue, not a larger test queue.
    for mode in ("disabled", "console", "console_and_file"):
        settings = LoggingOptions(
            _env_file=None,
            directory=directory / "logs",
            file_enabled=mode == "console_and_file",
            max_file_bytes=5 * 1024**2,
            backup_count=3,
            total_max_bytes=100 * 1024**2,
            queue_size=256,
            shutdown_timeout=2,
        )
        stream = io.StringIO()
        tracemalloc.start()
        started = time.perf_counter()
        with (
            redirect_stderr(stream),
            logging_session(
                role="cli",
                level="ERROR" if mode == "disabled" else "INFO",
                console_format="json",
                file_options=settings,
            ),
        ):
            with operation("command"):
                emitting = time.perf_counter()
                for index in range(events):
                    emit_event(logger, "experiment.progress", count=index, result="success")
                enqueue_ms = (time.perf_counter() - emitting) * 1000
        total_ms = (time.perf_counter() - started) * 1000
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows = [json.loads(line) for line in stream.getvalue().splitlines()]
        results.append(
            {
                "mode": mode,
                "events": events,
                "emission_ms": round(enqueue_ms, 3),
                "total_with_setup_and_flush_ms": round(total_ms, 3),
                "python_peak_bytes": peak,
                "console_events": len(rows),
                "loss_notices": sum(row["event"] == "logging.file_degraded" for row in rows),
            }
        )
    paths = list((directory / "logs").glob("*.jsonl"))
    file_rows = [
        json.loads(line) for path in paths for line in path.read_text(encoding="utf-8").splitlines()
    ]
    return {
        "version": 1,
        "max_events": MAX_EVENTS,
        "queue_size": 256,
        "modes": results,
        "file_events": len(file_rows),
        "file_bytes": sum(path.stat().st_size for path in paths),
        "complete": len(file_rows) == events and not any(row["loss_notices"] for row in results),
        "scope": "synthetic_single_process_python_allocations_not_rss",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True, help="Новый тестовый каталог")
    parser.add_argument("--events", type=int, choices=range(1, MAX_EVENTS + 1), default=MAX_EVENTS)
    args = parser.parse_args()
    report = measure(args.directory, args.events)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
