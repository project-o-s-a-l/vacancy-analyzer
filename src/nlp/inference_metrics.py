"""Payload-free stage timings, also consumable by the reproducible measurement."""

import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar

from observability import emit_event

logger = logging.getLogger(__name__)
measurements = ContextVar("inference_measurements", default=None)


def record(stage, milliseconds, *, result="success", **fields):
    row = {"stage": stage, "duration_ms": milliseconds, "result": result, **fields}
    sink = measurements.get()
    if sink is not None:
        sink.append(row)
    emit_event(logger, "model.stage_finished", **row)


@contextmanager
def stage(name):
    started = time.perf_counter()
    outcome = "success"
    try:
        yield
    except BaseException:
        outcome = "failed"
        raise
    finally:
        record(name, (time.perf_counter() - started) * 1000, result=outcome)


def server_timings(payload):
    for source, name in (
        ("load_duration", "model_load"),
        ("prompt_eval_duration", "prompt_eval"),
        ("eval_duration", "inference"),
    ):
        value = payload.get(source)
        if type(value) in {int, float} and value >= 0:
            record(name, value / 1_000_000)


def cache_key(kind, model, model_digest, body):
    from domain.professional import digest
    from nlp.analysis_identity import pipeline_hash

    return digest(
        {
            "pipeline": pipeline_hash(kind, model),
            "weights": model_digest,
            "body": {k: v for k, v in body.items() if k != "keep_alive"},
        }
    )
