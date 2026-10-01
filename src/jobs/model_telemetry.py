"""Transport diagnostics without payloads. Analyzer validation and requests stay unchanged."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager

import httpx

from nlp.analysis_identity import pipeline_hash
from observability import bind_context, emit_event

logger = logging.getLogger(__name__)


def validation_code(error):
    """Only known exception types; never inspect messages, payloads or validation inputs."""
    seen = set()
    for _ in range(8):
        if error is None or id(error) in seen:
            break
        seen.add(id(error))
        code = {
            "InputTooLarge": "input_too_large",
            "UnknownEvidence": "unknown_evidence",
            "ValidationError": "invalid_schema",
            "JSONDecodeError": "invalid_json",
        }.get(type(error).__name__)
        if code:
            return code
        error = error.__cause__ or error.__context__
    return "invalid_response"


class ObservedOllamaClient(httpx.AsyncClient):
    def __init__(self, *args, model, job_kind, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_model, self.log_kind = model, job_kind
        self.log_pipeline = pipeline_hash(job_kind, model)

    @asynccontextmanager
    async def _request(self, stage):
        started = time.monotonic()
        with bind_context(model=self.log_model, job_kind=self.log_kind, pipeline=self.log_pipeline):
            emit_event(logger, "model.request_started", stage=stage)
            try:
                yield
            except asyncio.CancelledError:
                emit_event(
                    logger,
                    "model.request_failed",
                    level=logging.WARNING,
                    stage=stage,
                    result="cancelled",
                    duration_ms=(time.monotonic() - started) * 1000,
                )
                raise
            except httpx.HTTPError as exc:
                code = (
                    "model_timeout"
                    if isinstance(exc, httpx.TimeoutException)
                    else "model_unavailable"
                )
                status = (
                    exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
                )
                if status and status < 500 and status != 429:
                    code = "model_http_rejected"
                emit_event(
                    logger,
                    "model.request_failed",
                    level=logging.WARNING,
                    stage=stage,
                    result="failed" if code == "model_http_rejected" else "deferred",
                    code=code,
                    status_code=status,
                    duration_ms=(time.monotonic() - started) * 1000,
                )
                raise
            except Exception:
                emit_event(
                    logger,
                    "model.response_rejected",
                    level=logging.WARNING,
                    stage=stage,
                    result="failed",
                    code="invalid_response",
                    duration_ms=(time.monotonic() - started) * 1000,
                )
                raise
            else:
                emit_event(
                    logger,
                    "model.request_finished",
                    stage=stage,
                    result="success",
                    duration_ms=(time.monotonic() - started) * 1000,
                )

    async def send(self, request, **kwargs):
        if request.url.path != "/api/tags":
            return await super().send(request, **kwargs)
        async with self._request("model_ready"):
            return await super().send(request, **kwargs)

    @asynccontextmanager
    async def stream(self, *args, **kwargs):
        async with self._request("model_chat"), super().stream(*args, **kwargs) as response:
            yield response
