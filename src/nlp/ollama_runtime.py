"""One local Ollama service owns weights. No Python handler loads a model."""

import asyncio
import ctypes
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx

from nlp import ollama_slot
from observability import emit_event

logger = logging.getLogger(__name__)


def available_memory():
    if os.name == "nt":

        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong)
                for name in (
                    "total",
                    "available",
                    "page_total",
                    "page_available",
                    "virtual_total",
                    "virtual_available",
                    "extended",
                )
            ]

        value = Memory()
        value.length = ctypes.sizeof(value)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
            raise OSError("Cannot read memory budget")
        return value.available
    values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(values["MemAvailable"].split()[0]) * 1024
    maximum, current = Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")
    if maximum.exists() and current.exists() and maximum.read_text().strip() != "max":
        available = min(available, int(maximum.read_text()) - int(current.read_text()))
    return available


class OllamaRuntime:
    def __init__(self, client, *, model, keep_alive=30, min_available_mb=4096, request_seconds=180):
        self.client, self.model = client, model
        self.keep_alive = keep_alive
        self.min_available = min_available_mb * 1024**2
        self.request_seconds = request_seconds

    @property
    def identity(self):
        return {"url": str(self.client.base_url).rstrip("/"), "model": self.model}

    @property
    def marker(self):
        return ollama_slot.LOCK_PATH.with_suffix(".uncertain.json")

    async def unload(self, model=None):
        # Unload ACK only schedules expiration (Ollama sched.expireRunner). It does
        # NOT prove generation stopped: wait until the runner disappears from /ps.
        async with asyncio.timeout(10):
            response = await self.client.post(
                "/api/generate",
                json={"model": model or self.model, "keep_alive": 0, "stream": False},
                timeout=10,
            )
            if response.status_code != 404:
                response.raise_for_status()
            if response.status_code != 404 and response.json().get("done") is not True:
                raise ValueError("Unconfirmed unload")
            while True:
                state = await self.client.get("/api/ps", timeout=5)
                state.raise_for_status()
                if not any(
                    m.get("name") == (model or self.model)
                    or m.get("model") == (model or self.model)
                    for m in state.json()["models"]
                ):
                    break
                await asyncio.sleep(0.1)
        emit_event(logger, "model.unloaded", result="success")

    async def drain(self):
        async with ollama_slot.model_slot():
            if (
                self.marker.exists()
                and json.loads(self.marker.read_text(encoding="utf-8"))["runtime"] != self.identity
            ):
                raise ValueError(
                    "Drain must target the runtime recorded by the interrupted request"
                )
            await self.unload()
            self.marker.unlink(missing_ok=True)

    async def maintain(self):
        if available_memory() < self.min_available:
            async with ollama_slot.model_slot():
                if not self.marker.exists():
                    await self.unload()

    @asynccontextmanager
    async def request(self, body):
        from nlp.ollama import OllamaUnavailable

        async with ollama_slot.model_slot():
            if self.marker.exists():
                # Native direct callers share the same crash quarantine. Never retry
                # a generation just because a client stopped waiting for its response.
                marker = json.loads(self.marker.read_text(encoding="utf-8"))
                if marker["until"] > time.time():
                    raise OllamaUnavailable("Ожидается завершение предыдущего запроса модели")
                if marker["runtime"] != self.identity:
                    raise OllamaUnavailable("Требуется drain прежнего локального runtime")
                await self.unload()
                self.marker.unlink()
            if available_memory() < self.min_available:
                await self.unload()
                emit_event(
                    logger, "model.memory_deferred", result="deferred", code="memory_pressure"
                )
                raise OllamaUnavailable("Недостаточно свободной памяти; разбор отложен")
            # A shared service must be dedicated to this application. Enforce a single
            # resident model even if an operator forgot the server environment limits.
            response = await self.client.get("/api/ps", timeout=5)
            response.raise_for_status()
            for loaded in response.json()["models"]:
                if loaded["name"] != self.model:
                    await self.unload(loaded["name"])
            body["keep_alive"] = self.keep_alive  # Per-request value wins over server env.
            temporary = self.marker.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(
                    {"until": time.time() + self.request_seconds + 60, "runtime": self.identity},
                    stream,
                )
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.marker)
            try:
                yield
            except BaseException:
                raise  # Keep durable quarantine for timeout, disconnect and process cancellation.
            else:
                self.marker.unlink(missing_ok=True)
                if available_memory() < self.min_available:
                    try:
                        await self.unload()
                    except (httpx.HTTPError, ValueError, TimeoutError):
                        pass  # The completed result remains usable; server idle TTL still applies.


@asynccontextmanager
async def request_slot(runtime, body):
    if runtime is None:
        async with ollama_slot.model_slot():
            yield
    else:
        async with runtime.request(body):
            yield
