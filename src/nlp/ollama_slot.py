"""One inference at a time for application adapters sharing an event loop."""

import asyncio
import errno
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from weakref import WeakKeyDictionary

from observability import emit_event

logger = logging.getLogger(__name__)

_slots = WeakKeyDictionary()
LOCK_PATH = Path(tempfile.gettempdir()) / "vacancy-analyzer-ollama.lock"


def _lock(stream):
    try:
        if os.name == "nt":
            import msvcrt

            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError as exc:
        if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
            return False
        raise


def _unlock(stream):
    if os.name == "nt":
        import msvcrt

        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@asynccontextmanager
async def _model_slot():
    loop = asyncio.get_running_loop()
    lock = _slots.setdefault(loop, asyncio.Lock())
    async with lock:
        # Native diagnostic callers also serialize across processes. Application
        # workers additionally use the shared PostgreSQL lease across containers.
        with LOCK_PATH.open("a+b") as stream:
            if stream.seek(0, 2) == 0:
                stream.write(b"0")
                stream.flush()
            while not _lock(stream):  # noqa: ASYNC110 - OS lock has no asyncio notification.
                await asyncio.sleep(0.1)
            try:
                yield
            finally:
                _unlock(stream)


@asynccontextmanager
async def model_slot():
    started = time.monotonic()
    acquired = False
    emit_event(logger, "model.slot_wait_started")
    try:
        async with _model_slot():
            acquired = True
            emit_event(
                logger,
                "model.slot_acquired",
                wait_ms=(time.monotonic() - started) * 1000,
                result="ready",
            )
            yield
    except asyncio.CancelledError:
        if not acquired:
            emit_event(
                logger,
                "model.slot_wait_cancelled",
                result="cancelled",
                wait_ms=(time.monotonic() - started) * 1000,
            )
        raise
