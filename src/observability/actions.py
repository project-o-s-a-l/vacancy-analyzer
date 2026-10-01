"""Operation boundaries without inspecting arguments, results or exception text."""

import asyncio
import logging
import time
from functools import wraps

from observability import emit_event, operation


def observed(kind):
    def decorate(function):
        @wraps(function)
        async def wrapped(*args, **kwargs):
            with operation(kind):
                started = time.monotonic()
                try:
                    return await function(*args, **kwargs)
                except BaseException as exc:
                    rejected = isinstance(exc, ValueError) and kind in {
                        "resume",
                        "history",
                        "feedback",
                    }
                    cancelled = isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt))
                    emit_event(
                        logging.getLogger(function.__module__),
                        "operation.finished",
                        result="rejected" if rejected else "cancelled" if cancelled else "failed",
                        level=logging.INFO
                        if rejected
                        else logging.WARNING
                        if cancelled
                        else logging.ERROR,
                        error=None if rejected else exc,
                        duration_ms=(time.monotonic() - started) * 1000,
                    )
                    raise

        return wrapped

    return decorate
