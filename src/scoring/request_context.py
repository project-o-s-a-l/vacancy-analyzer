"""Request-local reuse only; ContextVar isolates concurrent users and resets on cancellation."""

from contextvars import ContextVar
from functools import wraps

_current = ContextVar("comparison_request", default=None)


def prepared_profile_request(function):
    @wraps(function)
    async def wrapped(repository, profile, *args, **kwargs):
        current = _current.get()
        if current is not None and current[0] is profile:
            return await function(repository, profile, *args, **kwargs)
        token = _current.set((profile, {}))
        try:
            return await function(repository, profile, *args, **kwargs)
        finally:
            _current.reset(token)

    return wrapped


def profile_value(profile, key, calculate):
    current = _current.get()
    if current is None or current[0] is not profile:
        return calculate()
    if key not in current[1]:
        from scoring.serving_cache import cached_value, profiles

        current[1][key] = cached_value(profiles, key, calculate)
    return current[1][key]
