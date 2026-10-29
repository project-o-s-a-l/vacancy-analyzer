"""Private bounded in-process profile/pair caches. No catalogue, text or contact logs."""

import sys
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy


def retained_size(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(retained_size(k, seen) + retained_size(v, seen) for k, v in value.items())
    elif isinstance(value, (tuple, list, set, frozenset)):
        size += sum(retained_size(v, seen) for v in value)
    elif hasattr(value, "__dict__"):
        size += retained_size(value.__dict__, seen)
    return size


class BoundedCache:
    def __init__(self, max_bytes, max_entries, ttl=900):
        self.max_bytes, self.max_entries, self.ttl = max_bytes, max_entries, ttl
        self.entries = OrderedDict()
        self.size = 0
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            row = self.entries.pop(key, None)
            if row is None:
                return None
            if row[0] < time.monotonic():
                self.size -= row[2]
                return None
            self.entries[key] = row
        return deepcopy(row[1])

    def put(self, key, value):
        size = retained_size(value) + retained_size(key)
        if size > self.max_bytes // 4:
            return
        with self.lock:
            old = self.entries.pop(key, None)
            if old:
                self.size -= old[2]
            self.entries[key] = (time.monotonic() + self.ttl, deepcopy(value), size)
            self.size += size
            while self.size > self.max_bytes or len(self.entries) > self.max_entries:
                _, row = self.entries.popitem(last=False)
                self.size -= row[2]


profiles = BoundedCache(4 * 1024 * 1024, 1024)
pairs = BoundedCache(16 * 1024 * 1024, 2048)
_scope = ContextVar("serving_cache_scope", default=None)


def vacancy_revision(vacancy):
    from scoring.serving_features import digest

    return digest(
        dict(
            source=vacancy.model_dump(
                mode="json",
                exclude={"raw_payload", "collected_at", "professional_facts", "last_checked_at"},
            ),
            facts=vacancy.professional_facts.revision if vacancy.professional_facts else None,
        )
    )


@contextmanager
def cache_scope(identity):
    token = _scope.set(identity)
    try:
        yield
    finally:
        _scope.reset(token)


def cached_value(cache, key, calculate):
    scope = _scope.get()
    if scope is None:
        return calculate()
    key = (scope, key)
    value = cache.get(key)
    if value is None:
        value = calculate()
        cache.put(key, value)
    return value
