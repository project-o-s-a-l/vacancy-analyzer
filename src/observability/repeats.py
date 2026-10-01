"""Bounded suppression of unchanged recoverable warnings; ERROR always passes."""

import threading
import time
from collections import OrderedDict

RECOVERABLE = frozenset(
    {
        "collection.network_deferred",
        "analysis.startup_deferred",
        "analysis.unavailable",
        "analysis.deferred",
    }
)


class RepeatLimiter:
    def __init__(self, interval=60, max_keys=128, clock=time.monotonic):
        self.interval, self.max_keys, self.clock = interval, max_keys, clock
        self.keys = OrderedDict()
        self.mutex = threading.Lock()
        self.evicted = 0

    def filter(self, record):
        if record._safe_repeat is not None:
            return record._safe_repeat
        decision = True
        if record.levelno == 30 and record._safe_event in RECOVERABLE:
            fields = record._safe_fields
            key = (
                record._safe_event,
                record._safe_component,
                fields.get("code"),
                (record._safe_error or {}).get("type"),
                fields.get("stage"),
                record._safe_context.get("job_id"),
            )
            now = self.clock()
            with self.mutex:
                old = self.keys.get(key)
                if old and now - old[0] < self.interval:
                    self.keys[key] = (old[0], min(10**9, old[1] + 1))
                    decision = False
                else:
                    if old and old[1]:
                        fields["suppressed_count"] = old[1]
                    self.keys[key] = (now, 0)
                    self.keys.move_to_end(key)
                    if len(self.keys) > self.max_keys:
                        _, (_, count) = self.keys.popitem(last=False)
                        self.evicted = min(10**9, self.evicted + count)
        record._safe_repeat = decision
        return decision

    def drain(self):
        with self.mutex:
            result = [
                {"repeated_event": key[0], "suppressed_count": value[1]}
                for key, value in self.keys.items()
                if value[1]
            ]
            if self.evicted:
                result.append({"suppressed_count": self.evicted})
            self.keys.clear()
            self.evicted = 0
            return result
