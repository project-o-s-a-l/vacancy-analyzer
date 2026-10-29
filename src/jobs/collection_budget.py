"""Reserve attempts before I/O, including failed/cancelled attempts and restarts."""

import asyncio
import logging
import time
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime

from collectors.errors import CollectionStopped
from observability import emit_event
from storage.models import CollectionBudget

logger = logging.getLogger(__name__)


def aware(value):
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class BudgetDeferred(Exception):
    """A budget/cooldown is not a failed or closed vacancy."""


class BudgetedSource:
    def __init__(
        self,
        source,
        sessions,
        *,
        search_limit=30,
        detail_limit=90,
        delay=10,
        now=lambda: datetime.now(UTC),
        sleep=asyncio.sleep,
    ):
        self.source, self.sessions = source, sessions
        self.search_limit, self.detail_limit, self.delay = search_limit, detail_limit, delay
        self.now, self.sleep = now, sleep
        self.paging_kind, self.depth_limit = source.paging_kind, source.depth_limit

    async def _row(self, session):
        row = await session.get(CollectionBudget, "hh", with_for_update=True)
        if row is None:
            row = CollectionBudget(
                source="hh",
                window_at=self.now(),
                search_used=0,
                detail_used=0,
                next_request_at=self.now(),
                blocked=False,
                metrics={},
            )
            session.add(row)
            await session.flush()
        return row

    async def _call(self, kind, call):
        async with self.sessions.begin() as session:
            row = await self._row(session)
            now = self.now()
            if row.blocked:
                emit_event(logger, "collection.deferred", code="source_blocked", result="deferred")
                raise BudgetDeferred("source_blocked")
            if aware(row.window_at) + timedelta(hours=1) <= now:
                row.window_at, row.search_used, row.detail_used = now, 0, 0
            used = getattr(row, f"{kind}_used")
            if used >= getattr(self, f"{kind}_limit"):
                emit_event(logger, "collection.deferred", code="source_budget", result="deferred")
                raise BudgetDeferred(f"{kind}_budget")
            wait = max(0, (aware(row.next_request_at) - now).total_seconds())
            if wait > self.delay + 1:
                emit_event(logger, "collection.deferred", code="source_cooldown", result="deferred")
                raise BudgetDeferred("source_cooldown")
            setattr(row, f"{kind}_used", used + 1)
            row.next_request_at = now + timedelta(seconds=wait + self.delay)
        # Reservation is committed before sleeping/network; cancellation cannot refund it.
        await self.sleep(wait)
        if resume := getattr(self.source, "resume_requests", None):
            resume()
        started = time.monotonic()
        error = None
        try:
            return await call()
        except BaseException as exc:
            error = exc
            raise
        finally:
            async with self.sessions.begin() as session:
                row = await self._row(session)
                metrics = dict(row.metrics)
                for key, value in {
                    f"{kind}_requests": 1,
                    f"{kind}_ms": (time.monotonic() - started) * 1000,
                    f"{kind}_errors": int(error is not None),
                    f"{kind}_bytes": getattr(self.source, "response_bytes", 0),
                }.items():
                    metrics[key] = metrics.get(key, 0) + value
                row.metrics = metrics
                if error is None:
                    row.error = None
                if error:
                    response = getattr(error, "response", None)
                    status = getattr(response, "status_code", None) or getattr(
                        error, "status_code", None
                    )
                    code = getattr(error, "code", "request_failed")
                    row.error = f"http_{status}" if status else code
                    if status == 403 or isinstance(error, CollectionStopped) and status != 429:
                        row.blocked = True
                    if status == 429 or status and status >= 500:
                        value = (
                            response.headers.get("Retry-After", "") if response is not None else ""
                        )
                        if not value:
                            value = getattr(self.source, "retry_after", None)
                        try:
                            seconds = float(value)
                        except ValueError:
                            try:
                                seconds = (
                                    parsedate_to_datetime(value) - self.now()
                                ).total_seconds()
                            except (ValueError, TypeError, OverflowError):
                                seconds = 3600 if status == 429 else 300
                        row.next_request_at = self.now() + timedelta(seconds=max(60, seconds))

    async def search_page(self, query, page):
        return await self._call("search", lambda: self.source.search_page(query, page))

    async def get_vacancy(self, source_id):
        return await self._call("detail", lambda: self.source.get_vacancy(source_id))


async def resume_source(sessions):
    async with sessions.begin() as session:
        row = await session.get(CollectionBudget, "hh", with_for_update=True)
        if row:
            row.blocked, row.error = False, None
            # Explicit operator reset never refunds the request budget or Retry-After.
