"""Reserve attempts before I/O, including failed/cancelled attempts and restarts."""

import asyncio
import logging
import time
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime

import httpx

from collectors.base import SearchCard
from collectors.errors import CollectionStopped
from observability import emit_event
from storage.models import CollectionBudget

logger = logging.getLogger(__name__)


def aware(value):
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class BudgetDeferred(Exception):
    """A budget/cooldown is not a failed or closed vacancy."""


class AvitoRateLimited(RuntimeError):
    def __init__(self, retry_at, code):
        super().__init__("Доступ к выдаче Авито ограничен")
        self.retry_at = retry_at
        self.code = code


class InitialSearchThrottled(AvitoRateLimited):
    def __init__(self, retry_at):
        super().__init__(retry_at, "avito_search_throttled")


def pending_http_backoff(budget: CollectionBudget, now: datetime) -> tuple[int, datetime] | None:
    """Read a persisted HTTP backoff, separate from normal request spacing."""
    error = budget.error or ""
    status = error.removeprefix("http_")
    if not error.startswith("http_") or not status.isdecimal():
        return None
    code = int(status)
    retry_at = aware(budget.next_request_at)
    if (code == 429 or code >= 500) and retry_at > now:
        return code, retry_at
    return None


class BudgetedSource:
    def __init__(
        self,
        source,
        sessions,
        *,
        search_limit: int | None = 30,
        detail_limit: int | None = 90,
        new_limit: int | None = None,
        stop_on_initial_429: bool = False,
        delay=10,
        request_intervals: dict[str, float] | None = None,
        now=lambda: datetime.now(UTC),
        sleep=asyncio.sleep,
    ):
        self.source, self.sessions = source, sessions
        self.search_limit, self.detail_limit, self.delay = search_limit, detail_limit, delay
        self.new_limit = new_limit
        self.stop_on_initial_429 = stop_on_initial_429
        self._received_search_page = False
        self.request_intervals = request_intervals or {}
        self.now, self.sleep = now, sleep
        self.paging_kind, self.depth_limit = source.paging_kind, source.depth_limit
        self.window_kind = getattr(source, "window_kind", None)
        self.experience_levels = getattr(source, "experience_levels", ())
        self.catalog_roles = getattr(source, "catalog_roles", ())
        self.it_search_terms = getattr(source, "it_search_terms", ())
        self.matches_catalog = getattr(source, "matches_catalog", None)
        self.inline_details = getattr(source, "inline_details", False)
        self.source_name = getattr(source, "source_name", "hh")

    async def _row(self, session):
        row = await session.get(CollectionBudget, self.source_name, with_for_update=True)
        if row is None:
            row = CollectionBudget(
                source=self.source_name,
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

    def _start_window_if_due(self, row, now):
        if aware(row.window_at) + timedelta(hours=1) <= now:
            row.window_at, row.search_used, row.detail_used = now, 0, 0
            if "new_saved_in_window" in row.metrics:
                row.metrics = dict(row.metrics) | {"new_saved_in_window": 0}

    async def record_new(self, session):
        """Count the insert in the same transaction as its vacancy and observation."""
        row = await self._row(session)
        self._start_window_if_due(row, self.now())
        row.metrics = dict(row.metrics) | {
            "new_saved_in_window": row.metrics.get("new_saved_in_window", 0) + 1
        }

    async def _call(self, kind, call):
        try:
            return await self._reserved_call(kind, call)
        except httpx.HTTPStatusError as exc:
            if self.stop_on_initial_429 and exc.response.status_code == 429:
                # The attempt and the server's deadline are already committed.
                async with self.sessions() as session:
                    budget = await session.get(CollectionBudget, self.source_name)
                    if budget and (backoff := pending_http_backoff(budget, self.now())):
                        self.check_throttle(budget, *backoff)
            raise

    async def _reserved_call(self, kind, call):
        async with self.sessions.begin() as session:
            row = await self._row(session)
            now = self.now()
            if self.stop_on_initial_429 and (backoff := pending_http_backoff(row, now)):
                self.check_throttle(row, *backoff)
            if row.blocked:
                emit_event(logger, "collection.deferred", code="source_blocked", result="deferred")
                raise BudgetDeferred("source_blocked")
            self._start_window_if_due(row, now)
            saved = row.metrics.get("new_saved_in_window", 0)
            if self.new_limit is not None and saved >= self.new_limit:
                emit_event(logger, "collection.deferred", code="source_budget", result="deferred")
                raise BudgetDeferred("new_vacancy_budget")
            used = getattr(row, f"{kind}_used")
            limit = getattr(self, f"{kind}_limit")
            if limit is not None and used >= limit:
                emit_event(logger, "collection.deferred", code="source_budget", result="deferred")
                raise BudgetDeferred(f"{kind}_budget")
            wait = max(0, (aware(row.next_request_at) - now).total_seconds())
            if wait > self.delay + 1:
                emit_event(logger, "collection.deferred", code="source_cooldown", result="deferred")
                raise BudgetDeferred("source_cooldown")
            if self.new_limit is not None and kind == "detail":
                next_insert_slot = aware(row.window_at) + timedelta(
                    seconds=3600 * saved / self.new_limit
                )
                wait = max(wait, (next_insert_slot - now).total_seconds())
            if spacing := self.request_intervals.get(kind):
                slot_key = f"{kind}_slot_at"
                metrics = dict(row.metrics)
                if previous := metrics.get(slot_key):
                    next_slot = datetime.fromisoformat(previous) + timedelta(seconds=spacing)
                    wait = max(wait, (next_slot - now).total_seconds())
                metrics[slot_key] = (now + timedelta(seconds=wait)).isoformat()
                row.metrics = metrics
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
                metrics.pop("rate_limit_reason", None)
                if error is None:
                    row.error = None
                if error:
                    response = getattr(error, "response", None)
                    status = getattr(response, "status_code", None) or getattr(
                        error, "status_code", None
                    )
                    code = getattr(error, "code", "request_failed")
                    row.error = f"http_{status}" if status else code
                    if status == 429 and (reason := getattr(error, "rate_limit_reason", None)):
                        metrics["rate_limit_reason"] = reason
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
                row.metrics = metrics

    def check_throttle(self, budget, status, retry_at):
        if not self.stop_on_initial_429 or status != 429:
            return
        if reason := budget.metrics.get("rate_limit_reason"):
            raise AvitoRateLimited(retry_at, reason)
        if not self._received_search_page:
            raise InitialSearchThrottled(retry_at)

    async def search_page(self, query, page):
        result = await self._call("search", lambda: self.source.search_page(query, page))
        self._received_search_page = True
        return result

    async def get_vacancy(self, source_id):
        return await self._call("detail", lambda: self.source.get_vacancy(source_id))

    async def get_observed(self, card: SearchCard):
        if card.payload is not None and self.inline_details:
            # The listing request already paid for this full API record.
            return await self.source.get_observed(card)
        if card.detail_url and hasattr(self.source, "get_observed"):
            return await self._call("detail", lambda: self.source.get_observed(card))
        return await self.get_vacancy(card.source_id)


async def resume_source(sessions, source_name="hh"):
    async with sessions.begin() as session:
        row = await session.get(CollectionBudget, source_name, with_for_update=True)
        if row:
            row.blocked = False
            # Releasing a hard stop must preserve an active HTTP backoff and its reason.
            if pending_http_backoff(row, datetime.now(UTC)) is None:
                row.error = None
                if "rate_limit_reason" in row.metrics:
                    metrics = dict(row.metrics)
                    del metrics["rate_limit_reason"]
                    row.metrics = metrics
