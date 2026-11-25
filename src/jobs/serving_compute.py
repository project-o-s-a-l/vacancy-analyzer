"""Bounded CPU submissions. Cancellation never releases a still-running worker slot."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from contextvars import ContextVar, copy_context
from functools import partial, wraps
from weakref import WeakKeyDictionary

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="recommendation")
_limits = WeakKeyDictionary()
_requests = WeakKeyDictionary()
_inside = ContextVar("recommendation_slot", default=False)


class RecommendationBusy(ValueError):
    def __init__(self):
        super().__init__(
            "Подбор уже выполняется или очередь занята. Повторите /latest немного позже."
        )


def bounded_recommendations(function):
    @wraps(function)
    async def wrapped(*args, **kwargs):
        if _inside.get():
            return await function(*args, **kwargs)
        loop = asyncio.get_running_loop()
        state = _requests.setdefault(loop, [asyncio.Semaphore(2), 0])
        if state[1] >= 32:
            raise RecommendationBusy()
        state[1] += 1
        try:
            await asyncio.wait_for(state[0].acquire(), timeout=10)
        except TimeoutError as exc:
            state[1] -= 1
            raise RecommendationBusy() from exc
        except BaseException:
            state[1] -= 1
            raise
        token = _inside.set(True)
        try:
            return await function(*args, **kwargs)
        finally:
            _inside.reset(token)
            state[0].release()
            state[1] -= 1

    return wrapped


async def compute(function, *args, **kwargs):
    loop = asyncio.get_running_loop()
    limit = _limits.setdefault(loop, asyncio.Semaphore(2))
    async with limit:
        call = partial(copy_context().run, partial(function, *args, **kwargs))
        future = loop.run_in_executor(_executor, call)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # Python cannot stop an executing thread. Drain before exposing mutable inputs
            # to another request/closing its transaction. At most two submitted jobs.
            while not future.done():
                try:
                    await asyncio.shield(future)
                except asyncio.CancelledError:
                    continue
            future.result()
            raise


@asynccontextmanager
async def owner_selection(sessions, owner):
    """Avoid duplicate ranking across processes without reserving a pool connection."""
    from datetime import UTC, datetime, timedelta
    from uuid import uuid4

    from sqlalchemy import delete
    from sqlalchemy.dialects.postgresql import insert

    from storage.models.selection import RecommendationLease as Lease

    token = uuid4()
    async with sessions.begin() as session:
        postgres = session.get_bind().dialect.name == "postgresql"
        if postgres:
            now = datetime.now(UTC)
            statement = insert(Lease).values(
                owner_id=owner, token=token, expires_at=now + timedelta(minutes=5)
            )
            acquired = await session.scalar(
                statement.on_conflict_do_update(
                    index_elements=[Lease.owner_id],
                    set_={"token": token, "expires_at": statement.excluded.expires_at},
                    where=Lease.expires_at <= now,
                ).returning(Lease.token)
            )
            if acquired is None:
                raise RecommendationBusy()
    try:
        yield
    finally:
        if postgres:
            async with sessions.begin() as session:
                await session.execute(
                    delete(Lease).where(Lease.owner_id == owner, Lease.token == token)
                )
