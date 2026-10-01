from asyncpg import PostgresError
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

DATABASE_ERRORS = (SQLAlchemyError, PostgresError, ConnectionError, TimeoutError)


def create_engine(database_url: str) -> AsyncEngine:
    url = make_url(database_url)
    if url.drivername not in {"postgresql+asyncpg", "sqlite+aiosqlite"}:
        raise ValueError("Use postgresql+asyncpg (or sqlite+aiosqlite for tests)")
    return create_async_engine(url, pool_pre_ping=True)


def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)
