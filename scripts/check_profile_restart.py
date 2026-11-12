"""Verify durable profile/editor state in a fresh Python process on disposable PG."""

import asyncio
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from jobs.users import UserService
from storage.models import Base

SCRIPT_PATH = str(Path(__file__).resolve())


async def read(url, schema):
    engine = create_async_engine(url, connect_args={"server_settings": {"search_path": schema}})
    try:
        users = UserService(async_sessionmaker(engine, expire_on_commit=False))
        user, edit = await users.get(42), await users.editor.get(42)
        return dict(
            profile=user.profile.model_dump(mode="json"),
            draft=edit.profile.model_dump(mode="json"),
            view=edit.view,
            token=edit.token,
            revision=user.revision,
        )
    finally:
        await engine.dispose()


async def check(url):
    schema = "f15_restart_" + uuid4().hex
    admin = create_async_engine(url)
    engine = create_async_engine(url, connect_args={"server_settings": {"search_path": schema}})
    try:
        async with admin.begin() as connection:
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        users = UserService(async_sessionmaker(engine, expire_on_commit=False))
        await users.register(42)
        edit = await users.editor.open(42)
        edit = await users.editor.act(42, edit.token, ["set", "skill", "", "new", "Python"])
        edit = await users.editor.act(42, edit.token, ["set", "episode", "", "new", "employment"])
        key = edit.view[2]
        edit = await users.editor.act(42, edit.token, ["set", "episode", key, "roles", "Инженер"])
        await users.editor.act(42, edit.token, ["save"])
        edit = await users.editor.open(42)
        await users.editor.act(42, edit.token, ["set", "episode", key, "tasks", "Готовить отчёты"])
        expected = await read(url, schema)
        assert expected["profile"] != expected["draft"]
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            SCRIPT_PATH,
            "--read",
            schema,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        output, error = await child.communicate()
        if child.returncode or json.loads(output) != expected:
            raise AssertionError("Fresh process profile/editor round trip failed")
        return dict(
            status="passed",
            fresh_python_process=True,
            confirmed_profile=True,
            unconfirmed_editor=True,
            separate_from_active=True,
        )
    finally:
        await engine.dispose()
        async with admin.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await admin.dispose()


if __name__ == "__main__":
    url = os.environ["TEST_DATABASE_URL"]
    parsed = make_url(url)
    if parsed.get_backend_name() != "postgresql" or "test" not in (parsed.database or ""):
        raise SystemExit("TEST_DATABASE_URL must explicitly name a disposable PostgreSQL test DB")
    result = asyncio.run(read(url, sys.argv[2]) if sys.argv[1:2] == ["--read"] else check(url))
    print(json.dumps(result))
