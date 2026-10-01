"""Host wiring and privacy checks use artificial secrets, never live services."""

import asyncio
import io
import json
import logging
import subprocess
import sys
from datetime import datetime
from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx
import pytest
import requests
from pydantic import BaseModel, ValidationError
from sqlalchemy import create_engine, text
from sqlalchemy.exc import StatementError

from observability import (
    configure_logging,
    current_context,
    emit_event,
    logging_session,
    operation,
    shutdown_logging,
)
from observability.logging import MAX_RECORD_BYTES, ConsoleHandler

logger = logging.getLogger("jobs.logging_test")
SECRET = "CONTROL_SECRET_resume_Иван+79991234567@example.com\nFORGED_EVENT"


def records(capsys):
    captured = capsys.readouterr()
    assert captured.out == ""
    return [json.loads(line) for line in captured.err.splitlines()]


def test_import_has_no_logging_side_effect():
    code = """
import logging
before = (tuple(logging.getLogger().handlers), logging.getLogRecordFactory())
import observability
assert before == (tuple(logging.getLogger().handlers), logging.getLogRecordFactory())
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, timeout=20)


def test_idempotent_setup_and_restart(capsys):
    factory = logging.getLogRecordFactory()
    handlers = tuple(logging.getLogger().handlers)
    with logging_session(role="bot", console_format="json") as first:
        assert configure_logging(role="worker", console_format="json") == first
        assert len([h for h in logging.getLogger().handlers if isinstance(h, ConsoleHandler)]) == 1
        with operation("command"):
            emit_event(logger, "application.started", result="ready")
        with logging_session(role="worker", console_format="json") as nested:
            assert nested == first
        emit_event(logger, "application.finished", result="success")
    rows = records(capsys)
    assert len(rows) == 2
    assert all(r["process_role"] == "bot" and r["run_id"] == first for r in rows)
    assert rows[0]["operation_id"] and rows[1]["operation_id"] is None
    assert tuple(logging.getLogger().handlers) == handlers
    assert logging.getLogRecordFactory() is factory
    with logging_session(console_format="json") as second:
        assert second != first


async def test_parallel_context_threads_cancel_and_job_attempts(capsys):
    job_id = uuid4()
    with logging_session(console_format="json"):
        with operation("command") as parent:

            async def child():
                with operation("job_attempt", job_id=job_id) as context:
                    await asyncio.sleep(0)
                    assert await asyncio.to_thread(current_context) == context
                    await asyncio.to_thread(
                        emit_event, logger, "operation.finished", result="success"
                    )
                    return context["operation_id"]

            ids = await asyncio.gather(child(), child())
            assert len(set(ids)) == 2
            assert current_context() == parent
            entered = asyncio.Event()

            async def cancelled():
                try:
                    with operation("analysis", emit=True):
                        entered.set()
                        await asyncio.Event().wait()
                finally:
                    assert current_context() == parent

            task = asyncio.create_task(cancelled())
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert current_context() == parent
        assert current_context() == {}
    rows = records(capsys)
    assert [r["job_id"] for r in rows[:2]] == [str(job_id)] * 2
    assert all(r["parent_operation_id"] == parent["operation_id"] for r in rows)
    assert rows[-1]["result"] == "cancelled"
    assert rows[-1]["duration_ms"] >= 0


@pytest.mark.parametrize("level", ["INFO", "DEBUG"])
def test_nested_errors_allowlist_and_bounded_utf8(capsys, level):
    class Input(BaseModel):
        years: int

    try:
        Input(years=SECRET)
    except ValidationError as validation:
        errors = [validation]
    errors.extend(
        [
            json.JSONDecodeError(SECRET, SECRET, 1),
            StatementError(SECRET, "SELECT " + SECRET, {"resume": SECRET}, ValueError(SECRET)),
            httpx.HTTPStatusError(
                SECRET,
                request=httpx.Request("GET", "https://host/?token=" + SECRET.encode().hex()),
                response=httpx.Response(403, text=SECRET),
            ),
            requests.RequestException(
                SECRET, request=requests.Request("POST", "https://" + SECRET)
            ),
        ]
    )
    group = ExceptionGroup(SECRET, errors)
    group.__cause__ = RuntimeError(SECRET)
    group.__cause__.__cause__ = group  # Cyclic chains stay bounded.
    with logging_session(level=level, console_format="json"):
        with operation("command"):
            emit_event(
                logger,
                "application.failed",
                level=logging.ERROR,
                error=group,
                code="unexpected_error",
                result="failed",
                query=SECRET * 10000,
                resume=SECRET,
                user_id=999,
                source_vacancy_id="123456",
                duration_ms=1.5,
            )
            for error in errors:
                logger.error(
                    "raw %s", SECRET, exc_info=(type(error), error, None), extra={"resume": SECRET}
                )
    rows = records(capsys)
    assert len(rows) == len(errors) + 1
    assert rows[0]["source_vacancy_id"] == "123456"
    assert rows[0]["error"]["type"] == "ExceptionGroup"
    assert len(rows[0]["error"]["chain"]) <= 8
    for row in rows:
        serialized = json.dumps(row, ensure_ascii=False)
        assert "CONTROL_SECRET" not in serialized
        assert "FORGED_EVENT" not in serialized
        assert len(serialized.encode("utf-8")) + 1 <= MAX_RECORD_BYTES
        assert datetime.fromisoformat(row["timestamp"]).utcoffset().total_seconds() == 0
        assert UUID(row["run_id"]) and UUID(row["operation_id"])
        assert row["v"] == 1 and "duration_ms" in row and "result" in row
        assert "resume" not in row and "user_id" not in row


def test_never_stringifies_arbitrary_messages_or_extras(capsys):
    class Unsafe:
        def __str__(self):
            raise AssertionError("Raw object was formatted")

        __repr__ = __str__

    with logging_session(console_format="json", level="DEBUG"):
        logger.warning(Unsafe(), Unsafe(), extra={"raw": Unsafe(), "event_fields": {"code": []}})
        emit_event(
            logger, "unknown", count=True, source_vacancy_id=SECRET, duration_ms=float("nan")
        )
    rows = records(capsys)
    assert [r["event"] for r in rows] == ["logging.unstructured"] * 2
    assert "code" not in rows[0] and "count" not in rows[1]


def test_uuid_subclasses_are_normalized_without_custom_string_conversion(capsys):
    class DriverUUID(UUID):
        def __str__(self):
            raise AssertionError("Subclass must not format raw data")

    value = DriverUUID(int=123)
    with logging_session(console_format="json"):
        emit_event(logger, "job.claimed", job_id=value, vacancy_id=value)
    row = records(capsys)[0]
    assert row["job_id"] == row["vacancy_id"] == str(UUID(int=123))


def test_debug_has_only_safe_frame_locations(capsys):
    namespace = {}
    exec(
        compile("def fail():\n raise ValueError('CONTROL_SECRET')", "C:/private/cli.py", "exec"),
        namespace,
    )
    with logging_session(console_format="json", level="DEBUG"):
        try:
            namespace["fail"]()
        except ValueError as error:
            emit_event(logger, "application.failed", error=error, level=logging.ERROR)
    row = records(capsys)[0]
    assert row["error"]["frames"] == [{"file": "cli.py", "line": 2}]
    assert "private" not in json.dumps(row) and "CONTROL_SECRET" not in json.dumps(row)


def test_vendor_debug_and_late_sql_echo_never_write_stdout(capsys):
    with logging_session(console_format="json", level="DEBUG"):
        for name in ("aiogram", "httpx", "httpcore", "requests", "sqlalchemy.engine"):
            vendor = logging.getLogger(name)
            previous = vendor.level
            try:
                vendor.setLevel(logging.DEBUG)  # Even a library re-enabling itself is guarded.
                vendor.error(SECRET, extra={"body": SECRET})
            finally:
                vendor.setLevel(previous)
        engine = create_engine("sqlite://", echo=True)
        try:
            with engine.connect() as connection:
                assert connection.scalar(text("SELECT :value"), {"value": SECRET}) == SECRET
        finally:
            engine.dispose()
        emit_event(logger, "application.finished", result="success")
    assert len(records(capsys)) == 1


def test_existing_root_handlers_receive_sanitized_records(capsys):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        with logging_session(console_format="json"):
            with operation("command"):
                emit_event(logger, "operation.finished", count=2, result="success")
                logger.error(SECRET, extra={"private": SECRET})
        rows = records(capsys)
        assert rows[0]["count"] == 2  # Multiple handler filters preserve safe fields.
        assert rows[0]["operation_id"]
        assert "CONTROL_SECRET" not in stream.getvalue()
        assert handler.filters == []
    finally:
        root.removeHandler(handler)


def test_broken_console_does_not_raise(monkeypatch):
    class Broken:
        def write(self, value):
            raise OSError(SECRET)

        def flush(self):
            raise OSError(SECRET)

    monkeypatch.setattr(sys, "stderr", Broken())
    with logging_session():
        emit_event(logger, "application.finished")


@pytest.mark.parametrize(
    "command,role", [("bot", "bot"), ("collect", "collect"), ("worker", "worker"), ("rank", "cli")]
)
def test_cli_bootstrap_and_json_stdout(monkeypatch, capsys, command, role):
    from jobs import cli

    async def fake_run(args):
        assert current_context()["operation_id"]
        await asyncio.to_thread(logger.warning, SECRET)
        print(json.dumps({"result": "тест"}, ensure_ascii=False))

    arguments = ["--profile", "synthetic.yml", "--json"] if command == "rank" else []
    monkeypatch.setattr(sys, "argv", ["vacancy-analyzer", command, *arguments])
    monkeypatch.setattr(
        cli, "LoggingOptions", lambda: SimpleNamespace(level="INFO", console_format="json")
    )
    monkeypatch.setattr(cli, "run_command", fake_run)
    monkeypatch.setattr(cli, "run_telegram", fake_run)
    cli.main()
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"result": "тест"}
    rows = [json.loads(line) for line in captured.err.splitlines()]
    assert rows[0]["event"] == "application.started"
    assert rows[-1]["event"] == "application.finished"
    assert all(r["process_role"] == role and r["operation_id"] for r in rows)
    assert "CONTROL_SECRET" not in captured.err
    assert current_context() == {}


@pytest.mark.parametrize(
    "error,exit_code",
    [
        (ValueError(SECRET), 1),
        (RuntimeError(SECRET), 1),
        (KeyboardInterrupt(), 130),
        (SystemExit(SECRET), 1),
    ],
)
def test_cli_errors_and_ctrl_c(monkeypatch, capsys, error, exit_code):
    from jobs import cli

    async def fail(args):
        raise error

    monkeypatch.setattr(sys, "argv", ["vacancy-analyzer", "rank", "--profile", "synthetic.yml"])
    monkeypatch.setattr(cli, "run_command", fail)
    monkeypatch.setattr(
        cli, "LoggingOptions", lambda: SimpleNamespace(level="INFO", console_format="json")
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == exit_code
    rows = records(capsys)
    assert rows[-1]["result"] == ("cancelled" if exit_code == 130 else "failed")
    assert "CONTROL_SECRET" not in json.dumps(rows)


def test_argument_error_does_not_print_rejected_input(capsys):
    from jobs.cli import parser

    with pytest.raises(SystemExit) as caught:
        parser().parse_args([SECRET])
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "--help" in captured.err
    assert "CONTROL_SECRET" not in captured.err and "FORGED_EVENT" not in captured.err


def test_text_console_is_utf8_and_keeps_short_operation(capsys):
    with logging_session():
        with operation("command") as context:
            emit_event(logger, "application.started")
    captured = capsys.readouterr()
    assert "Приложение запущено" in captured.err
    assert "оп. " + context["operation_id"][:8] in captured.err
    assert context["operation_id"] not in captured.err
    assert "{" not in captured.err
    assert not captured.out


@pytest.mark.parametrize(
    "code,status", [("hh_http_403", 403), ("hh_http_429", 429), ("hh_captcha", None)]
)
async def test_collection_stop_explains_safe_reason(monkeypatch, capsys, code, status):
    from collectors.errors import CollectionStopped
    from jobs import collection_runtime

    error = CollectionStopped(SECRET, code=code, status_code=status)
    error.stage, error.source_vacancy_id = "catalog_check", "314"

    async def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(collection_runtime, "run_collection", fail)
    with logging_session(role="bot", console_format="json"):
        await collection_runtime._background_worker(SimpleNamespace(), None)
    row = records(capsys)[0]
    assert row["event"] == "collection.stopped" and row["code"] == code
    assert row["stage"] == "catalog_check" and row["source_vacancy_id"] == "314"
    assert row.get("status_code") == status and row["result"] == "stopped"
    assert "CONTROL_SECRET" not in json.dumps(row)


def test_old_task_context_does_not_enter_new_run(capsys):
    with operation("command"):
        with logging_session(console_format="json"):
            assert current_context() == {}
            emit_event(logger, "application.started")
    assert records(capsys)[0]["operation_id"] is None


def test_bad_logger_settings_do_not_print_validation_input(monkeypatch, capsys):
    from jobs import cli

    monkeypatch.setattr(sys, "argv", ["vacancy-analyzer", "worker"])
    monkeypatch.setenv("LOG_LEVEL", SECRET)
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 1
    captured = capsys.readouterr()
    assert not captured.out and "LOG_LEVEL" in captured.err
    assert "CONTROL_SECRET" not in captured.err


def test_argument_errors_follow_json_console_setting(monkeypatch, capsys):
    from jobs.cli import parser

    monkeypatch.setenv("LOG_CONSOLE_FORMAT", "json")
    with pytest.raises(SystemExit):
        parser().parse_args([SECRET])
    assert records(capsys)[0]["code"] == "invalid_arguments"


@pytest.fixture(autouse=True)
def restore_logger():
    yield
    shutdown_logging()
