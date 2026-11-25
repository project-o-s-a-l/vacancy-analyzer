"""acceptance: bounded synthetic data and isolated databases only."""

import asyncio
import errno
import json
import logging
import subprocess
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import httpx
import pytest
from pydantic import BaseModel, ValidationError
from sqlalchemy import select, update
from sqlalchemy.exc import StatementError
from test_pipeline_logging import Analyzer

from collectors.base import SearchQuery
from jobs.analysis_worker import AnalysisWorker
from jobs.collect import collect_vacancies
from jobs.notify import NotificationService
from jobs.users import UserService
from nlp.analysis_identity import pipeline_hash
from nlp.ollama import OllamaUnavailable
from observability import emit_event, logging_session, operation, shutdown_logging
from observability.files import FileJournal, Store
from observability.logging import MAX_RECORD_BYTES
from observability.settings import LoggingOptions
from storage.models import BackgroundJob
from storage.repositories import VacancyRepository
from storage.repositories.job_repository import JobRepository
from storage.repositories.user_repository import RegisteredUser, UserRepository

MARKERS = (
    "987654321:AA_TEST_PRIVATE_TOKEN_0123456789",
    "https://api.telegram.org/bot987654321:PRIVATE/sendMessage?chat_id=88776655",
    "postgresql+asyncpg://private_user:PRIVATE_PASSWORD@localhost/private_database",
    "private.person@example.invalid",
    "+79998887766",
    "Private_Resume_Иванов.docx",
    "PRIVATE_RESUME_SKILLS_AND_EXPERIENCE",
    "PRIVATE_PROMPT_AND_MODEL_RESPONSE",
    "FORGED_EVENT",
    '\n{"event":"FORGED_EVENT","result":"success"}\r\n',
)
SECRET = " | ".join(MARKERS)
logger = logging.getLogger("jobs.logging_acceptance")


def read_jsonl(directory):
    return [
        json.loads(line)
        for path in sorted(directory.glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def validate(rows):
    raw = json.dumps(rows, ensure_ascii=False)
    assert all(marker.strip() not in raw for marker in MARKERS)
    for row in rows:
        assert row["v"] == 1 and UUID(row["run_id"])
        assert datetime.fromisoformat(row["timestamp"]).utcoffset() == timedelta(0)
        assert row["level"] in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        assert (
            len(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode()) + 1
            <= MAX_RECORD_BYTES
        )
        assert (
            not {
                "user_id",
                "chat_id",
                "username",
                "resume",
                "prompt",
                "callback",
                "payload",
                "url",
                "dsn",
            }
            & row.keys()
        )


@pytest.fixture(autouse=True)
def isolated_worker(monkeypatch):
    monkeypatch.setattr("jobs.analysis_worker.INFERENCE_LOCK", uuid4().int % 2**63)
    yield
    shutdown_logging()


@pytest.mark.parametrize("level", ["INFO", "DEBUG"])
def test_secret_matrix_cannot_forge_console_or_file_events(tmp_path, capsys, level):
    class Input(BaseModel):
        years: int

    try:
        Input(years=SECRET)
    except ValidationError as exc:
        validation = exc
    errors = [
        validation,
        json.JSONDecodeError(SECRET, SECRET, 1),
        StatementError(SECRET, "SELECT " + SECRET, {"private": SECRET}, ValueError(SECRET)),
        httpx.HTTPStatusError(
            SECRET,
            request=httpx.Request("POST", MARKERS[1]),
            response=httpx.Response(403, text=SECRET),
        ),
    ]
    group = ExceptionGroup(SECRET, errors)
    group.__cause__ = RuntimeError(SECRET)
    group.__cause__.__cause__ = group
    options = LoggingOptions(_env_file=None, directory=tmp_path / "logs")
    with logging_session(level=level, console_format="json", file_options=options):
        with operation("command"):
            for exc in [*errors, group]:
                emit_event(
                    logger,
                    "application.failed",
                    level=logging.ERROR,
                    error=exc,
                    code="unexpected_error",
                    result="failed",
                    resume=SECRET,
                    prompt=SECRET,
                    model=SECRET,
                    source_vacancy_id=SECRET,
                    vacancy_id=SECRET,
                    pipeline=SECRET,
                    duplicate_group=SECRET,
                    next_run_at=SECRET,
                    feedback_action=SECRET,
                    callback=SECRET,
                    payload={"text": SECRET},
                    username=SECRET,
                )
            logger.error(
                SECRET,
                extra={"run_id": SECRET, "user_id": 88776655},
                exc_info=(type(group), group, None),
            )
            for name in ("aiogram", "httpx", "httpcore", "sqlalchemy.engine", "asyncpg", "aiohttp"):
                vendor = logging.getLogger(name)
                before = vendor.level
                try:
                    vendor.setLevel(logging.DEBUG)
                    vendor.error(SECRET)
                finally:
                    vendor.setLevel(before)
    captured = capsys.readouterr()
    assert not captured.out
    assert all(marker.strip() not in captured.err for marker in MARKERS)
    console = [json.loads(line) for line in captured.err.splitlines()]
    assert len(console) == 6  # Five structured errors and one unstructured application event.
    assert read_jsonl(options.directory) == console
    validate(console)


async def test_collect_retry_restart_lease_cancel_and_delivery(
    db_sessions, vacancy, profile, tmp_path, capsys
):
    sessions = db_sessions
    vacancy = vacancy.model_copy(
        update={
            "published_at": datetime.now(UTC),
            "valid_until": None,
            "source_id": "123456789",
            "raw_payload": {"private": SECRET},
        }
    )
    options = LoggingOptions(_env_file=None, directory=tmp_path / "logs")
    source = SimpleNamespace(search=AsyncMock(return_value=[vacancy]))
    query = SearchQuery(text=SECRET)
    analyzer = Analyzer(OllamaUnavailable(SECRET))
    with logging_session(role="collect", console_format="json", file_options=options) as first:
        with operation("command"):
            assert (
                await collect_vacancies(
                    source,
                    sessions,
                    query,
                    analysis_pipeline=pipeline_hash("vacancy_analysis", analyzer.model),
                )
                == 1
            )
            assert await AnalysisWorker(sessions, vacancy_analyzer=analyzer).run_once() == "retry"
            async with sessions.begin() as session:
                task = await session.scalar(select(BackgroundJob))
                job_id = task.id
                # Advance only synthetic test deadlines; do not sleep through retry cooldowns.
                task.next_run_at = datetime.now(UTC) - timedelta(seconds=1)
                task.lease_until = None
            async with sessions.begin() as session:
                crashed = await JobRepository(session).claim("synthetic-crashed-worker")
                assert crashed.id == job_id and crashed.attempts == 2
    async with sessions.begin() as session:
        await session.execute(
            update(BackgroundJob)
            .where(BackgroundJob.id == job_id)
            .values(lease_until=datetime.now(UTC) - timedelta(seconds=1))
        )
    analyzer.error = None
    with logging_session(role="bot", console_format="json", file_options=options) as second:
        with operation("command"):
            assert (
                await AnalysisWorker(sessions, vacancy_analyzer=analyzer).run_once() == "analyzed"
            )
            users = UserService(sessions, queued_resumes=True)
            await users.register(42)
            await users.resumes.submit(42, "Навыки: Python\n" + SECRET)
            await users.resumes.cancel(42)
            async with sessions.begin() as session:
                await UserRepository(session).save(RegisteredUser(42, profile, None))
            sender = SimpleNamespace(send_recommendation=AsyncMock(return_value=23))
            service = NotificationService(sessions, profile, chat_id=42, min_score=0)
            assert (await service.deliver(sender)).sent == 1
            assert (await service.deliver(sender)).sent == 0
            sender.send_recommendation.assert_awaited_once()
    source.search.assert_awaited_once_with(query)
    captured = capsys.readouterr()
    assert not captured.out
    rows = [json.loads(line) for line in captured.err.splitlines()]
    validate(rows)
    disk = read_jsonl(options.directory)
    assert sorted(disk, key=lambda r: r["timestamp"]) == sorted(rows, key=lambda r: r["timestamp"])
    assert first != second
    history = [r for r in rows if r.get("job_id") == str(job_id)]
    retry = next(r for r in history if r["event"] == "job.retry_scheduled")
    recovered = next(r for r in history if r["event"] == "job.lease_recovered")
    finished = next(r for r in history if r["event"] == "job.finished")
    assert retry["run_id"] == first and recovered["run_id"] == finished["run_id"] == second
    assert retry["attempt"] == 1 and recovered["attempt"] == finished["attempt"] == 3
    assert retry["operation_id"] != recovered["operation_id"] == finished["operation_id"]
    assert any(r["event"] == "job.finished" and r["result"] == "cancelled" for r in rows)
    assert any(r["event"] == "resume.cancelled" for r in rows)
    assert any(r["event"] == "delivery.persisted" and r["result"] == "sent" for r in rows)
    assert any(
        r["event"] == "selection.finished" and r.get("code") == "history_feedback_blocked"
        for r in rows
    )


async def test_disk_full_does_not_change_real_selection_or_delivery(
    sessions, vacancy, profile, tmp_path, monkeypatch, capsys
):
    vacancy.published_at = datetime.now(UTC)
    async with sessions.begin() as session:
        await VacancyRepository(session).save(vacancy)

    def full(*args):
        raise OSError(errno.ENOSPC, SECRET)

    monkeypatch.setattr(Store, "write", full)
    options = LoggingOptions(_env_file=None, directory=tmp_path / "logs")
    sender = SimpleNamespace(send_recommendation=AsyncMock(return_value=27))
    with logging_session(console_format="json", file_options=options):
        service = NotificationService(sessions, profile, chat_id=42, min_score=0)
        assert (await service.deliver(sender)).sent == 1
        assert await service.pending() == []
        assert await service.delivery_counts() == {"sent": 1}
    captured = capsys.readouterr()
    assert not captured.out
    rows = [json.loads(line) for line in captured.err.splitlines()]
    validate(rows)
    losses = [r for r in rows if r["event"] == "logging.file_degraded"]
    assert len(losses) == 2 and all(r["code"] == "log_disk_full" for r in losses)
    assert losses[-1]["count"] == len(rows) - len(losses)
    assert not read_jsonl(options.directory)


async def test_actual_rank_cli_json_with_console_and_file(
    sessions, vacancy, profile, tmp_path, monkeypatch, capsys
):
    import sys

    from jobs import cli
    from jobs.settings import Settings

    vacancy.published_at = datetime.now(UTC)
    async with sessions.begin() as session:
        await VacancyRepository(session).save(vacancy)
    options = LoggingOptions(_env_file=None, directory=tmp_path / "logs", console_format="json")
    monkeypatch.setattr(
        sys, "argv", ["vacancy-analyzer", "rank", "--profile", MARKERS[5], "--json"]
    )
    monkeypatch.setattr(cli, "LoggingOptions", lambda: options)
    monkeypatch.setattr(cli, "Settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(cli, "load_profile", lambda _: profile)
    monkeypatch.setattr(cli, "create_engine", lambda _: SimpleNamespace(dispose=AsyncMock()))
    monkeypatch.setattr(cli, "session_factory", lambda _: sessions)
    await asyncio.to_thread(cli.main)
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert len(result) == 1 and result[0]["vacancy"]["source_id"] == vacancy.source_id
    rows = [json.loads(line) for line in captured.err.splitlines()]
    validate(rows)
    assert read_jsonl(options.directory) == rows
    assert rows[0]["event"] == "application.started" and rows[-1]["event"] == "application.finished"
    assert any(r["event"] == "selection.scored" for r in rows)


def test_measurement_is_bounded_sequential_and_has_clean_json(tmp_path):
    script = Path(__file__).parents[1] / "scripts" / "measure_logging.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--directory",
            str(tmp_path / "measurement"),
            "--events",
            "8",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    report = json.loads(result.stdout)
    assert result.stderr == "" and report["complete"]
    assert report["max_events"] == 200 and report["queue_size"] == 256
    assert report["file_events"] == 8
    assert [r["mode"] for r in report["modes"]] == ["disabled", "console", "console_and_file"]
    assert [r["console_events"] for r in report["modes"]] == [0, 8, 8]


def test_shutdown_distinguishes_known_loss_and_unresolved_inflight(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def blocked(self, data):
        entered.set()
        assert release.wait(timeout=5)

    monkeypatch.setattr(Store, "write", blocked)
    settings = LoggingOptions(
        _env_file=None, directory=tmp_path / "logs", queue_size=1, shutdown_timeout=0.01
    )
    notices = []
    journal = FileJournal(settings, "cli", str(uuid4()), lambda *args: notices.append(args))
    try:
        journal.enqueue(b"{}\n")
        assert entered.wait(timeout=2)
        journal.enqueue(b"{}\n")
        journal.enqueue(b"{}\n")
        journal.close()
        assert notices[-1] == ("log_shutdown_timeout", 1, 2)
        assert journal.dropped == 1 and journal.thread.is_alive()
    finally:
        release.set()
        journal.thread.join(timeout=2)
    assert not journal.thread.is_alive()
