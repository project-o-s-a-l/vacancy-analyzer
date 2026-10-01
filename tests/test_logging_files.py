"""Small synthetic journals only: limits, locks, faults and privacy, no external services."""

import errno
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from observability import emit_event, logging_session, operation, shutdown_logging
from observability.files import MAX_ENTRIES, FileJournal, StorageFailure, Store
from observability.logging import SafeFilter
from observability.repeats import RepeatLimiter
from observability.settings import LoggingOptions

logger = logging.getLogger("jobs.file_test")
SECRET = "CONTROL_SECRET+79991234567@example.com\nFORGED"


def options(tmp_path, **values):
    return LoggingOptions(
        _env_file=None,
        directory=tmp_path / "logs",
        max_file_bytes=4096,
        total_max_bytes=16384,
        queue_size=128,
        **values,
    )


def lines(root):
    return [
        json.loads(line)
        for path in root.glob("*.jsonl*")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_file_json_rotation_and_privacy(tmp_path, capsys):
    settings = options(tmp_path, backup_count=2)
    with logging_session(role="bot", level="DEBUG", console_format="json", file_options=settings):
        with operation("command"):
            for index in range(22):
                emit_event(
                    logger,
                    "experiment.progress",
                    count=index,
                    query=SECRET,
                    error=ValueError(SECRET),
                )
                logger.warning("raw %s", SECRET, extra={"resume": SECRET})
    paths = list(settings.directory.glob("*.jsonl*"))
    assert len(paths) == 3 and all(p.stat().st_size <= 4096 for p in paths)
    rows = lines(settings.directory)
    assert rows and all(r["v"] == 1 and r["process_role"] == "bot" for r in rows)
    assert all(r["operation_id"] and r["run_id"] for r in rows)
    assert "CONTROL_SECRET" not in "".join(p.read_text(encoding="utf-8") for p in paths)
    captured = capsys.readouterr()
    assert not captured.out and "CONTROL_SECRET" not in captured.err


def test_disabled_files_and_idempotent_setup(tmp_path):
    settings = options(tmp_path, file_enabled=False)
    with logging_session(file_options=settings), logging_session(file_options=settings):
        emit_event(logger, "application.started")
    assert not settings.directory.exists()


@pytest.mark.parametrize("command", ["bot", "collect", "worker", "rank"])
def test_normal_cli_roles_write_managed_files(tmp_path, monkeypatch, command, capsys):
    from jobs import cli

    settings = options(tmp_path)

    async def fake_run(args):
        print('{"ok": true}')

    arguments = ["--profile", "synthetic.yml", "--json"] if command == "rank" else []
    monkeypatch.setattr(sys, "argv", ["vacancy-analyzer", command, *arguments])
    monkeypatch.setattr(cli, "LoggingOptions", lambda: settings)
    monkeypatch.setattr(cli, "run_command", fake_run)
    monkeypatch.setattr(cli, "run_telegram", fake_run)
    cli.main()
    assert json.loads(capsys.readouterr().out) == {"ok": True}
    rows = lines(settings.directory)
    assert [r["event"] for r in rows] == ["application.started", "application.finished"]
    assert all(r["process_role"] == ("cli" if command == "rank" else command) for r in rows)


def test_ttl_budget_unknown_files_and_active_lease(tmp_path):
    settings = options(tmp_path, backup_count=0)
    old, active, current = [Store(settings, "cli", str(uuid4())) for _ in range(3)]
    try:
        old.write(b"{}\n" * 800)
        old.close()
        os.utime(old.path, (1, 1))
        active.write(b"{}\n" * 1000)
        os.utime(active.path, (1, 1))
        unknown = settings.directory / "keep.txt"
        unknown.write_text("untouched", encoding="utf-8")
        current.write(b"{}\n")
        assert not old.path.exists() and active.path.exists()
        assert unknown.read_text() == "untouched"
        assert list(settings.directory.glob("*.lease"))
    finally:
        old.close()
        active.close()
        current.close()


def test_active_files_exhaust_budget_then_recover(tmp_path):
    settings = LoggingOptions(
        _env_file=None,
        directory=tmp_path / "logs",
        max_file_bytes=4096,
        total_max_bytes=4096,
        backup_count=0,
    )
    first, second = [Store(settings, "bot", str(uuid4())) for _ in range(2)]
    try:
        first.write(b"x" * 4096)
        original = first.path.read_bytes()
        with pytest.raises(StorageFailure) as caught:
            second.write(b"{}\n")
        assert caught.value.code == "log_budget_exceeded"
        assert first.path.read_bytes() == original
        first.close()
        second.write(b"{}\n")
        assert not first.path.exists() and second.path.read_bytes() == b"{}\n"
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("backup_count", [0, 1, 3])
def test_small_rotation_exact_budget(tmp_path, backup_count):
    settings = options(tmp_path, backup_count=backup_count)
    store = Store(settings, "collect", str(uuid4()))
    try:
        for _ in range(10):
            store.write(b"{}\n" * 1000)
        paths = list(settings.directory.glob("*.jsonl*"))
        assert len(paths) == backup_count + 1
        assert sum(p.stat().st_size for p in paths) <= settings.total_max_bytes
        assert all(p.stat().st_size <= settings.max_file_bytes for p in paths)
    finally:
        store.close()


@pytest.mark.parametrize(
    "failure,code", [(errno.ENOSPC, "log_disk_full"), (errno.EACCES, "log_permission_denied")]
)
def test_disk_failure_console_survives_and_is_bounded(tmp_path, monkeypatch, capsys, failure, code):
    def fail(*args):
        raise OSError(failure, SECRET)

    monkeypatch.setattr(Store, "write", fail)
    with logging_session(console_format="json", file_options=options(tmp_path)):
        for _ in range(20):
            emit_event(logger, "experiment.progress")
    rows = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    diagnostics = [r for r in rows if r["event"] == "logging.file_degraded"]
    assert len(diagnostics) == 2  # Initial notice and bounded shutdown loss summary.
    assert all(r["code"] == code for r in diagnostics)
    assert diagnostics[-1]["count"] == 20
    assert "CONTROL_SECRET" not in json.dumps(rows)


def test_queue_backpressure_and_bounded_shutdown(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def slow(self, data):
        entered.set()
        release.wait(timeout=5)

    monkeypatch.setattr(Store, "write", slow)
    settings = LoggingOptions(
        _env_file=None, directory=tmp_path / "logs", queue_size=2, shutdown_timeout=0.05
    )
    notices = []
    journal = FileJournal(settings, "worker", str(uuid4()), lambda *args: notices.append(args))
    try:
        journal.enqueue(b"{}\n")
        assert entered.wait(timeout=2)
        started = time.monotonic()
        for _ in range(100):
            journal.enqueue(b"{}\n")
        assert time.monotonic() - started < 0.5
        assert journal.queue.qsize() == 2 and journal.dropped == 98
        started = time.monotonic()
        journal.close()
        assert time.monotonic() - started < 0.5
        assert notices[-1][0] == "log_shutdown_timeout"
    finally:
        release.set()
        journal.thread.join(timeout=2)
    assert not journal.thread.is_alive()


def test_unwritable_directory_is_safe_for_normal_host(tmp_path, capsys):
    settings = options(tmp_path)
    settings.directory.write_text(SECRET, encoding="utf-8")
    with logging_session(file_options=settings):
        emit_event(logger, "application.started")
    assert settings.directory.read_text(encoding="utf-8") == SECRET
    output = capsys.readouterr()
    assert "log_unsafe_path" in output.err and "CONTROL_SECRET" not in output.err


def test_rotation_failure_does_not_delete_unrelated_data(tmp_path, monkeypatch):
    settings = options(tmp_path)
    store = Store(settings, "cli", str(uuid4()))
    store.write(b"x" * 4096)
    unknown = settings.directory / "keep.jsonl"
    unknown.write_text(SECRET, encoding="utf-8")

    def fail(*args):
        raise PermissionError(SECRET)

    monkeypatch.setattr(Path, "replace", fail)
    try:
        with pytest.raises(StorageFailure) as caught:
            store.write(b"{}\n")
        assert caught.value.code == "log_rotation_failed"
        assert unknown.read_text(encoding="utf-8") == SECRET
        assert store.path.stat().st_size == 4096
    finally:
        store.close()


def test_no_symlink_or_hardlink_following(tmp_path):
    settings = options(tmp_path)
    store = Store(settings, "cli", str(uuid4()))
    store.prepare()
    outside = tmp_path / "outside.txt"
    outside.write_text(SECRET, encoding="utf-8")
    os.link(outside, store.path)
    try:
        with pytest.raises(StorageFailure):
            store.write(b"{}\n")
        assert outside.read_text(encoding="utf-8") == SECRET
    finally:
        store.close()
        store.path.unlink()
    try:
        store.path.symlink_to(outside)
    except OSError:
        pytest.skip("Symlinks unavailable")
    try:
        with pytest.raises(StorageFailure) as caught:
            store.write(b"{}\n")
        assert caught.value.code == "log_unsafe_path"
        assert outside.read_text(encoding="utf-8") == SECRET
    finally:
        store.close()
        store.path.unlink()


def test_managed_directory_scan_is_bounded(tmp_path, monkeypatch):
    from observability import files

    monkeypatch.setattr(files, "MAX_ENTRIES", 3)
    settings = options(tmp_path)
    settings.directory.mkdir()
    for index in range(4):
        (settings.directory / f"unknown-{index}").touch()
    store = Store(settings, "cli", str(uuid4()))
    try:
        with pytest.raises(StorageFailure) as caught:
            store.write(b"{}\n")
        assert caught.value.code == "log_directory_limit"
        assert MAX_ENTRIES == 4096
    finally:
        store.close()


def test_cleanup_uses_oldest_closed_groups_for_budget(tmp_path):
    settings = LoggingOptions(
        _env_file=None,
        directory=tmp_path / "logs",
        max_file_bytes=4096,
        total_max_bytes=8192,
        backup_count=0,
    )
    stores = [Store(settings, "cli", str(uuid4())) for _ in range(3)]
    try:
        for index, store in enumerate(stores[:2]):
            store.write(b"{}\n" * 1000)
            store.close()
            modified = time.time() - (100 - index)
            os.utime(store.path, (modified, modified))
        stores[2].write(b"{}\n" * 1000)
        assert not stores[0].path.exists()
        assert stores[1].path.exists() and stores[2].path.exists()
    finally:
        for store in stores:
            store.close()


def test_failed_disk_recovers_and_keeps_loss_count(tmp_path, monkeypatch):
    from observability import files

    settings = options(tmp_path)
    original = Store.write
    calls = []
    notices = []

    def temporary(self, data):
        calls.append(data)
        if len(calls) == 1:
            raise OSError(errno.ENOSPC, SECRET)
        return original(self, data)

    monkeypatch.setattr(Store, "write", temporary)
    monkeypatch.setattr(files, "RETRY_DELAY", 0)
    journal = FileJournal(settings, "cli", str(uuid4()), lambda *args: notices.append(args))
    journal.enqueue(b"{}\n")
    journal.enqueue(b"{}\n")
    journal.close()
    assert journal.dropped == 1
    assert [n[0] for n in notices] == ["log_disk_full", "log_recovered", "log_records_lost"]
    assert lines(settings.directory) == [{}]


def test_oversized_line_never_enters_queue(tmp_path):
    notices = []
    journal = FileJournal(
        options(tmp_path), "cli", str(uuid4()), lambda *args: notices.append(args)
    )
    try:
        journal.enqueue(b"x" * 4097)
        assert journal.queue.qsize() == 0 and journal.dropped == 1
        assert notices[0][0] == "log_record_invalid"
    finally:
        journal.close()


def test_idle_maintenance_deletes_expired_closed_run(tmp_path, monkeypatch):
    from observability import files

    settings = options(tmp_path)
    monkeypatch.setattr(files, "MAINTENANCE_INTERVAL", 0.01)
    journal = FileJournal(settings, "worker", str(uuid4()), lambda *args: None)
    old = Store(settings, "cli", str(uuid4()))
    try:
        journal.enqueue(b"{}\n")
        deadline = time.monotonic() + 2
        while journal.store.lease is None and time.monotonic() < deadline:
            time.sleep(0.005)
        old.write(b"{}\n")
        old.close()
        os.utime(old.path, (1, 1))
        deadline = time.monotonic() + 2
        while old.path.exists() and time.monotonic() < deadline:
            time.sleep(0.005)
        assert not old.path.exists()
    finally:
        old.close()
        journal.close()


def test_directory_link_and_root_paths_are_rejected(tmp_path):
    settings = options(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        settings.directory.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlinks unavailable")
    store = Store(settings, "cli", str(uuid4()))
    with pytest.raises(StorageFailure):
        store.write(b"{}\n")
    assert not list(outside.iterdir())
    settings.directory.unlink()
    unsafe = Store(LoggingOptions(_env_file=None, directory=Path.cwd()), "cli", str(uuid4()))
    with pytest.raises(StorageFailure):
        unsafe.write(b"{}\n")


def test_writer_start_failure_keeps_console(tmp_path, monkeypatch, capsys):
    from observability import logging as app_logging

    def fail(*args):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(app_logging, "FileJournal", fail)
    with logging_session(file_options=options(tmp_path)):
        emit_event(logger, "application.started")
    output = capsys.readouterr()
    assert "log_io_error" in output.err and "Приложение запущено" in output.err
    assert "CONTROL_SECRET" not in output.err


def test_repeat_keys_bounded_error_and_changed_reason_always_pass(tmp_path, capsys):
    clock = [0]
    limiter = RepeatLimiter(interval=10, max_keys=2, clock=lambda: clock[0])
    guard = SafeFilter()

    def record(code="network_error", level=logging.WARNING):
        value = logger.makeRecord(
            "jobs.file_test",
            level,
            "",
            0,
            "analysis.unavailable",
            (),
            None,
            extra={"event_fields": {"code": code}},
        )
        assert guard.filter(value)
        return value

    with logging_session(console_format="json", file_options=options(tmp_path)):
        assert limiter.filter(record())
        assert not limiter.filter(record())
        assert limiter.filter(record("database_unavailable"))
        assert limiter.filter(record(level=logging.ERROR))
        clock[0] = 11
        summary = record()
        assert limiter.filter(summary) and summary._safe_fields["suppressed_count"] == 1
        for code in ("http_error", "unexpected_error", "invalid_html"):
            assert limiter.filter(record(code))
        assert len(limiter.keys) == 2
        emit_event(logger, "analysis.unavailable", level=logging.WARNING)
        emit_event(logger, "analysis.unavailable", level=logging.WARNING)
        emit_event(logger, "analysis.unavailable", level=logging.ERROR)
    rows = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert len(rows) == 3 and rows[-1]["event"] == "logging.repeats"
    assert rows[-1]["suppressed_count"] == 1
    file_rows = lines(tmp_path / "logs")
    assert [r["event"] for r in file_rows] == [r["event"] for r in rows]


@pytest.mark.parametrize(
    "values",
    [
        {"max_file_bytes": 4095},
        {"total_max_bytes": 4096},
        {"max_file_bytes": 4096, "total_max_bytes": 4096, "backup_count": 1},
        {"queue_size": 0},
        {"retention_days": 0},
        {"shutdown_timeout": float("inf")},
    ],
)
def test_settings_reject_unbounded_or_conflicting_limits(values):
    with pytest.raises(ValidationError):
        LoggingOptions(_env_file=None, **values)


def test_two_live_processes_keep_separate_files_and_leases(tmp_path):
    code = """
import sys, time
from pathlib import Path
from uuid import uuid4
from observability.files import Store
from observability.settings import LoggingOptions
root, ready, release, role = map(Path, sys.argv[1:5])
settings = LoggingOptions(_env_file=None, directory=root, max_file_bytes=4096,
                          total_max_bytes=8192, backup_count=0)
store = Store(settings, str(role), str(uuid4()))
try:
    store.write(b'{}\\n' * 1000)
    ready.touch()
    deadline = time.monotonic() + 10
    while not release.exists() and time.monotonic() < deadline:
        time.sleep(.01)
finally:
    store.close()
"""
    root, release = tmp_path / "logs", tmp_path / "release"
    children = []
    try:
        for role in ("bot", "worker"):
            ready = tmp_path / (role + ".ready")
            child = subprocess.Popen(
                [sys.executable, "-c", code, str(root), str(ready), str(release), role],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            children.append(child)
            deadline = time.monotonic() + 8
            while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            assert ready.exists(), child.communicate(timeout=1)
        paths = list(root.glob("*.jsonl"))
        assert len(paths) == 2 and sum(p.stat().st_size for p in paths) == 6000
        settings = LoggingOptions(
            _env_file=None,
            directory=root,
            max_file_bytes=4096,
            total_max_bytes=8192,
            backup_count=0,
        )
        parent = Store(settings, "collect", str(uuid4()))
        try:
            with pytest.raises(StorageFailure) as caught:
                parent.write(b"{}\n" * 1000)
            assert caught.value.code == "log_budget_exceeded"
            assert all(p.exists() for p in paths)
        finally:
            parent.close()
    finally:
        release.touch()
        for child in children:
            try:
                _, stderr = child.communicate(timeout=3)
                assert child.returncode == 0, stderr
            except subprocess.TimeoutExpired:
                child.kill()
                child.communicate(timeout=2)


def test_concurrent_rotation_stays_within_shared_budget(tmp_path):
    code = """
import sys, time
from pathlib import Path
from uuid import uuid4
from observability.files import Store
from observability.settings import LoggingOptions
root, role = Path(sys.argv[1]), sys.argv[2]
settings = LoggingOptions(_env_file=None, directory=root, max_file_bytes=4096,
                          total_max_bytes=16384, backup_count=1)
store = Store(settings, role, str(uuid4()))
try:
    store.write(b'{}\\n' * 1000)
    (root.parent / (role + '.ready')).touch()
    deadline = time.monotonic() + 8
    while not all((root.parent / (r + '.ready')).exists() for r in ('bot','worker')):
        if time.monotonic() > deadline:
            raise RuntimeError('Test barrier timeout')
        time.sleep(.005)
    for _ in range(12):
        store.write(b'{}\\n' * 1000)
        time.sleep(.001)
finally:
    store.close()
"""
    root = tmp_path / "logs"
    children = []
    try:
        for role in ("bot", "worker"):
            children.append(
                subprocess.Popen(
                    [sys.executable, "-c", code, str(root), role],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            )
        for child in children:
            _, error = child.communicate(timeout=12)
            assert child.returncode == 0, error
        paths = list(root.glob("*.jsonl*"))
        assert len(paths) == 4 and sum(p.stat().st_size for p in paths) <= 16384
        assert all(p.stat().st_size <= 4096 for p in paths)
        assert len(lines(root)) == 4000
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.communicate(timeout=2)


@pytest.fixture(autouse=True)
def restore_logging():
    yield
    shutdown_logging()
