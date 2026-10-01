"""Readable console keeps useful outcomes; the file retains the complete contract."""

import json
import logging
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from observability import bind_context, emit_event, logging_session, operation
from observability.settings import LoggingOptions

logger = logging.getLogger("jobs.analysis_worker")


def test_compact_model_console_and_full_file_have_same_identity(tmp_path, capsys):
    settings = LoggingOptions(_env_file=None, directory=tmp_path / "logs")
    job = uuid4()
    pipeline = "9d79dfd8" + "1" * 56
    with logging_session(role="bot", file_options=settings) as run:
        with (
            operation("job_attempt", job_id=job) as context,
            bind_context(job_kind="vacancy_analysis", pipeline=pipeline, model="qwen3:4b"),
        ):
            emit_event(logger, "model.request_started", stage="model_chat", resume="PRIVATE_RESUME")
    captured = capsys.readouterr()
    assert not captured.out and len(captured.err.splitlines()) == 1
    assert "[Анализ] Начат разбор вакансии" in captured.err
    assert "модель qwen3:4b" in captured.err
    assert "оп. " + context["operation_id"][:8] in captured.err
    for value in ("{", str(job), run, pipeline, context["operation_id"], "PRIVATE_RESUME"):
        assert value not in captured.err
    rows = [
        json.loads(line)
        for p in settings.directory.glob("*.jsonl")
        for line in p.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row["job_id"] == str(job) and row["pipeline"] == pipeline
    assert row["run_id"] == run and row["operation_id"] == context["operation_id"]
    assert row["timestamp"].endswith("+00:00") and row["event"] == "model.request_started"
    assert "PRIVATE_RESUME" not in json.dumps(row)


@pytest.mark.parametrize(
    "kind,stage,expected",
    [
        ("vacancy_analysis", "model_chat", "Начат разбор вакансии"),
        ("resume_analysis", "model_chat", "Начат разбор резюме"),
        ("vacancy_analysis", "model_ready", "Проверка доступности модели"),
    ],
)
def test_local_console_time_and_model_stage(capsys, kind, stage, expected):
    stamp = datetime(2026, 10, 28, 9, 41, 44, tzinfo=UTC).timestamp()
    with logging_session():
        record = logger.makeRecord(
            logger.name,
            logging.INFO,
            "",
            0,
            "model.request_started",
            (),
            None,
            extra={"event_fields": {"job_kind": kind, "stage": stage}},
        )
        record.created = stamp
        logger.handle(record)
    line = capsys.readouterr().err
    assert line.startswith(datetime.fromtimestamp(stamp).astimezone().strftime("%d.%m %H:%M:%S"))
    assert expected in line and "+00:00" not in line and "{" not in line


def test_console_outcomes_counters_and_retry_are_readable(capsys):
    with logging_session():
        emit_event(logger, "job.finished", result="failed")
        emit_event(
            logger,
            "job.retry_scheduled",
            level=logging.WARNING,
            code="model_timeout",
            attempt=1,
            max_attempts=3,
            next_run_at=datetime(2026, 10, 28, 10, tzinfo=UTC),
        )
        emit_event(
            logger,
            "selection.finished",
            scanned_count=6,
            selected_count=1,
            copy_excluded=5,
            below_threshold_count=0,
            duration_ms=1250,
        )
    lines = capsys.readouterr().err.splitlines()
    assert "[Очередь]" in lines[0] and "ошибка" in lines[0]
    assert "попытка 1/3" in lines[1] and "повтор после" in lines[1] and "model_timeout" in lines[1]
    assert "проверено: 6" in lines[2] and "выбрано: 1" in lines[2]
    assert "копий исключено: 5" in lines[2] and "1.2 с" in lines[2]
    assert all("{" not in line for line in lines)
