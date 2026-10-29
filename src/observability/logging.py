"""Allowlisted events, contextvars, console and bounded file output. No raw messages."""

import asyncio
import json
import logging
import math
import os
import re
import sys
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from observability.console import render_text
from observability.events import (
    PIPELINE_CODES,
    PIPELINE_EVENTS,
    RECOMMENDATION_CODES,
    RECOMMENDATION_EVENTS,
)
from observability.files import FileJournal
from observability.repeats import RepeatLimiter

MAX_RECORD_BYTES = 4096
EVENTS = {
    "serving.command_finished": "Команда обработана",
    "serving.loop_health": "Задержка цикла событий",
    "delivery.http_finished": "Сетевой запрос Telegram завершён",
    "application.started": "Приложение запущено",
    "application.finished": "Приложение завершено",
    "application.failed": "Операция завершилась ошибкой",
    "application.cancelled": "Работа остановлена",
    "operation.finished": "Операция завершена",
    "collection.plan_changed": "План поиска обновлён",
    "collection.no_profiles": "Нет готовых профилей для сбора. Заполните анкету через /start.",
    "collection.direction_started": "Поиск HH начат",
    "collection.direction_finished": "Обработка направления HH завершена",
    "collection.network_deferred": "Сбой HTTP при сборе; повтор после заданного интервала",
    "collection.already_running": "Сборщик уже работает; повторный запуск пропущен",
    "collection.stopped": "Фоновый сбор HH остановлен. Приложение продолжает работать с базой.",
    "collection.card_skipped": "Карточка HH пропущена",
    "analysis.deferred": "Задание обработки отложено",
    "analysis.startup_deferred": "Проверка каталога отложена",
    "analysis.unavailable": "Исполнитель ожидает восстановления",
    "experiment.progress": "Проверка данных продолжается",
    "experiment.indexing": "Подготовка представлений",
    "logging.unstructured": "Служебное событие без структурированного контекста",
    "logging.truncated": "Служебное событие сокращено",
    "logging.file_degraded": "Файловый журнал работает с потерями",
    "logging.file_recovered": "Файловая запись восстановлена",
    "logging.repeats": "Сводка повторяющихся предупреждений",
    "runtime.unhandled": "Необработанная ошибка фоновой задачи",
}
HINTS = {
    "log_io_error": "Ошибка файлового журнала; консоль работает. Записи могут быть потеряны.",
    "log_disk_full": "На диске нет места для журнала; консоль работает.",
    "log_permission_denied": "Нет прав на запись журнала; консоль работает.",
    "log_unsafe_path": "Небезопасный путь журнала или ссылка: файловая запись пропущена.",
    "log_directory_limit": "В каталоге журнала слишком много файлов; файловая запись пропущена.",
    "log_lock_busy": "Не удалось получить блокировку файлового журнала; запись пропущена.",
    "log_budget_exceeded": "Общий лимит журнала исчерпан активными файлами; запись пропущена.",
    "log_rotation_failed": "Ошибка ротации журнала; консоль работает.",
    "log_queue_full": "Очередь файлового журнала заполнена; часть записей потеряна.",
    "log_record_invalid": "Запись журнала превышает допустимый размер или имеет неверный тип.",
    "log_shutdown_timeout": (
        "Время сброса журнала истекло; сохранение всех записей не подтверждено."
    ),
    "log_records_lost": "Часть записей файлового журнала потеряна; проверьте диагностику консоли.",
    "log_recovered": "Файловая запись восстановлена; ранее потерянные записи не восстановлены.",
    "database_unavailable": "Ошибка БД. Проверьте DATABASE_URL, PostgreSQL и alembic upgrade head.",
    "invalid_token": "Неверный формат TELEGRAM_BOT_TOKEN. Проверьте локальный .env.",
    "invalid_configuration": "Ошибка конфигурации. Проверьте параметры команды и локальный .env.",
    "invalid_arguments": "Неверные аргументы команды. Используйте --help.",
    "collection_stopped": (
        "Сбор HH остановлен. Проверьте доступ и сохранённые страницы; затем перезапустите сбор."
    ),
    "collection_cancelled": "Сбор HH остановлен приложением.",
    "hh_http_403": "HH вернул HTTP 403: доступ запрещён. Фоновый сбор остановлен.",
    "hh_http_429": "HH вернул HTTP 429: ограничение частоты запросов. Фоновый сбор остановлен.",
    "hh_captcha": "HH запросил капчу. Фоновый сбор остановлен.",
    "hh_access_restricted": "HH сообщил об ограничении доступа в данных страницы. Сбор остановлен.",
    "hh_unexpected_search": "Не распознана поисковая страница HH; HTML сохранён. Сбор остановлен.",
    "hh_parse_limit": "Три карточки HH подряд не разобраны; HTML сохранён. Сбор остановлен.",
    "invalid_html": "Ошибка разбора HTML. Проверьте сохранённую страницу.",
    "already_running": "Сборщик уже работает для этой базы. Остановите его перед ручным сбором.",
    "network_error": "Не удалось связаться с внешним сервисом. Проверьте сеть и доступ.",
    "http_error": "Внешний сервис вернул ошибку HTTP. Проверьте доступ и настройки.",
    "telegram_error": "Ошибка Telegram. Проверьте доступ и настройки бота.",
    "unexpected_error": "Неожиданная ошибка приложения. Найдите операцию в журнале.",
    "access_blocked": "Доступ HH ограничен; проверьте сохранённую страницу.",
    "parse_error": "Не удалось разобрать карточку HH; исходная страница сохранена.",
}
EVENTS.update(PIPELINE_EVENTS)
HINTS.update(PIPELINE_CODES)
EVENTS.update(RECOMMENDATION_EVENTS)
HINTS.update(RECOMMENDATION_CODES)
ROLES = {"cli", "bot", "collect", "worker", "semantic_worker", "web"}
KINDS = {
    "command",
    "bot",
    "collection",
    "analysis",
    "job_attempt",
    "direction",
    "message",
    "operation",
    "catalog_check",
    "catalog_save",
    "matching",
    "delivery",
    "feedback",
    "history",
    "resume",
}
RESULTS = {
    "success",
    "failed",
    "cancelled",
    "deferred",
    "skipped",
    "ready",
    "stopped",
    "superseded",
    "unknown",
    "closed",
    "expired",
    "sent",
    "uncertain",
    "rejected",
}
APP_ROOTS = {"jobs", "bot", "collectors", "nlp", "scoring", "storage", "observability"}
THIRD_PARTY = (
    "aiogram",
    "httpx",
    "httpcore",
    "sqlalchemy",
    "urllib3",
    "requests",
    "asyncpg",
    "aiohttp",
    "huggingface_hub",
    "onnxruntime",
    "tokenizers",
    "py.warnings",
)
ERROR_TYPES = {
    "ValueError",
    "TypeError",
    "KeyError",
    "AttributeError",
    "RuntimeError",
    "OSError",
    "PermissionError",
    "FileNotFoundError",
    "ConnectionError",
    "ConnectionResetError",
    "ConnectionRefusedError",
    "TimeoutError",
    "CancelledError",
    "KeyboardInterrupt",
    "ValidationError",
    "JSONDecodeError",
    "YAMLError",
    "ScannerError",
    "ParserError",
    "SQLAlchemyError",
    "OperationalError",
    "DBAPIError",
    "IntegrityError",
    "PostgresError",
    "ConnectionDoesNotExistError",
    "HTTPError",
    "RequestException",
    "RequestError",
    "HTTPStatusError",
    "ConnectError",
    "ReadError",
    "ReadTimeout",
    "ConnectTimeout",
    "TelegramAPIError",
    "TelegramBadRequest",
    "TelegramForbiddenError",
    "TelegramNetworkError",
    "TokenValidationError",
    "CollectionStopped",
    "VacancyParseError",
    "CollectionAlreadyRunning",
    "OllamaUnavailable",
    "VacancyAnalysisError",
    "StaleResumeWork",
    "ExceptionGroup",
    "BaseExceptionGroup",
    "OtherError",
}
SAFE_FILES = {
    "cli.py",
    "analysis_worker.py",
    "collection_runtime.py",
    "collection_scheduler.py",
    "collect.py",
    "resumes.py",
    "notify.py",
    "ollama.py",
    "ollama_resume.py",
    "runtime.py",
    "common.py",
    "semantic_worker.py",
    "logging.py",
}
_context = ContextVar("vacancy_logging_context", default=None)
_state = None
_lock = threading.RLock()


def _uuid(value):
    if isinstance(value, UUID):
        # asyncpg uses a UUID subclass. Do not call an arbitrary subclass's __str__.
        try:
            number = value.int
            return str(UUID(int=number)) if type(number) is int else None
        except Exception:
            return None
    if type(value) is str and len(value) <= 45:
        try:
            return str(UUID(value))
        except ValueError:
            pass
    return None


def _component(name):
    if type(name) is not str:
        return "application"
    if name.startswith(
        (
            "jobs.analysis",
            "jobs.model_telemetry",
            "jobs.resumes",
            "jobs.resume_analysis",
            "jobs.vacancy_analysis",
            "nlp.",
        )
    ):
        return "analysis"
    if name.startswith(("jobs.collect", "jobs.catalog_checks", "collectors.")):
        return "collector"
    if name.startswith("jobs.semantic") or name.startswith("jobs.retrieval_evaluation"):
        return "experiment"
    if name.startswith("scoring."):
        return "matching"
    if name.startswith(("jobs.rank", "jobs.candidates")):
        return "matching"
    if name.startswith("jobs.notify"):
        return "delivery"
    if name.startswith("bot."):
        return "bot"
    if name.startswith("storage."):
        return "database"
    if name == "asyncio":
        return "runtime"
    return "application"


def _fields(values):
    if type(values) is not dict:
        return {}
    result = {}
    for key in (
        "count",
        "total",
        "attempt",
        "exit_code",
        "status_code",
        "suppressed_count",
        "pending_count",
        "page",
        "page_budget",
        "new_count",
        "changed_count",
        "unchanged_count",
        "skipped_count",
        "revision_number",
        "max_attempts",
        "catalog_rows",
        "state_excluded",
        "required_sql_excluded",
        "history_feedback_excluded",
        "available_rows",
        "scanned_count",
        "required_excluded",
        "copy_excluded",
        "recheck_excluded",
        "retained_groups",
        "scored_count",
        "below_threshold_count",
        "selected_count",
        "sent_count",
        "failed_count",
        "uncertain_count",
        "top_k",
        "compact_count",
        "detail_count",
        "repair_count",
        "refill_rounds",
        "pool_limit",
        "repair_limit",
        "pool_budget_exhausted",
        "projection_budget_exhausted",
        "source_exhausted",
        "catalog_counts_sampled",
        "history_ids",
        "history_groups",
        "candidate_cache_hit",
    ):
        value = values.get(key)
        if type(value) is int and 0 <= value <= 10**9:
            if key != "status_code" or 100 <= value <= 599:
                result[key] = value
    for key in ("duration_ms", "wait_ms", "shortlist_ms", "detail_ms", "residual_ms", "repair_ms"):
        value = values.get(key)
        if type(value) in {int, float} and math.isfinite(value) and 0 <= value <= 10**12:
            result[key] = round(value, 3)
    for key, allowed in (
        ("result", RESULTS),
        ("code", HINTS.keys()),
        ("scope", {"all", "name"}),
        (
            "stage",
            {
                "benchmark",
                "review",
                "query",
                "catalog_check",
                "search",
                "model_ready",
                "model_chat",
                "input_prepare",
                "document_prepare",
                "model_load",
                "prompt_eval",
                "inference",
                "validation",
                "result_write",
                "cache_write",
                "slot_wait",
                "queue_wait",
                "blocked_probe",
            },
        ),
        ("operation_kind", KINDS),
        ("feedback_action", {"like", "dislike", "save", "unsave", "apply", "unapply", "reason"}),
        ("repeated_event", EVENTS.keys()),
        ("job_kind", {"vacancy_analysis", "resume_analysis", "vacancy_projection"}),
        ("job_state", {"pending", "running", "succeeded", "failed", "superseded", "cancelled"}),
        ("change_kind", {"new", "changed", "unchanged"}),
        ("snapshot_state", {"new", "reused"}),
        ("source_kind", {"html", "api"}),
        ("component_kind", {"bot", "collector", "analysis"}),
        ("availability", {"active", "unknown", "closed", "expired"}),
        (
            "validation_code",
            {
                "input_too_large",
                "unknown_evidence",
                "invalid_schema",
                "invalid_json",
                "invalid_response",
            },
        ),
    ):
        value = values.get(key)
        if type(value) is str and value in allowed:
            result[key] = value
    for key in (
        "operation_id",
        "parent_operation_id",
        "job_id",
        "vacancy_id",
        "related_vacancy_id",
    ):
        value = _uuid(values.get(key))
        if value:
            result[key] = value
    value = values.get("source_vacancy_id")
    if type(value) is str and value.isascii() and value.isdecimal() and 1 <= len(value) <= 20:
        result["source_vacancy_id"] = value
    for key in ("pipeline", "content_revision", "raw_revision", "duplicate_group"):
        value = values.get(key)
        if type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value):
            result[key] = value
    value = values.get("model")
    if type(value) is str:
        known = re.fullmatch(
            r"(?:qwen[23](?:\.[0-9])?|llama[23](?:\.[0-9])?|gemma[23]?|phi[34]|mistral|deepseek-r1)(?::(?:[0-9]{1,3}(?:\.[0-9])?[bm]|latest))?",
            value,
        )
        result["model"] = value if known or value in {"rules", "test-model", "other"} else "other"
    value = values.get("next_run_at")
    if type(value) is str and re.fullmatch(
        r"(?:20[0-9]{2}|2100)-[0-9]{2}-[0-9]{2}T[0-9:.]{8,15}\+00:00", value
    ):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            value = None
    if type(value) is datetime and 2000 <= value.year <= 2100:
        result["next_run_at"] = (
            value.replace(tzinfo=UTC).isoformat()
            if value.tzinfo is None
            else value.astimezone(UTC).isoformat()
        )
    return result


def safe_exception(exc, *, debug=False):
    if not isinstance(exc, BaseException):
        return None

    def kind(value):
        name = type(value).__name__
        return name if name in ERROR_TYPES else "OtherError"

    result = {"type": kind(exc)}
    pending = [exc]
    seen = set()
    chain = []
    while pending and len(chain) < 8:
        error = pending.pop(0)
        if id(error) in seen:
            continue
        seen.add(id(error))
        chain.append(kind(error))
        if isinstance(error, BaseExceptionGroup):
            pending.extend(error.exceptions[:3])
        cause = error.__cause__ or error.__context__
        if cause is not None:
            pending.append(cause)
    if len(chain) > 1:
        result["chain"] = chain
    if debug:
        frames = []
        trace = exc.__traceback__
        while trace is not None and len(frames) < 6:
            filename = Path(trace.tb_frame.f_code.co_filename).name
            if filename in SAFE_FILES:
                frames.append({"file": filename, "line": trace.tb_lineno})
            trace = trace.tb_next
        if frames:
            result["frames"] = frames
    return result


def current_context():
    value = _context.get()
    if value is None:
        return {}
    if value.get("run_id") != (_state.run_id if _state else None):
        return {}
    return {key: item for key, item in value.items() if key != "run_id"}


def _factory(*args, **kwargs):
    state = _state
    record = (state.previous_factory if state else logging.LogRecord)(*args, **kwargs)
    original = record.name
    known = type(original) is str and original.split(".")[0] in APP_ROOTS
    event = (
        record.msg
        if type(record.msg) is str and len(record.msg) <= 80 and record.msg in EVENTS
        else "logging.unstructured"
    )
    if original == "asyncio" and record.levelno >= logging.ERROR:
        known = True
        event = "runtime.unhandled"
    if original == "nlp.ollama" and record.levelno == logging.WARNING:
        event = "model.corrected"
    record._safe_event = event
    record._safe_component = _component(original)
    record._safe_context = current_context()
    record._safe_repeat = None
    record._safe_error = (
        safe_exception(record.exc_info[1], debug=bool(state and state.level == logging.DEBUG))
        if record.exc_info
        else None
    )
    record.msg = EVENTS[event]
    record.args = ()
    record.exc_info = record.exc_text = record.stack_info = None
    record.name = record._safe_component
    record.pathname = record.filename = record.module = record.funcName = ""
    record.threadName = "worker"
    record.processName = "application"
    if not known:
        record.levelno = -1  # Also blocks late vendor handlers, including SQL echo stdout.
    record.levelname = {10: "DEBUG", 20: "INFO", 30: "WARNING", 40: "ERROR", 50: "CRITICAL"}.get(
        record.levelno, "DEBUG"
    )
    return record


class SafeFilter(logging.Filter):
    def filter(self, record):
        if record.levelno < 0:
            return False
        # Extras are appended AFTER LogRecordFactory. Keep only typed, registered fields.
        fields = _fields(getattr(record, "event_fields", getattr(record, "_safe_fields", {})))
        context = _fields(getattr(record, "event_context", getattr(record, "_safe_context", {})))
        event = getattr(
            record, "event_code", getattr(record, "_safe_event", "logging.unstructured")
        )
        if type(event) is not str or event not in EVENTS:
            event = "logging.unstructured"
        error = getattr(record, "_safe_error", None) or getattr(record, "event_error", None)
        safe_error = None
        if type(error) is dict and type(error.get("type")) is str and error["type"] in ERROR_TYPES:
            safe_error = {"type": error["type"]}
            if type(error.get("chain")) is list:
                safe_error["chain"] = [
                    v for v in error["chain"][:8] if type(v) is str and v in ERROR_TYPES
                ]
            if _state and _state.level == logging.DEBUG and type(error.get("frames")) is list:
                safe_error["frames"] = [
                    {"file": f["file"], "line": f["line"]}
                    for f in error["frames"][:6]
                    if type(f) is dict
                    and type(f.get("file")) is str
                    and f["file"] in SAFE_FILES
                    and type(f.get("line")) is int
                    and f["line"] > 0
                ]
        # Discard arbitrary extras before any managed root handler can inspect them.
        standard = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)
        for key in list(record.__dict__):
            if key not in standard and key not in {
                "_safe_component",
                "_safe_event",
                "_safe_context",
                "_safe_error",
                "_safe_repeat",
            }:
                del record.__dict__[key]
        record._safe_event = event
        record._safe_fields = fields
        record._safe_context = context
        record._safe_error = safe_error
        record.msg = EVENTS[event]
        record.args = ()
        record.name = getattr(record, "_safe_component", "application")
        record.pathname = record.filename = record.module = record.funcName = ""
        record.threadName = "worker"
        record.processName = "application"
        record.exc_info = record.exc_text = record.stack_info = None
        return True


class SafeFormatter(logging.Formatter):
    def __init__(self, style="text"):
        super().__init__()
        self.style = style

    def format(self, record):
        state = _state
        fields = getattr(record, "_safe_fields", {})
        message = HINTS.get(fields.get("code"), EVENTS[record._safe_event])
        payload = {
            "v": 1,
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "event": record._safe_event,
            "component": getattr(record, "_safe_component", "application"),
            "process_role": state.role if state else "cli",
            "run_id": state.run_id if state else None,
            "operation_id": None,
            "result": None,
            "duration_ms": None,
            "message": message,
        }
        payload.update(getattr(record, "_safe_context", {}))
        payload.update(fields)
        if getattr(record, "_safe_error", None):
            payload["error"] = record._safe_error
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if len(raw.encode("utf-8")) + 1 > MAX_RECORD_BYTES:
            payload.pop("error", None)
            payload["event"] = "logging.truncated"
            payload["message"] = EVENTS["logging.truncated"]
            raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if self.style == "json":
            return raw
        text = render_text(payload)
        if len(text.encode("utf-8")) + 1 > MAX_RECORD_BYTES:
            text = (
                text.encode("utf-8")[: MAX_RECORD_BYTES - 5].decode("utf-8", errors="ignore")
                + "..."
            )
        return text


@dataclass
class _State:
    pid: int
    run_id: str
    role: str
    level: int
    handler: object
    previous_factory: object
    previous_root_level: int
    previous_handlers: tuple
    previous_levels: dict
    filter: object
    limiter: object
    journal: object = None
    file_handler: object = None


class ConsoleHandler(logging.StreamHandler):
    def handleError(self, record):
        # A broken stderr must not print raw diagnostics or fail a recommendation.
        pass


class FileHandler(logging.Handler):
    def __init__(self, journal):
        super().__init__()
        self.journal = journal

    def emit(self, record):
        # No path access or disk I/O in an event-loop thread.
        data = (self.format(record) + "\n").encode("utf-8")
        self.journal.enqueue(data)


def configure_logging(*, role="cli", level="INFO", console_format="text", file_options=None):
    global _state
    if (
        type(level) is not str
        or type(console_format) is not str
        or level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        or console_format
        not in {
            "text",
            "json",
        }
    ):
        raise ValueError("Неверные настройки журнала")
    with _lock:
        if _state and _state.pid != os.getpid():
            shutdown_logging()
        if _state and _state.pid == os.getpid():
            _state.level = getattr(logging, level)
            _state.handler.setFormatter(SafeFormatter(console_format))
            logging.getLogger().setLevel(_state.level)
            return _state.run_id
        root = logging.getLogger()
        handler = ConsoleHandler(sys.stderr)
        guard = SafeFilter()
        limiter = RepeatLimiter(
            getattr(file_options, "repeat_interval", 60), getattr(file_options, "repeat_keys", 128)
        )
        previous_levels = {name: logging.getLogger(name).level for name in THIRD_PARTY}
        state = _State(
            os.getpid(),
            str(uuid4()),
            role if type(role) is str and role in ROLES else "cli",
            getattr(logging, level),
            handler,
            logging.getLogRecordFactory(),
            root.level,
            tuple(root.handlers),
            previous_levels,
            guard,
            limiter,
        )
        _state = state
        logging.setLogRecordFactory(_factory)
        for existing in root.handlers:
            existing.filters.insert(0, guard)
            existing.filters.insert(1, limiter)
        handler.addFilter(guard)
        handler.addFilter(limiter)
        handler.setFormatter(SafeFormatter(console_format))
        root.addHandler(handler)
        root.setLevel(state.level)
        for name in THIRD_PARTY:
            logging.getLogger(name).setLevel(logging.CRITICAL + 1)
        if getattr(file_options, "file_enabled", False):

            def notice(code, count, pending_count=0):
                if _state is not state:
                    return
                if code == "log_recovered" and state.level > logging.INFO:
                    return
                record = logging.getLogger(__name__).makeRecord(
                    __name__,
                    logging.INFO if code == "log_recovered" else logging.WARNING,
                    "",
                    0,
                    "logging.file_recovered"
                    if code == "log_recovered"
                    else "logging.file_degraded",
                    (),
                    None,
                    extra={
                        "event_fields": {
                            "code": code,
                            "count": count,
                            "pending_count": pending_count,
                        }
                    },
                )
                # Direct console only: never put a disk failure back into the file queue.
                state.handler.handle(record)

            try:
                state.journal = FileJournal(file_options, state.role, state.run_id, notice)
            except (RuntimeError, OSError):
                notice("log_io_error", 0)
            else:
                state.file_handler = FileHandler(state.journal)
                state.file_handler.addFilter(guard)
                state.file_handler.addFilter(limiter)
                state.file_handler.setFormatter(SafeFormatter("json"))
                root.addHandler(state.file_handler)
        return state.run_id


def shutdown_logging():
    global _state
    with _lock:
        state = _state
        if state is None:
            return
        root = logging.getLogger()
        for summary in state.limiter.drain():
            emit_event(
                logging.getLogger(__name__), "logging.repeats", level=logging.WARNING, **summary
            )
        if state.file_handler:
            root.removeHandler(state.file_handler)
            state.file_handler.close()
            state.journal.close()
        root.removeHandler(state.handler)
        try:
            state.handler.flush()
        except (OSError, ValueError):
            pass
        state.handler.close()
        for handler in state.previous_handlers:
            handler.removeFilter(state.filter)
            handler.removeFilter(state.limiter)
        for name, level in state.previous_levels.items():
            logging.getLogger(name).setLevel(level)
        root.setLevel(state.previous_root_level)
        if logging.getLogRecordFactory() is _factory:
            logging.setLogRecordFactory(state.previous_factory)
        _state = None


@contextmanager
def logging_session(*, role="cli", level="INFO", console_format="text", file_options=None):
    owner = _state is None or _state.pid != os.getpid()
    if owner:
        configure_logging(
            role=role, level=level, console_format=console_format, file_options=file_options
        )
    try:
        yield _state.run_id
    finally:
        if owner:
            shutdown_logging()


def emit_event(logger, event, *, level=logging.INFO, error=None, _context=None, **fields):
    event = event if type(event) is str and event in EVENTS else "logging.unstructured"
    if type(level) is not int or level not in {10, 20, 30, 40, 50}:
        level = logging.INFO
    logger.log(
        level,
        event,
        extra={
            "event_code": event,
            "event_fields": _fields(fields),
            "event_context": current_context() if _context is None else _fields(_context),
            "event_error": safe_exception(
                error, debug=bool(_state and _state.level == logging.DEBUG)
            ),
        },
    )


@contextmanager
def bind_context(**fields):
    token = _context.set(
        current_context() | _fields(fields) | {"run_id": _state.run_id if _state else None}
    )
    try:
        yield
    finally:
        _context.reset(token)


@contextmanager
def operation(
    kind="operation", *, job_id=None, source_vacancy_id=None, parent_operation_id=None, emit=False
):
    previous = current_context()
    context = {
        **previous,
        "operation_id": str(uuid4()),
        "parent_operation_id": _uuid(parent_operation_id) or previous.get("operation_id"),
    }
    context["operation_kind"] = kind if type(kind) is str and kind in KINDS else "operation"
    context.update(_fields({"job_id": job_id, "source_vacancy_id": source_vacancy_id}))
    token = _context.set(context | {"run_id": _state.run_id if _state else None})
    started = time.monotonic()
    try:
        yield context.copy()
    except BaseException as exc:
        if emit:
            emit_event(
                logging.getLogger(__name__),
                "operation.finished",
                level=logging.WARNING
                if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt))
                else logging.ERROR,
                result="cancelled"
                if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt))
                else "failed",
                error=exc,
                duration_ms=(time.monotonic() - started) * 1000,
            )
        raise
    else:
        if emit:
            emit_event(
                logging.getLogger(__name__),
                "operation.finished",
                result="success",
                duration_ms=(time.monotonic() - started) * 1000,
            )
    finally:
        _context.reset(token)
