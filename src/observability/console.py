"""Compact console presentation of an already sanitized event. JSON stays unchanged."""

from datetime import datetime

COMPONENTS = {
    "collector": "Сбор",
    "analysis": "Анализ",
    "bot": "Бот",
    "matching": "Подбор",
    "delivery": "Доставка",
    "database": "БД",
    "experiment": "Проверка",
    "runtime": "Приложение",
    "application": "Приложение",
}
SOURCE_LABELS = {
    "hh": "HH",
    "html": "HH",
    "api": "HH",
    "habr_career": "Хабр Карьера",
    "trudvsem": "Работа России",
    "superjob": "SuperJob",
    "avito": "Авито Работа",
}
RESULTS = {
    "success": "успешно",
    "failed": "ошибка",
    "cancelled": "отменено",
    "superseded": "заменено новым",
    "deferred": "отложено",
    "skipped": "пропущено",
    "ready": "готово",
    "stopped": "остановлено",
    "unknown": "не определено",
    "closed": "закрыто",
    "expired": "срок истёк",
    "sent": "отправлено",
    "uncertain": "доставка не подтверждена",
    "rejected": "отклонено",
}
COUNTERS = {
    "catalog.batch_finished": (
        ("new_count", "новых"),
        ("changed_count", "обновлено"),
        ("unchanged_count", "без изменений"),
        ("skipped_count", "пропущено"),
    ),
    "selection.finished": (
        ("scanned_count", "проверено"),
        ("selected_count", "выбрано"),
        ("copy_excluded", "копий исключено"),
        ("below_threshold_count", "ниже порога"),
    ),
    "selection.scored": (("scanned_count", "проверено"), ("scored_count", "оценено")),
    "delivery.finished": (
        ("sent_count", "отправлено"),
        ("failed_count", "отклонено"),
        ("uncertain_count", "без подтверждения"),
    ),
}


def duration(value):
    return f"{value:.0f} мс" if value < 1000 else f"{value / 1000:.1f} с"


def render_text(payload):
    event = payload["event"]
    message = payload["message"].rstrip(".")
    if event.startswith("model.request_") and not payload.get("code"):
        kind = "резюме" if payload.get("job_kind") == "resume_analysis" else "вакансии"
        if payload.get("stage") == "model_ready":
            message = {
                "model.request_started": "Проверка доступности модели",
                "model.request_finished": "Проверка доступности модели завершена",
            }.get(event, message)
        elif payload.get("stage") == "model_chat":
            message = {
                "model.request_started": f"Начат разбор {kind}",
                "model.request_finished": f"Ответ модели для {kind} получен",
            }.get(event, message)
    details = []
    source = SOURCE_LABELS.get(payload.get("source_kind"))
    if event.startswith("model.") and payload.get("model"):
        details.append(f"модель {payload['model']}")
    if payload.get("source_vacancy_id"):
        details.append(f"{source or 'ID'} {payload['source_vacancy_id']}")
    if "status_code" in payload:
        details.append(f"HTTP {payload['status_code']}")
    if payload.get("scope"):
        details.append(
            "поиск: название и описание" if payload["scope"] == "all" else "поиск: название"
        )
    if payload.get("stage") == "catalog_check":
        details.append("проверка актуальности")
    if "page" in payload:
        page = str(payload["page"])
        if "page_budget" in payload:
            page += f"/{payload['page_budget']}"
        details.append(f"страница {page}")
    if "attempt" in payload:
        attempt = str(payload["attempt"])
        if "max_attempts" in payload:
            attempt += f"/{payload['max_attempts']}"
        details.append(f"попытка {attempt}")
    for key, label in COUNTERS.get(event, ()):
        if key in payload:
            details.append(f"{label}: {payload[key]}")
    if event not in COUNTERS and "count" in payload:
        label = {
            "logging.file_degraded": "потеряно строк",
            "collection.progress": "сохранено вакансий",
        }.get(event, "количество")
        details.append(f"{label}: {payload['count']}")
    if event == "collection.sweep_finished":
        if "total" in payload:
            details.append(f"по данным площадки: {payload['total']}")
        if "pending_count" in payload:
            details.append(f"карточек ждут загрузки: {payload['pending_count']}")
    elif payload.get("pending_count"):
        details.append(f"исход записи не подтверждён: {payload['pending_count']}")
    if payload.get("suppressed_count"):
        details.append(f"повторов: {payload['suppressed_count']}")
    if event in {
        "job.finished",
        "delivery.persisted",
        "delivery.transport_finished",
        "operation.finished",
        "application.finished",
    }:
        result = RESULTS.get(payload.get("result"))
        if result:
            details.append(result)
    if payload.get("next_run_at"):
        next_run = datetime.fromisoformat(payload["next_run_at"]).astimezone()
        label = "следующий обход после" if event == "collection.sweep_finished" else "повтор после"
        details.append(f"{label} {next_run:%d.%m %H:%M:%S}")
    if "wait_ms" in payload:
        details.append(f"ожидание {duration(payload['wait_ms'])}")
    if "duration_ms" in payload and payload["duration_ms"] is not None:
        details.append(duration(payload["duration_ms"]))
    if payload.get("operation_id"):
        details.append(f"оп. {payload['operation_id'][:8]}")
    if payload["level"] in {"WARNING", "ERROR", "CRITICAL"} and payload.get("code"):
        details.append(payload["code"])
    component = (
        "Очередь"
        if event.startswith("job.")
        else COMPONENTS.get(payload["component"], "Приложение")
    )
    if payload["component"] == "collector" and source:
        component = f"{component} {source}"
    timestamp = datetime.fromisoformat(payload["timestamp"]).astimezone()
    text = f"{timestamp:%d.%m %H:%M:%S} {payload['level']:<7} [{component}] {message}"
    if details:
        text += " | " + "; ".join(details)
    return text
