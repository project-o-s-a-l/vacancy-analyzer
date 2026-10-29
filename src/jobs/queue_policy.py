"""Immutable process configuration. Restart all producers/workers after a policy change."""

from functools import lru_cache

from jobs.settings import Settings


@lru_cache(maxsize=1)
def queue_settings():
    return Settings()


class QueueFull(ValueError):
    def __init__(self):
        super().__init__("Очередь обработки заполнена. Повторите загрузку позже; профиль сохранён.")


class SubmissionLimited(ValueError):
    def __init__(self):
        super().__init__("Слишком частые загрузки резюме. Подождите минуту перед новой загрузкой.")
