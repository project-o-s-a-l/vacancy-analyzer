"""Optional vacancy source adapters registered by installed distributions."""

from importlib.metadata import entry_points


class SourceAdapterError(ValueError):
    """A source was not registered unambiguously."""


def source_adapter(name):
    """Return the adapter registered for this collection mode."""
    matches = list(entry_points(group="vacancy_analyzer.sources", name=name))
    if not matches:
        raise SourceAdapterError(f"Для источника {name!r} требуется установленный пакет адаптеров.")
    if len(matches) > 1:
        raise SourceAdapterError(f"Для источника {name!r} установлено несколько адаптеров.")
    try:
        return matches[0].load()
    except ImportError as exc:
        raise SourceAdapterError(f"Не удалось загрузить адаптер источника {name!r}.") from exc
