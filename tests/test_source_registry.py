from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from collectors import registry


@pytest.mark.parametrize(
    ("entries", "message"), [([], "требуется"), ([object(), object()], "несколько")]
)
def test_registry_rejects_an_unconfigured_or_ambiguous_source(monkeypatch, entries, message):
    monkeypatch.setattr(registry, "entry_points", lambda **kwargs: entries)
    with pytest.raises(registry.SourceAdapterError, match=message):
        registry.source_adapter("example")


async def test_registered_source_yields_normalized_vacancies(monkeypatch, vacancy):
    @asynccontextmanager
    async def connect(settings):
        assert settings == "test-settings"
        yield SimpleNamespace(vacancy=vacancy)

    adapter = SimpleNamespace(open=connect)

    def installed(*, group, name):
        assert group == "vacancy_analyzer.sources" and name == "example"
        return [SimpleNamespace(load=lambda: adapter)]

    monkeypatch.setattr(registry, "entry_points", installed)
    async with registry.source_adapter("example").open("test-settings") as source:
        assert source.vacancy is vacancy
