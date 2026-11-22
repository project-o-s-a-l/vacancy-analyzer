import os
from datetime import date
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from bot.cards import render_card
from domain import CandidateProfile
from jobs import exchange_rates
from jobs.exchange_rates import CBR_URL, RateCache
from jobs.notify import NotificationService
from jobs.profiles import update_field
from jobs.rank import rank_recent
from storage.repositories import VacancyRepository

QUOTES = b"""<ValCurs Date="20.11.2026">
<Valute><CharCode>EUR</CharCode><Nominal>1</Nominal><Value>100,00</Value></Valute>
<Valute><CharCode>USD</CharCode><Nominal>1</Nominal><Value>90,00</Value></Valute>
<Valute><CharCode>KZT</CharCode><Nominal>100</Nominal><Value>20,00</Value></Valute>
</ValCurs>"""


@pytest.mark.parametrize(
    "text,currency,gross",
    [
        ("от 100000 рублей", "RUB", True),
        ("100 000 RUB", "RUB", True),
        ("100\u202f000 € на руки", "EUR", False),
        ("100000 долларов до вычета налогов", "USD", True),
        ("от 100000", "USD", True),
        ("100000 net", "USD", False),
        ("100000 KZT gross", "KZT", True),
    ],
)
def test_monthly_minimum_with_selected_currency(text, currency, gross):
    profile = update_field(CandidateProfile(), "salary", text, salary_currency="USD")
    assert (
        profile.preferences.salary_min,
        profile.preferences.currency,
        profile.preferences.salary_gross,
    ) == (100000, currency, gross)


def test_official_quotes_cached_after_restart_and_during_outage(tmp_path, monkeypatch):
    response = httpx.Response(200, content=QUOTES, request=httpx.Request("GET", CBR_URL))
    fetch = Mock(return_value=response)
    monkeypatch.setattr(exchange_rates.httpx, "get", fetch)
    path = tmp_path / "quotes.xml"
    rates = RateCache(path).load()
    assert rates.as_of == date(2026, 11, 20)
    assert rates.convert(500000, "KZT", "EUR") == Decimal(1000)
    assert rates.convert(123, "RUR", "RUB") == 123
    assert rates.convert(123, "XYZ", "EUR") is None
    assert rates.convert(123, "XYZ", "XYZ") == 123
    assert RateCache(path).load() == rates and fetch.call_count == 1
    os.utime(path, (1, 1))
    fetch.side_effect = httpx.ConnectError("offline")
    restarted = RateCache(path)
    assert restarted.load() == rates and restarted.load() == rates and fetch.call_count == 2
    assert RateCache(tmp_path / "empty.xml").load().convert(1, "EUR", "RUB") is None


@pytest.fixture(params=["sessions", pytest.param("postgres_sessions", marks=pytest.mark.postgres)])
def salary_sessions(request):
    return request.getfixturevalue(request.param)


async def test_conversion_in_shortlist_cache_and_final_delivery(
    salary_sessions, vacancy, profile, tmp_path, monkeypatch
):
    sessions = salary_sessions
    profile.preferences.currency = "EUR"
    profile.preferences.salary_min = 1000
    profile.preferences.salary_required = True
    vacancy.currency, vacancy.salary_from, vacancy.salary_to = "KZT", 450000, 500000
    path = tmp_path / "quotes.xml"
    path.write_bytes(QUOTES)
    monkeypatch.setattr(exchange_rates, "_cache", RateCache(path))
    async with sessions.begin() as session:
        await VacancyRepository(session).save(vacancy)

    async def rank():
        stats = {}
        async with sessions.begin() as session:
            result = await rank_recent(
                VacancyRepository(session), profile, cache_owner=42, cache_stats=stats
            )
        return result, stats

    result, _ = await rank()
    card, match = result[0]
    assert match.components["salary"] == 1 and match.salary_conversion.lower == 900
    assert "900–1 000 EUR" in render_card(card, match)
    assert any(i.code == "salary_not_guaranteed" for i in match.eligibility.uncertainties)
    assert (await rank())[1]["reused"] == 1
    sender = SimpleNamespace(send_recommendation=AsyncMock(return_value=1))
    assert (
        await NotificationService(sessions, profile, chat_id=42, min_score=0).deliver(sender)
    ).sent == 1
    # A new rate on the same day makes this exact-boundary vacancy fall below the threshold.
    path.write_bytes(QUOTES.replace(b"100,00", b"100,01"))
    monkeypatch.setattr(exchange_rates, "_cache", RateCache(path))
    assert not (await rank())[0]
    profile.preferences.salary_required = False
    result, _ = await rank()
    assert 0.99 < result[0][1].components["salary"] < 1
