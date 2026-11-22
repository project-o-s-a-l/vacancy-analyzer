"""Daily Bank of Russia quotes, cached on disk for use during outages or restarts."""

import asyncio
import logging
import os
import time
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from xml.etree import ElementTree

import httpx

from domain.exchange_rates import SAME_CURRENCY, ExchangeRates, active_rates

CBR_URL = "https://www.cbr.ru/scripts/XML_daily.asp"
REFRESH_SECONDS = 24 * 3600
INVALID_QUOTE = (ValueError, InvalidOperation, ElementTree.ParseError)
logger = logging.getLogger(__name__)


def parse_rates(data: bytes) -> ExchangeRates:
    root = ElementTree.fromstring(data)
    as_of = datetime.strptime(root.get("Date", ""), "%d.%m.%Y").date()
    values = {"RUB": Decimal(1)}
    for entry in root.findall("Valute"):
        code = entry.findtext("CharCode", "").strip()
        nominal = Decimal(entry.findtext("Nominal", ""))
        value = Decimal(entry.findtext("Value", "").replace(",", "."))
        if not nominal.is_finite() or not value.is_finite() or nominal <= 0 or value <= 0:
            raise ValueError("Invalid currency quote")
        values[code] = value / nominal
    if not {"USD", "EUR"} <= values.keys():
        raise ValueError("Incomplete currency table")
    return ExchangeRates(as_of, values)


class RateCache:
    def __init__(self, path: Path):
        self.path = path
        self.snapshot = SAME_CURRENCY
        self.next_refresh = 0.0
        self.lock = Lock()

    def load(self) -> ExchangeRates:
        with self.lock:
            if time.monotonic() < self.next_refresh:
                return self.snapshot
            try:
                saved = parse_rates(self.path.read_bytes())
                self.snapshot = saved
                age = time.time() - self.path.stat().st_mtime
                if 0 <= age < REFRESH_SECONDS:
                    self.next_refresh = time.monotonic() + REFRESH_SECONDS - age
                    return saved
            except (OSError, *INVALID_QUOTE):
                pass
            try:
                response = httpx.get(CBR_URL, timeout=4, follow_redirects=True)
                response.raise_for_status()
                self.snapshot = parse_rates(response.content)
            except (httpx.HTTPError, *INVALID_QUOTE):
                self.next_refresh = time.monotonic() + 600
                logger.warning(
                    "Currency quotes unavailable; using rates dated %s", self.snapshot.as_of
                )
                return self.snapshot
            self.next_refresh = time.monotonic() + REFRESH_SECONDS
            try:
                self._save(response.content)
            except OSError:
                logger.warning("Could not persist currency quotes")
            return self.snapshot

    def _save(self, content: bytes):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = NamedTemporaryFile(dir=self.path.parent, delete=False)
        try:
            with tmp:
                tmp.write(content)
            os.replace(tmp.name, self.path)
        finally:
            Path(tmp.name).unlink(missing_ok=True)


_cache = RateCache(Path("data/cbr-daily.xml"))


async def rates_for(profile, *, live=True) -> ExchangeRates:
    if snapshot := active_rates.get():
        return snapshot
    if not live or not profile.preferences.salary_min:
        return SAME_CURRENCY
    if time.monotonic() < _cache.next_refresh:
        return _cache.snapshot
    return await asyncio.to_thread(_cache.load)
