"""One exchange-rate snapshot for retrieval, scoring and the final delivery check."""

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_CEILING, Decimal
from functools import cached_property

from domain.professional import digest

HH_CURRENCY_ALIASES = {"RUR": "RUB", "BYR": "BYN"}


def currency_code(value: str) -> str:
    value = value.upper()
    return HH_CURRENCY_ALIASES.get(value, value)


@dataclass(frozen=True)
class ExchangeRates:
    as_of: date | None
    rubles_per_unit: Mapping[str, Decimal]

    @cached_property
    def revision(self):
        return digest([str(self.as_of), {k: str(v) for k, v in self.rubles_per_unit.items()}])

    def convert(self, amount, source: str, target: str) -> Decimal | None:
        if amount is None or not source or not target:
            return None
        source, target = currency_code(source), currency_code(target)
        if source == target:
            return Decimal(amount)
        if source not in self.rubles_per_unit or target not in self.rubles_per_unit:
            return None
        return Decimal(amount) * self.rubles_per_unit[source] / self.rubles_per_unit[target]

    def minimum_by_currency(self, amount: int, currency: str) -> dict[str, int]:
        """Integer vacancy bounds must reach the exact converted minimum."""
        minimums = {currency_code(currency): amount}
        for code in self.rubles_per_unit:
            value = self.convert(amount, currency, code)
            if value is not None:
                minimums[code] = int(value.to_integral_value(rounding=ROUND_CEILING))
        for alias, code in HH_CURRENCY_ALIASES.items():
            if code in minimums:
                minimums[alias] = minimums[code]
        return minimums


SAME_CURRENCY = ExchangeRates(None, {"RUB": Decimal(1)})
active_rates: ContextVar[ExchangeRates | None] = ContextVar("salary_rates", default=None)


def current_rates() -> ExchangeRates:
    return active_rates.get() or SAME_CURRENCY


@contextmanager
def use_rates(snapshot: ExchangeRates):
    token = active_rates.set(snapshot)
    try:
        yield
    finally:
        active_rates.reset(token)
