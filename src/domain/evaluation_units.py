"""Strict evaluation boundary; never repairs evidence or changes serving policy."""

import re

from pydantic import BaseModel

UNIT_CONTRACT = "evaluation-units-v2"
SALARY_PERIODS = frozenset({"MONTH", "HOUR", "YEAR", "DAY", "WEEK", "SHIFT"})


def _wire(value):
    return value.model_dump() if isinstance(value, BaseModel) else value


def _currency(value, path):
    if value is not None and (
        not isinstance(value, str) or not re.fullmatch(r"[A-Z]{3}", value) or value == "RUR"
    ):
        raise ValueError(f"{path}: use canonical currency (RUB, not RUR/rub) or null")


def _boolean(value, path, *, nullable=True):
    if type(value) is not bool and not (nullable and value is None):
        raise ValueError(f"{path}: expected JSON boolean" + (" or null" if nullable else ""))


def _location(value, path):
    if value is not None and (
        not isinstance(value, str) or not value.strip() or value != value.strip()
    ):
        raise ValueError(f"{path}: expected nonblank trimmed location or null")


def validate_units(data):
    """Run before Pydantic can coerce false/0/strings or default unknown salary units."""
    for case in data.get("vacancies", []):
        case = _wire(case)
        vacancy = _wire(case["vacancy"])
        path = f"vacancies[{case.get('id', '?')}].vacancy"
        period = vacancy.get("salary_period")
        if period is not None and (not isinstance(period, str) or period not in SALARY_PERIODS):
            raise ValueError(f"{path}.salary_period: use MONTH/HOUR/YEAR/DAY/WEEK/SHIFT or null")
        _currency(vacancy.get("currency"), path + ".currency")
        for key in ("salary_gross", "remote"):
            _boolean(vacancy.get(key), path + "." + key)
        _location(vacancy.get("location"), path + ".location")
        if any(vacancy.get(key) is not None for key in ("salary_from", "salary_to")):
            for key in ("currency", "salary_gross", "salary_period"):
                if key not in vacancy:
                    raise ValueError(f"{path}.{key}: explicit unit required; use null if unknown")
    for case in data.get("profiles", []):
        case = _wire(case)
        prefs = _wire(_wire(case["profile"]).get("preferences", {}))
        path = f"profiles[{case.get('id', '?')}].profile.preferences"
        _currency(prefs.get("currency"), path + ".currency")
        for key in ("salary_gross", "salary_required", "locations_required"):
            if key in prefs:
                _boolean(prefs[key], path + "." + key, nullable=False)
        for location in prefs.get("locations", []):
            _location(location, path + ".locations")
        if prefs.get("salary_min") is not None:
            for key in ("currency", "salary_gross"):
                if prefs.get(key) is None:
                    raise ValueError(f"{path}.{key}: explicit unit required for salary_min")
