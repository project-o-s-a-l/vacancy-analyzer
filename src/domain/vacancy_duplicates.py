"""Conservative exact-content grouping. Keep v1 stable for stored delivery history."""

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from urllib.parse import urlsplit, urlunsplit

from processing.cleaner import clean_html

DUPLICATE_FIELDS = (
    "source",
    "title",
    "company",
    "company_url",
    "description",
    "description_html",
    "skills",
    "salary_from",
    "salary_to",
    "currency",
    "salary_gross",
    "salary_period",
    "salary_raw",
    "experience",
    "employment",
    "schedule",
    "work_format",
    "working_hours",
    "location",
    "address",
    "remote",
)


def _text(value):
    return " ".join(value.split()).casefold() if isinstance(value, str) else value


def duplicate_key_v1(values: Mapping) -> str | None:
    """Different source IDs/dates may be copies; different conditions never are.

    Sparse cards without a description are left separate. We do not infer fuzzy
    similarity, merge records, or replace their original links and feedback.
    """
    description = _text(clean_html(values.get("description") or values.get("description_html")))
    if not description:
        return None
    document = {
        name: _text(values.get(name))
        for name in DUPLICATE_FIELDS
        if name not in {"description", "description_html", "skills", "company_url"}
    }
    document["description"] = description
    document["skills"] = sorted({_text(skill) for skill in values.get("skills") or []})
    company_url = values.get("company_url")
    if company_url:
        url = urlsplit(str(company_url))
        company_url = urlunsplit(
            (url.scheme.casefold(), url.netloc.casefold(), url.path.rstrip("/"), "", "")
        )
    document["company_url"] = company_url
    serialized = json.dumps({"version": 1, "fields": document}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def vacancy_duplicate_key(vacancy) -> str | None:
    if vacancy._catalog_group is not None:
        return vacancy._catalog_group
    return duplicate_key_v1(vacancy.model_dump(mode="json", include=set(DUPLICATE_FIELDS)))


CROSS_PUNCTUATION = str.maketrans("«»“”‑–—", '""""---')
SALARY_PERIODS = {
    "hour": re.compile(r"\b(?:час|hour)\b"),
    "month": re.compile(r"\b(?:месяц|month)\b"),
    "year": re.compile(r"\b(?:год|year)\b"),
}


def _cross_text(value):
    if not isinstance(value, str):
        return value
    text = unicodedata.normalize("NFKC", value)
    return _text(text.translate(CROSS_PUNCTUATION)) or None


def cross_source_key(values: Mapping) -> str | None:
    """Exact comparison across publishers. Preserve differing pay and working conditions."""
    description = _cross_text(
        clean_html(values.get("description") or values.get("description_html"))
    )
    company = _cross_text(values.get("company"))
    if not description or company in {None, "", "не указан", "не указана", "неизвестно"}:
        return None
    if len(description) < 40 and not (values.get("location") or values.get("salary_from")):
        return None
    # Platforms give the same employer different profile URLs and tag the same text differently.
    fields = {
        key: _cross_text(values.get(key))
        for key in DUPLICATE_FIELDS
        if key
        not in {"source", "company_url", "skills", "description", "description_html", "salary_raw"}
    }
    fields["description"] = description
    salary = _cross_text(values.get("salary_raw"))
    if values.get("salary_from") is None and values.get("salary_to") is None:
        fields["salary_raw"] = salary
    if salary and fields["salary_gross"] is None:
        if "на руки" in salary or "после вычета" in salary:
            fields["salary_gross"] = False
        elif "до вычета" in salary:
            fields["salary_gross"] = True
    period = fields["salary_period"] or salary or ""
    for normalized, pattern in SALARY_PERIODS.items():
        if pattern.search(period):
            fields["salary_period"] = normalized
            break
    if fields["location"]:
        fields["location"] = re.sub(r"^г\.\s*", "", fields["location"])
    employment = fields["employment"]
    if employment in {"full", "full_time", "полная занятость", "полный рабочий день"}:
        fields["employment"] = "full_time"
    if fields["currency"] == "rur":
        fields["currency"] = "rub"
    return hashlib.sha256(
        json.dumps(fields, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
