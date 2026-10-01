"""Conservative exact-content grouping. Keep v1 stable for stored delivery history."""

import hashlib
import json
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
    return duplicate_key_v1(vacancy.model_dump(mode="json", include=set(DUPLICATE_FIELDS)))
