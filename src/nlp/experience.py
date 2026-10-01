"""Conservative experience bounds from source text; job titles never supply years."""

import re

from nlp.text import word_tokens

_CONTEXT = re.compile(
    r"\b(?:опыт\w*|стаж\w*|разработк\w*|работ[аыуе]|experience|development|production)\b",
    re.I,
)
_NO_EXPERIENCE = re.compile(
    r"\b(?:без|нет)\s+опыта\b|"
    r"\bno\s+(?:(?:prior|previous|commercial|professional|work)\s+){0,3}experience\b|"
    r"\bопыт(?:\s+(?:работы|работе|в|коммерческой|коммерческий|разработке|разработки|"
    r"профессиональной|профессиональный|общий|предыдущий)){0,7}\s+"
    r"не\s+(?:требуется|нужен|обязателен)\b|"
    r"\bне\s+(?:требуется|нужен)\s+(?:(?:коммерческий|профессиональный|предыдущий)\s+)?опыт\b|"
    r"\bexperience\s+(?:is\s+)?not\s+(?:required|necessary)\b",
    re.I,
)
_SCOPED_NO_EXPERIENCE = re.compile(
    r"\b(?:без\s+опыта|no\s+experience)\s+(?:работы\s+)?(?:с|со|with|using)\b",
    re.I,
)
_NUMERALS = {
    "один": 1,
    "два": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
    "десять": 10,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}
_DURATION = re.compile(
    r"(?<![\w.,])(?P<number>\d+(?:[.,]\d+)?)"
    r"(?P<range>\s*(?:[-‑–—]|до|to|and)\s*\d+(?:[.,]\d+)?)?"
    r"(?:\s*[-‑–—]?\s*х)?\s*\+?\s*[\])]?\s*"
    r"(?P<unit>год|лет|years?|yrs?|месяц|months?)\b",
    re.I,
)


def experience_bounds(quote: str) -> set[float]:
    """Each independent requirement retains its lower bound; contradictory bounds stay separate."""
    values = set()
    if _NO_EXPERIENCE.search(quote) and not _SCOPED_NO_EXPERIENCE.search(quote):
        values.add(0.0)
    normalized = re.sub(
        r"[a-zа-яё]+", lambda match: " ".join(word_tokens(match[0])), quote.casefold()
    )
    for word, number in _NUMERALS.items():
        normalized = re.sub(r"\b" + word + r"\b", str(number), normalized)
    matches = list(_DURATION.finditer(normalized))
    index = 0
    while index < len(matches):
        match = matches[index]
        prefix = normalized[: match.start()]
        value = float(match["number"].replace(",", "."))
        months = match["unit"].startswith(("месяц", "month"))
        if not months and not match["range"] and index + 1 < len(matches):
            following = matches[index + 1]
            gap = normalized[match.end() : following.start()]
            if (
                following["unit"].startswith(("месяц", "month"))
                and not following["range"]
                and re.fullmatch(r"\s*(?:и|and)?\s*", gap)
                and float(following["number"].replace(",", ".")) < 12
            ):
                value += float(following["number"].replace(",", ".")) / 12
                index += 1
        value = value / 12 if months else value
        upper_bound_only = re.search(r"\b(?:до|не более|менее|up to|less than|at most)\s*$", prefix)
        if not upper_bound_only and 0 <= value <= 60:
            values.add(round(value, 2))
        index += 1
    if re.search(r"\bот\s+год\b", normalized):
        values.add(1.0)
    return values


def experience_from_quote(quote: str) -> tuple[float | None, bool]:
    """Return one unambiguous source bound and whether the quote concerns work experience."""
    if not _CONTEXT.search(quote):
        return None, False
    values = experience_bounds(quote)
    return (next(iter(values)) if len(values) == 1 else None), True
