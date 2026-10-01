import re
from collections.abc import Iterable

from nlp.skills import SKILLS
from nlp.text import contains_phrase, word_tokens
from processing.cleaner import clean_html

_ALIASES = {alias.casefold(): name for name, aliases in SKILLS.items() for alias in aliases}
_PATTERNS = {
    name: re.compile(
        r"(?<![\w+#.])(?:" + "|".join(re.escape(a) for a in aliases) + r")(?![\w+#])",
        re.IGNORECASE,
    )
    for name, aliases in SKILLS.items()
}


def canonical_skill(value: str) -> str:
    normalized = " ".join(value.casefold().split())
    return _ALIASES.get(normalized, normalized)


def skill_key(value: str) -> tuple[str, ...]:
    canonical = canonical_skill(value)
    return word_tokens(canonical) or (canonical,)


def extract_skills(
    text: str | None,
    explicit: Iterable[str] = (),
    *,
    vocabulary: Iterable[str] = (),
) -> list[str]:
    """Accept any explicit/profile skill; the seed dictionary only adds known aliases."""
    cleaned = clean_html(text)
    result = {canonical_skill(value) for value in explicit if value.strip()}
    result.update(name for name, pattern in _PATTERNS.items() if pattern.search(cleaned))
    candidates = {canonical_skill(value) for value in vocabulary if value.strip()}
    if candidates and cleaned:
        tokens = word_tokens(cleaned)
        result.update(value for value in candidates if contains_phrase(tokens, skill_key(value)))
    return sorted(result)
