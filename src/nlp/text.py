"""Profession-independent word normalization, including Russian inflections."""

import re
import unicodedata
from functools import lru_cache

from pymorphy3 import MorphAnalyzer

_TOKEN = re.compile(r"\.?\w+(?:[+#]+)?", re.UNICODE)
_RUSSIAN = re.compile(r"[а-я]+")


def normalize_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold().replace("ё", "е")


@lru_cache(maxsize=1)
def _morphology() -> MorphAnalyzer:
    return MorphAnalyzer()


@lru_cache(maxsize=50_000)
def _lemma(token: str) -> str:
    if _RUSSIAN.fullmatch(token):
        return _morphology().parse(token)[0].normal_form.replace("ё", "е")
    return token


def word_tokens(value: str) -> tuple[str, ...]:
    return tuple(_lemma(token) for token in _TOKEN.findall(normalize_text(value)))


def contains_phrase(tokens: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    if not phrase:
        return False
    size = len(phrase)
    return any(tokens[index : index + size] == phrase for index in range(len(tokens) - size + 1))
