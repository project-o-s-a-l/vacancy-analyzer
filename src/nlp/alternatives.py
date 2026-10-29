"""Recognize complete, simple OR lists without inventing unknown branches."""

import re
from functools import lru_cache

from nlp.extraction import extract_skills, skill_key
from nlp.skills import SKILLS
from nlp.text import word_tokens

OR = re.compile(r"\b(?:или|либо|or)\b", re.I)


@lru_cache(maxsize=1)
def _grammar() -> set[str]:
    # These are connective/cue words, not a vocabulary of permitted professions.
    return set(
        word_tokens(
            "опыт работа разработка знание владение навык уверенный практический базовый "
            "хороший глубокий с на в по из один любой следующий обязательно обязательный "
            "необходимый необходимо требуется желательный желательно приветствуется будет "
            "плюсом преимущество experience work working development knowledge proficiency "
            "skill skills with of in on either one the following required mandatory essential "
            "preferred nice to have a plus bonus"
        )
    )


def _branch_skill(branch: str, terms: list[str]) -> str | None:
    found = {skill_key(value): value for value in extract_skills(branch, vocabulary=terms)}
    if len(found) != 1:
        return None
    value = next(iter(found.values()))
    tokens = word_tokens(branch)
    aliases = (value, *SKILLS.get(value, ()))
    for alias in aliases:
        phrase = word_tokens(alias)
        for start in range(len(tokens) - len(phrase) + 1):
            if tokens[start : start + len(phrase)] == phrase:
                remainder = tokens[:start] + tokens[start + len(phrase) :]
                if all(token in _grammar() for token in remainder):
                    return value
    return None


def choice_options(clause: str, terms: list[str]) -> list[str] | None:
    """None means incomplete/complex syntax; one option means equivalent aliases."""
    if any(character in clause for character in "()[]"):
        return None
    if atomic := _branch_skill(clause, terms):
        return [atomic]
    # A single trailing duration qualifies the complete simple choice. Multiple
    # branch-specific numbers remain unresolved instead of sharing one bound.
    duration = r"(?:от|не менее|at least)\s+\d+(?:[.,]\d+)?\s*(?:лет|года?|years?)\.?$"
    if len(re.findall(r"\d+(?:[.,]\d+)?", clause)) == 1:
        clause = re.sub(duration, "", clause, flags=re.I).strip()
    branches = OR.split(clause)
    if len(branches) < 2:
        return None
    options = {}
    for branch in branches:
        name = _branch_skill(branch, terms)
        if name is None:
            return None
        key = skill_key(name)
        options[key] = min(options.get(key, name), name)
    return sorted(options.values())
