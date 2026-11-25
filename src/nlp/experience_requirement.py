"""Reconcile explicit candidate experience requirements with HH's coarse field."""

import re

from domain import Vacancy
from domain.eligibility import EligibilityEvidence, ExperienceRequirement
from nlp.experience import experience_bounds, experience_from_quote
from processing.cleaner import clean_html

EXPERIENCE_MINIMUM = {"no_experience": 0, "1-3": 1, "3-6": 3, "6+": 6}
_EXPERIENCE_UPPER = {"no_experience": 0, "1-3": 3, "3-6": 6}
_WORD = re.compile(r"\b(?:опыт\w*|стаж\w*|experience)\b", re.I)
_GENERAL = re.compile(
    r"коммерческ\w*|профессиональн\w*|\b(?:commercial|professional)\b|"
    r"\b(?:опыт\s+работы(?!\s+(?:с|со)\b)|стаж\s+работы|"
    r"work\s+experience(?!\s+(?:with|using)\b))\b",
    re.I,
)
_EMPLOYER = re.compile(
    r"\b(?:наш\w*\s+опыт|опыт\s+наш\w*|опыт\s+компани\w*|"
    r"компани\w*\s+\d+|у\s+нас\s+(?:более\s+|уже\s+)?\d+|"
    r"our\s+experience|company['’]s\s+experience)\b",
    re.I,
)
_PREFERRED = re.compile(
    r"будет\s+(?:плюсом|преимуществом)|желател\w*|preferred|nice to have|bonus", re.I
)
_CONTEXT = re.compile(
    r"^(?:о компании|о нас|about us|about the company|обязанности|условия|"
    r"мы предлагаем|benefits|responsibilities|duties)(?:\s*:|\s*$)",
    re.I,
)
_REQUIREMENTS = re.compile(
    r"^(?:требования|что для нас важно|что мы ожидаем|квалификация|requirements|"
    r"qualifications)(?:\s*:|\s*$)",
    re.I,
)
_NOT_REQUIRED = re.compile(
    r"не\s+(?:требуется|нужен|обязателен)|not\s+(?:required|necessary)", re.I
)


def source_minimum(value):
    if not value:
        return None
    if value in EXPERIENCE_MINIMUM:
        return EXPERIENCE_MINIMUM[value]
    if value.casefold() == "без опыта":
        return 0
    bound = re.fullmatch(r"от\s+(\d{1,2})\s+(?:лет|года?)", value.strip(), re.I)
    return int(bound[1]) if bound else None


def experience_requirement(vacancy: Vacancy) -> ExperienceRequirement:
    hh = source_minimum(vacancy.experience)
    evidence = (
        [EligibilityEvidence(source=vacancy.source, text=vacancy.experience)]
        if hh is not None
        else []
    )
    quotes: list[tuple[str, float | None, bool]] = []
    context = False
    preferred = False
    text = clean_html(vacancy.description or vacancy.description_html)
    for raw in re.split(r"[\n;]+|(?<=[.!?])\s+(?=[А-ЯЁA-Z])", text):
        line = raw.strip(" •*-—–\t")
        if not line:
            continue
        if len(line) <= 100:
            if _CONTEXT.match(line):
                context, preferred = True, False
            elif _PREFERRED.fullmatch(line.rstrip(" :")):
                context, preferred = False, True
            elif _REQUIREMENTS.match(line):
                context, preferred = False, False
            elif line.endswith(":"):
                context, preferred = False, False
        if context or preferred or _PREFERRED.search(line) or _EMPLOYER.search(line):
            continue
        if not _WORD.search(line):
            continue
        minimum, relevant = experience_from_quote(line)
        if relevant and (
            minimum is not None or len(experience_bounds(line)) > 1 or _GENERAL.search(line)
        ):
            # A technical skill whose experience is not required does not waive total tenure.
            if minimum is None and _NOT_REQUIRED.search(line):
                continue
            quotes.append((line, minimum, minimum != 0))

    evidence += [EligibilityEvidence(source="description", text=q) for q, _, _ in quotes]
    # Selecting a shorter LLM quote cannot remove conflicting bounds in the full requirement.
    ambiguous = [experience_bounds(q) for q, _, _ in quotes if len(experience_bounds(q)) > 1]
    if ambiguous:
        return ExperienceRequirement(
            required=True if all(min(values) > 0 for values in ambiguous) else None,
            conflict=True,
            evidence=evidence,
        )
    numbers = {value for _, value, _ in quotes if value is not None}
    positive = any(required for _, _, required in quotes)
    zero = 0 in numbers
    if len(numbers) == 1 and not (zero and positive):
        minimum = next(iter(numbers))
        return ExperienceRequirement(
            minimum_years=minimum,
            minimum_source="description",
            required=minimum > 0,
            conflict=hh is not None
            and (minimum < hh or minimum > _EXPERIENCE_UPPER.get(vacancy.experience, float("inf"))),
            evidence=evidence,
        )
    if len(numbers) > 1 or zero and positive:
        return ExperienceRequirement(
            required=True if positive and not zero else None, conflict=True, evidence=evidence
        )
    if positive:
        return ExperienceRequirement(
            minimum_years=hh if hh and hh > 0 else None,
            minimum_source=vacancy.source if hh and hh > 0 else None,
            required=True,
            conflict=hh == 0,
            evidence=evidence,
        )
    return ExperienceRequirement(
        minimum_years=hh,
        minimum_source=vacancy.source if hh is not None else None,
        required=hh > 0 if hh is not None else None,
        evidence=evidence,
    )
