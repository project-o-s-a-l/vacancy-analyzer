"""Conservative RU/EN rules for skill mentions, without a profession whitelist.

Only recognizable terms (source tags, profile vocabulary, optional seed aliases) are
classified. This is evidence extraction, not a complete parser of job requirements.
"""

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable

from bs4 import BeautifulSoup, NavigableString

from domain.requirements import (
    AlternativeRequirement,
    Importance,
    RequirementAnalysis,
    RequirementEvidence,
    SkillRequirement,
)
from nlp.alternatives import OR, choice_options
from nlp.extraction import canonical_skill, extract_skills, skill_key
from nlp.text import normalize_text
from processing.cleaner import clean_html

RULES_VERSION = "requirements-v2"

# A heading sets the default until another heading. Inline cues affect only their line.
_HEADERS: tuple[tuple[str, Importance], ...] = (
    (r"что не входит в зону ответственности|что не требуется|not required", "not_required"),
    (
        r"обязательные (?:требования|навыки)|требования(?: к кандидату)?|"
        r"необходимые навыки|что (?:мы )?ожидаем(?: от вас)?|"
        r"(?:candidate )?requirements|qualifications|required(?: skills| qualifications)?|"
        r"must[- ]haves?",
        "required",
    ),
    (
        r"желательные (?:требования|навыки)|будет плюсом|будет преимуществом|"
        r"приветствуется|желательно|nice[- ]to[- ]haves?|preferred(?: skills| qualifications)?|"
        r"bonus(?: skills)?",
        "preferred",
    ),
    (
        r"(?:основные |ваши )?обязанности|(?:ваши |основные )?задачи|"
        r"чем предстоит заниматься|что предстоит делать|"
        r"условия(?: работы)?|мы предлагаем|о (?:компании|нас|команде)|"
        r"responsibilities|duties|what we offer|benefits|about us",
        "context",
    ),
    (
        r"наш (?:технологический )?стек|технологии(?: проекта)?|навыки|"
        r"our (?:tech )?stack|technical stack|tech stack|skills",
        "unspecified",
    ),
)
_HEADER_PATTERNS = [
    (re.compile(r"^(?:" + pattern + r")(?:\s*[:—–]\s*|\s*$)", re.I), importance)
    for pattern, importance in _HEADERS
]
_NEGATED = re.compile(
    r"\b(?:не\s+(?:требу\w*|нуж\w*|обязател\w*)|необязател\w*|"
    r"без\s+(?:знани\w*|владени\w*|опыта(?:\s+работы)?(?:\s+с)?)|"
    r"not\s+(?:strictly\s+)?(?:required|necessary|mandatory)|"
    r"no\s+(?:[^\s,;]+\s+){0,5}?required|"
    r"no\s+(?:prior\s+)?(?:experience|knowledge)(?:\s+(?:of|with|in))?|"
    r"(?:do(?:es)?\s+not|don['’]t|doesn['’]t)\s+require|"
    r"no\s+need(?:\s+(?:for|to))?)\b",
    re.I,
)
_REQUIRED = re.compile(
    r"\b(?:обязател(?:ен|ьна|ьно|ьны|ьный|ьная|ьное|ьные|ьным|ьной|ьных|ьными|ьного|ьную)|"
    r"требу(?:ется|ются)|необходим(?:о|а|ы|ый|ая|ое|ые|ым|ой|ых|ыми|ого|ую)?|нужно|нужен|нужны|"
    r"must|required|mandatory|essential)\b",
    re.I,
)
_PREFERRED = re.compile(
    r"\b(?:желател\w*|приветству\w*|(?:будет|будут)\s+(?:большим\s+)?"
    r"(?:плюсом|преимуществом)|preferred|nice[- ]to[- ]have|bonus|a\s+plus)\b",
    re.I,
)
_LINES = re.compile(r"[\n;]+|(?<=[.!?])\s+")
_HEADING = "\ue000"


def section_text(value: str | None) -> str:
    """Preserve standalone HTML headings, including unfamiliar section names."""
    if not value or not re.search(r"<(?:h[1-6]|p)\b", value, re.I):
        return clean_html(value)
    soup = BeautifulSoup(value, "html.parser")
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p"]):
        text = tag.get_text(" ", strip=True)
        bold = tag.find(["strong", "b"])
        is_heading = tag.name != "p" or bold and bold.get_text(" ", strip=True) == text
        if is_heading and text and len(text) <= 120:
            tag.replace_with(NavigableString("\n" + _HEADING + text + "\n"))
    return clean_html(str(soup))


def _importance(text: str, default: Importance) -> tuple[Importance, bool]:
    normalized = normalize_text(text)
    negative = bool(_NEGATED.search(normalized))
    rest = _NEGATED.sub("", normalized)
    required = bool(_REQUIRED.search(rest))
    preferred = bool(_PREFERRED.search(rest))
    if default == "context":
        return "context", False
    if negative and (required or preferred) or required and preferred:
        return "unspecified", True
    if negative:
        return "not_required", False
    if required:
        return "required", False
    if preferred:
        return "preferred", False
    return default, False


def _group_id(options: list[str]) -> str:
    identity = sorted(skill_key(value) for value in options)
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16]


def analyze_requirements(
    text: str | None,
    explicit: Iterable[str] = (),
    *,
    vocabulary: Iterable[str] = (),
) -> RequirementAnalysis:
    explicit = list(explicit)
    cleaned = section_text(text)
    terms = extract_skills(cleaned, explicit, vocabulary=vocabulary)
    names: dict[tuple[str, ...], str] = {}
    evidence = defaultdict(list)

    def add(value: str, item: RequirementEvidence) -> None:
        key = skill_key(value)
        label = canonical_skill(value)
        names[key] = min(names.get(key, label), label)
        if item not in evidence[key]:
            evidence[key].append(item)

    for value in explicit:
        if value.strip():
            add(
                value, RequirementEvidence(origin="tag", importance="unspecified", text=value[:280])
            )

    section: str | None = None
    default: Importance = "unspecified"
    warnings = set()
    alternatives: dict[str, AlternativeRequirement] = {}
    for raw_line in _LINES.split(cleaned):
        line = raw_line.strip(" \t•*-–—")
        if not line:
            continue
        marked_heading = line.startswith(_HEADING)
        if marked_heading:
            line = line.removeprefix(_HEADING)
        normalized = normalize_text(line)
        for pattern, importance in _HEADER_PATTERNS:
            if match := pattern.match(normalized):
                default, section = importance, line[: match.end()].strip(" :—–")[:100]
                line = line[match.end() :].strip()
                break
        else:
            if marked_heading:
                default, ambiguous = _importance(line, "unspecified")
                if ambiguous:
                    default = "unspecified"
                section = line[:100]
            # Unknown short headings terminate the previous section conservatively.
            elif line.endswith(":") and len(line) <= 100:
                default, section = "unspecified", line[:-1]
                continue
        if not line:
            continue
        alternative = bool(OR.search(line))
        line_status, line_ambiguous = _importance(line, default)
        carried = line_status if line_status in {"required", "preferred"} else default
        clauses = [
            (part, extract_skills(part, vocabulary=terms))
            for part in re.split(r",|\s+но\s+|\s+but\s+", line, flags=re.I)
        ]
        unbound_cue = line_ambiguous and any(
            not found and any(pattern.search(part) for pattern in (_NEGATED, _REQUIRED, _PREFERRED))
            for part, found in clauses
        )
        # A comma list like 'Java, Python or Kotlin' has ambiguous scope. Explicit
        # cues or an OR in every clause permit independent conditions on one line.
        unsafe_scope = (
            alternative
            and len(clauses) > 1
            and any(
                part.strip()
                and not OR.search(part)
                and not any(pattern.search(part) for pattern in (_REQUIRED, _PREFERRED, _NEGATED))
                for part, found in clauses
            )
        )
        for clause, found in clauses:
            status, ambiguous = _importance(clause, carried)
            if not found:
                continue
            options = None
            group_id = None
            if (
                OR.search(clause)
                and status in {"required", "preferred", "unspecified"}
                and not ambiguous
                and not unbound_cue
                and not unsafe_scope
            ):
                options = choice_options(clause, terms)
            if (
                ambiguous
                or unbound_cue
                or (unsafe_scope or OR.search(clause) and options is None)
                and status not in {"not_required", "context"}
            ):
                status = "unspecified"
                warnings.add("Альтернативные или противоречивые требования требуют ручной проверки")
            if status in {"required", "preferred"}:
                carried = status
            if options is not None and len(options) > 1:
                group_id = _group_id(options)
                item = RequirementEvidence(
                    origin="description",
                    importance=status,
                    text=clause.strip(),
                    section=section,
                    alternative_id=group_id,
                )
                if group_id not in alternatives:
                    alternatives[group_id] = AlternativeRequirement(
                        id=group_id, options=options, importance=status, evidence=[item]
                    )
                else:
                    group = alternatives[group_id]
                    priority = {"required": 3, "unspecified": 2, "preferred": 1}
                    if priority[status] > priority[group.importance]:
                        group.importance = status
                    if item not in group.evidence:
                        group.evidence.append(item)
            for value in found:
                add(
                    value,
                    RequirementEvidence(
                        origin="description",
                        importance=status,
                        text=clause.strip(),
                        section=section,
                        alternative_id=group_id,
                    ),
                )

    # A tag can itself contain a complete choice. It must not become an extra
    # independent skill when its wording uses another language's OR connector.
    for value in explicit:
        if not OR.search(value):
            continue
        options = choice_options(value, terms)
        if options is None or len(options) < 2:
            continue
        group_id = _group_id(options)
        if group_id not in alternatives:
            alternatives[group_id] = AlternativeRequirement(
                id=group_id, options=options, importance="unspecified"
            )
        for item in evidence[skill_key(value)]:
            if item.origin == "tag":
                item.alternative_id = group_id
                if item not in alternatives[group_id].evidence:
                    alternatives[group_id].evidence.append(item)

    skills = []
    for key, items in evidence.items():
        described = {
            v.importance for v in items if v.origin == "description" and v.alternative_id is None
        }
        grouped = any(v.alternative_id is not None for v in items)
        positive = described & {"required", "preferred"}
        if "not_required" in described and positive:
            importance = "unspecified"
            warnings.add("Для одного навыка найдены противоречивые требования")
        elif "required" in described:
            importance = "required"
        elif "preferred" in described:
            importance = "preferred"
        elif "not_required" in described:
            importance = "not_required"
        elif "unspecified" in described or any(v.origin == "tag" for v in items) or grouped:
            importance = "unspecified"
        else:
            importance = "context"
        counted = not grouped or bool(described & {"required", "preferred"})
        skills.append(
            SkillRequirement(
                skill=names[key], importance=importance, evidence=items, counted_separately=counted
            )
        )
    for group in alternatives.values():
        option_keys = {skill_key(value) for value in group.options}
        negated_keys = {
            skill_key(item.skill) for item in skills if item.importance == "not_required"
        }
        confirmed = any(
            item.origin == "description" and item.importance in {"required", "preferred"}
            for item in group.evidence
        )
        if option_keys <= negated_keys and not confirmed:
            group.importance = "not_required"
        elif option_keys & negated_keys:
            group.importance = "unspecified"
            warnings.add("Важность группы альтернатив противоречива; проверьте описание")
    return RequirementAnalysis(
        skills=sorted(skills, key=lambda v: v.skill),
        warnings=sorted(warnings),
        alternatives=sorted(alternatives.values(), key=lambda v: v.id),
    )
