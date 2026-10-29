"""Document mentions and self-reported profile comparisons, without verification."""

import re
from dataclasses import dataclass
from datetime import date

from domain.qualifications import Qualification, QualificationMatch
from nlp.alternatives import OR
from nlp.requirements import (
    _NEGATED,
    _PREFERRED,
    _REQUIRED,
    analyze_requirements,
    section_text,
)
from nlp.text import normalize_text, word_tokens

QUALIFICATIONS_VERSION = "qualifications-v1"
_CODE = r"[A-Za-zА-Яа-я]{1,2}\d?(?:E|Е)?"
_ID_NOUN = r"удостоверени(?:е|я|ю|ем|и|й|ям|ями|ях)"
_RIGHTS_NOUN = r"прав(?:а|ами|ах)?"
_DRIVER_PREFIX = (
    r"(?:водительск\w+\s+(?:" + _ID_NOUN + "|" + _RIGHTS_NOUN + r")\s+(?:категори\w*\s+)?|"
    r"права\s+категори\w*\s+|(?:driver['’]?s?|driving)\s+licen[cs]e\s+"
    r"(?:(?:category|class)\s+)?)"
)
_DRIVER = re.compile(
    r"\b" + _DRIVER_PREFIX + r"[«\"']?(?P<code>" + _CODE + r")[»\"']?(?!\w)"
    r"(?P<tail>(?:\s*(?:,|/|\bи\b|\bили\b)\s*[«\"']?" + _CODE + r"[»\"']?(?!\w))*)",
    re.I,
)
_BARE_DRIVER = re.compile(
    r"\b(?i:права)\s+(?P<code>[A-ZА-Я]{1,3}\d?[A-ZА-Я]?)(?!\w)"
    r"(?P<tail>(?:\s*(?:,|/|\bи\b|\bили\b)\s*" + _CODE + r"(?!\w))*)"
)
_DRIVER_ONLY = re.compile(r"\bводительск\w+\s+(?:" + _ID_NOUN + "|" + _RIGHTS_NOUN + r")\b", re.I)
_PROFILE_DRIVER = re.compile(
    r"(?:" + _DRIVER_PREFIX + r"|права\s+)[«\"']?(?P<code>" + _CODE + r")[»\"']?", re.I
)
_MEDICAL_BOOK = re.compile(r"\b(?:медицинск\w*\s+книжк\w*|медкнижк\w*)\b", re.I)
_GENERIC = re.compile(
    r"\b(?:" + _ID_NOUN + r"|сертификат(?:а|ы|ов|ом|ам|ами|ах|у|е)?|"
    r"лицензи(?:я|и|ю|ей|й|ям|ями|ях)|допуск(?:а|и|ов|ом|ам|ами|ах|у|е)?|"
    r"certificat(?:e|ion)s?|licen[cs]es?|permits?)\b:?"
    r"(?:[ \t]+\w+(?:[-:/]\w+)*){0,8}",
    re.I,
)
_CONNECTOR = re.compile(r"\b(?:и|или|либо|or|and|но|but)\b", re.I)
_HOMOGLYPHS = str.maketrans({"А": "A", "В": "B", "С": "C", "Е": "E", "М": "M", "Т": "T"})
_NON_PERSONAL = re.compile(
    r"(?:лицензи\w*\s+на\s+(?:ide\b|по\b|программ\w*|софт\w*)|"
    r"сертификат\w*\s+на\s+(?:услуг\w*|покупк\w*|красот\w*|подар\w*)|"
    r"подарочн\w*\s+сертификат\w*)",
    re.I,
)
_BENEFIT = re.compile(
    r"(?:возможность\s+получения|поможем\s+(?:получить|оформить)|"
    r"за\s+счет\s+компании|оплачива\w*\s+сертификат\w*)",
    re.I,
)
_TECHNICAL_TOLERANCE = re.compile(
    r"\b(?:расч[её]т\w*\s+допуск\w*|допуск\w*\s+и\s+посад\w*|"
    r"(?:размерн\w*|геометрическ\w*)\s+допуск\w*)\b",
    re.I,
)


def qualification_name(value: str) -> str:
    value = " ".join(value.split()).strip()
    if _MEDICAL_BOOK.fullmatch(value):
        return "медицинская книжка"
    if match := _PROFILE_DRIVER.fullmatch(value):
        code = match["code"].upper().translate(_HOMOGLYPHS)
        return "водительские права категории " + code
    return normalize_text(value)


def qualification_key(value: str) -> tuple[str, ...]:
    return word_tokens(qualification_name(value))


@dataclass(frozen=True)
class _Mention:
    start: int
    end: int
    name: str
    kind: str
    uncertain: bool = False
    context: bool = False


def _mentions(text: str) -> list[_Mention]:
    result = []
    technical = [match.span() for match in _TECHNICAL_TOLERANCE.finditer(text)]

    def available(start: int, end: int) -> bool:
        return not any(left <= start < right for left, right in technical) and not any(
            start < item.end and end > item.start for item in result
        )

    for pattern in (_DRIVER, _BARE_DRIVER):
        for found in pattern.finditer(text):
            if not available(found.start(), found.end()):
                continue
            code = found["code"].upper().translate(_HOMOGLYPHS)
            tail = found.groupdict().get("tail") or ""
            uncertain = bool(tail.strip()) or not code.isascii()
            name = "водительские права категории " + code
            if tail.strip():
                name += tail
            result.append(_Mention(found.start(), found.end(), name, "driving_license", uncertain))
    for found in _MEDICAL_BOOK.finditer(text):
        if available(found.start(), found.end()):
            result.append(
                _Mention(found.start(), found.end(), "медицинская книжка", "medical_book")
            )
    for found in _DRIVER_ONLY.finditer(text):
        if available(found.start(), found.end()):
            result.append(
                _Mention(
                    found.start(),
                    found.end(),
                    "водительские права: категория не уточнена",
                    "driving_license",
                    True,
                )
            )
    for found in _GENERIC.finditer(text):
        start, end = found.span()
        if not available(start, end):
            continue
        phrase = found[0]
        cuts = [
            m.start()
            for p in (_NEGATED, _REQUIRED, _PREFERRED, _CONNECTOR)
            if (m := p.search(phrase))
        ]
        if cuts:
            end = start + min(cuts)
        raw = text[start:end].rstrip(" :")
        end = start + len(raw)
        if not raw:
            continue
        words = word_tokens(raw)
        kind = (
            "certificate"
            if words[0].startswith(("сертификат", "certificat"))
            else "permit"
            if words[0].startswith("допуск")
            else "other"
        )
        line_start = max(text.rfind(boundary, 0, start) for boundary in ("\n", ";", ",")) + 1
        stops = [
            position for boundary in ("\n", ";", ",") if (position := text.find(boundary, end)) >= 0
        ]
        line_end = min(stops) if stops else len(text)
        line = normalize_text(text[line_start:line_end])
        local_name = normalize_text(text[max(line_start, start - 16) : end])
        context = bool(
            _NON_PERSONAL.search(local_name) or _BENEFIT.search(line) and not _REQUIRED.search(line)
        )
        result.append(_Mention(start, end, qualification_name(raw), kind, len(words) < 2, context))
    return sorted(result, key=lambda item: item.start)


def is_qualification(value: str) -> bool:
    return bool(_mentions(value)) or bool(_PROFILE_DRIVER.fullmatch(value.strip()))


def compare_qualifications(
    text: str | None, declared: list[Qualification], *, today: date, explicit: list[str] = ()
) -> tuple[list[QualificationMatch], str]:
    cleaned = section_text(text)
    tags_start = len(cleaned)
    tags = [value for value in explicit if is_qualification(value)]
    if tags:
        cleaned += "\n\ue000Навыки\n" + "\n".join(tags)
    mentions = _mentions(cleaned)
    if not mentions:
        return [], cleaned
    tokens = {f"qualification_{index}": mention for index, mention in enumerate(mentions)}
    changed = cleaned
    skill_text = cleaned
    for token, mention in reversed(list(tokens.items())):
        changed = changed[: mention.start] + token + changed[mention.end :]
        skill_text = skill_text[: mention.start] + " " + skill_text[mention.end :]
    analysis = analyze_requirements(changed, vocabulary=tokens)
    group_by_token = {
        value: group
        for group in analysis.alternatives
        for value in group.options
        if value in tokens
    }
    profile = {}
    for item in declared:
        profile.setdefault(qualification_key(item.name), []).append(item)
    output = {}
    for item in analysis.skills:
        if item.skill not in tokens:
            continue
        mention = tokens[item.skill]
        from_tag = mention.start >= tags_start and bool(tags)
        importance = (
            "unspecified"
            if from_tag
            else (
                group_by_token[item.skill].importance
                if item.skill in group_by_token
                else item.importance
            )
        )
        if mention.context:
            importance = "context"
        status = "unknown"
        entries = profile.get(qualification_key(mention.name), [])
        selected = entries[0] if entries else None
        if importance in {"not_required", "context"}:
            status = importance
        elif (
            mention.uncertain
            or item.skill in group_by_token
            or OR.search(" ".join(value.text for value in item.evidence))
        ):
            status = "uncertain"
        elif len({(entry.status, entry.valid_until) for entry in entries}) > 1:
            status = "uncertain"
        elif selected:
            if selected.status == "absent":
                status = "declared_absent"
            elif selected.valid_until is not None and selected.valid_until < today:
                status = "expired"
            else:
                status = "declared_present"
        evidence = [value.model_copy(deep=True) for value in item.evidence]
        for value in evidence:
            if from_tag:
                value.origin, value.importance, value.section = "tag", "unspecified", None
            if mention.context:
                value.importance = "context"
            for token in sorted(tokens, key=len, reverse=True):
                original = tokens[token]
                value.text = value.text.replace(token, cleaned[original.start : original.end])
                if value.section:
                    value.section = value.section.replace(token, original.name)
            # Keep source OR membership for the shared fact comparison; this legacy
            # presentation adapter still reports alternatives as uncertain.
        key = qualification_key(mention.name)
        result = QualificationMatch(
            name=mention.name,
            kind=mention.kind,
            importance=importance,
            status=status,
            profile_name=selected.name if selected else None,
            valid_until=selected.valid_until if selected else None,
            evidence=evidence,
        )
        if key in output:
            old = output[key]
            old.evidence.extend(value for value in evidence if value not in old.evidence)
            kinds = {value.importance for value in old.evidence}
            conflict = "not_required" in kinds and bool(kinds & {"required", "preferred"})
            if conflict:
                old.importance, old.status = "unspecified", "uncertain"
            else:
                old.importance = next(
                    (
                        v
                        for v in ("required", "preferred", "not_required", "unspecified", "context")
                        if v in kinds
                    ),
                    "unspecified",
                )
                if old.importance in {"not_required", "context"}:
                    old.status = old.importance
                elif "uncertain" in {old.status, result.status}:
                    old.status = "uncertain"
                elif old.status in {"context", "not_required"}:
                    old.status = result.status
        else:
            output[key] = result
    return sorted(output.values(), key=lambda value: value.name), skill_text
