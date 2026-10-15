"""Section-aware vacancy input. Unknown sections are retained, never silently truncated."""

import re
from dataclasses import dataclass

from nlp.ollama_input import _OTHER_HEADING, _WORK_HEADING
from nlp.requirements import _HEADER_PATTERNS
from processing.cleaner import clean_html

DOCUMENT_VERSION = "vacancy-sections-v2"
_TASK = re.compile(r"обязанност|задач|заниматься|делать|responsibilit|duties|tasks|what you", re.I)
_CONDITION = re.compile(r"условия|предлагаем|offer|benefits|perks", re.I)
_EXPLICIT_CANDIDATE = re.compile(
    r"(?:от (?:вас|кандидата) (?:требуется|ожидаем)|"
    r"(?:обязательно|требуется|необходимо)\s+(?:знание|владение|опыт|умение|наличие)|"
    r"(?:candidate|you)\s+must\s+(?:have|know)|required skills)",
    re.I,
)


@dataclass(frozen=True)
class VacancySection:
    start: int
    end: int
    text: str
    kind: str
    heading: str


def sections(value):
    text = clean_html(value)
    kind, heading = "unknown", ""
    result = []
    for match in re.finditer(r"[^\n]+", text):
        line = match[0]
        label = line.strip(" •●▪*–—-:")
        # Also recognize inline headings; do not discard the remainder of the line.
        prefix = label.split(":", 1)[0].strip()
        requirement_heading = any(
            status != "context" and pattern.fullmatch(prefix)
            for pattern, status in _HEADER_PATTERNS
        )
        if len(prefix) <= 100 and (_WORK_HEADING.fullmatch(prefix) or requirement_heading):
            kind = "tasks" if _TASK.search(prefix) else "requirements"
            heading = prefix
        elif len(prefix) <= 100 and _OTHER_HEADING.fullmatch(prefix):
            kind = "conditions" if _CONDITION.search(prefix) else "nonprofessional"
            heading = prefix
        elif line.endswith(":") and len(line) <= 100:
            kind, heading = "unknown", prefix
        line_kind = kind
        # Requirements occasionally appear below Benefits/About. Retain explicit
        # candidate constraints, but never infer them from employer history or perks.
        if kind in {"conditions", "nonprofessional"} and _EXPLICIT_CANDIDATE.search(line):
            line_kind = "requirements"
        result.append(VacancySection(match.start(), match.end(), line, line_kind, heading))
    return result


def professional_text(value):
    return "\n".join(
        part.text for part in sections(value) if part.kind not in {"conditions", "nonprofessional"}
    )


def professional_quote(value, quote):
    """Only exact professional evidence can introduce a model fact."""
    return source_quote(value, quote) is not None


def source_quote(value, quote):
    """Reanchor cached evidence after whitespace-only source formatting changes."""
    if not quote:
        return None
    text = professional_text(value)
    original = clean_html(value)
    if quote in text and quote in original:
        return quote
    pattern = r"\s+".join(re.escape(word) for word in quote.split())
    match = re.search(pattern, text)
    return match[0] if match and match[0] in original else None


def inference_documents(vacancy, *, max_chars=4000):
    """Bounded blocks retaining section heading, including every professional character."""
    from domain.vacancy_analysis import vacancy_document

    document = vacancy_document(vacancy)
    blocks = []
    for section in sections(vacancy.description or vacancy.description_html):
        if section.kind in {"conditions", "nonprofessional"}:
            continue
        # A long unbroken line is split with bounded overlap; rules still scan it in full.
        offset = 0
        while offset < len(section.text):
            piece = section.text[offset : offset + max_chars]
            prefix = section.heading + ":\n" if section.heading not in piece else ""
            blocks.append({**document, "description": prefix + piece})
            if offset + max_chars >= len(section.text):
                break
            offset += max_chars - 200
    # Pack small neighboring lines without losing headings.
    packed = []
    for block in blocks:
        if packed and len(packed[-1]["description"]) + len(block["description"]) < max_chars:
            packed[-1]["description"] += "\n" + block["description"]
        else:
            packed.append(block)
    return packed or [{**document, "description": ""}]
