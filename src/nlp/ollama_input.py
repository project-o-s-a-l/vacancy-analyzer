"""Fit explicitly marked work sections into one bounded request; never cut a fragment."""

import json
import re
from dataclasses import dataclass
from typing import Literal

from nlp.ollama_evidence import referenced_document

_WORK_HEADING = re.compile(
    r"(?:(?:основные |должностные |ваши )?(?:обязанности|задачи)|"
    r"чем (?:предстоит|нужно|вы будете) заниматься|что (?:предстоит|нужно) делать|"
    r"требования(?: к кандидату| к соискателю| к опыту)?|"
    r"(?:ключевые |профессиональные |технические )?(?:навыки|компетенции)|"
    r"опыт(?: работы)?|квалификация|документы(?: и допуски)?|допуски|"
    r"будет (?:плюсом|преимуществом)|желательно|наш стек(?: технологий)?|стек(?: технологий)?|"
    r"(?:key |main |your )?(?:responsibilities|duties|tasks)|"
    r"(?:job |candidate )?requirements|qualifications|(?:technical |key )?skills|"
    r"(?:work )?experience|nice to have|preferred(?: qualifications| skills)?|"
    r"(?:our )?(?:tech |technology )?stack|what you(?:'ll| will) do|what we expect(?: from you)?)",
    re.I,
)
_OTHER_HEADING = re.compile(
    r"(?:о (?:нас|компании|команде)|условия(?: работы)?|мы предлагаем|что мы предлагаем|"
    r"наши преимущества|преимущества|этапы (?:отбора|собеседования)|как откликнуться|контакты|"
    r"about (?:us|the company|the team)|what we offer|(?:our )?benefits|"
    r"perks(?: and benefits)?|hiring process|how to apply|contacts?)",
    re.I,
)


class InputTooLarge(ValueError):
    """Neither the full source nor complete marked work sections fit the request budget."""


@dataclass(frozen=True)
class PreparedInput:
    references: dict[str, dict[str, str]]
    content: str
    scope: Literal["full", "work_sections"]
    omitted_fragments: int


def prepare_input(document: dict, *, max_chars: int) -> PreparedInput:
    references = referenced_document(document)

    def serialize(value: dict) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    content = serialize(references)
    if len(content) <= max_chars:
        return PreparedInput(references, content, "full", 0)
    selected = {}
    in_work_section = False
    for reference, value in references["description"].items():
        heading = " ".join(value.split()).strip("•●▪*–—- :").casefold()
        if len(heading) <= 100:
            if _WORK_HEADING.fullmatch(heading):
                in_work_section = True
            elif _OTHER_HEADING.fullmatch(heading) or value.endswith(":"):
                in_work_section = False
        if in_work_section:
            selected[reference] = value
    if selected:
        scoped = {**references, "description": selected}
        content = serialize(scoped)
        if len(content) <= max_chars:
            return PreparedInput(
                scoped,
                content,
                "work_sections",
                len(references["description"]) - len(selected),
            )
    raise InputTooLarge(
        "Полный текст и выделенные рабочие разделы превышают лимит; текст не обрезан"
    )
