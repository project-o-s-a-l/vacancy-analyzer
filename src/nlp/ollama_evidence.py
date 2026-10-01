"""Resolve model-selected source references into verbatim evidence; no fuzzy matching."""

import re
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from domain.vacancy_analysis import (
    Name,
    SkillImportance,
    VacancyCategory,
    VacancyExtraction,
)
from nlp.alternatives import OR
from nlp.extraction import extract_skills, skill_key

Reference = Annotated[str, Field(pattern=r"^[tds][1-9]\d{0,3}$")]


class _SkillReference(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: Name
    importance: SkillImportance
    evidence: Reference


class _AlternativeReference(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    options: list[Name] = Field(min_length=2, max_length=10)
    importance: SkillImportance
    evidence: Reference


class ReferencedExtraction(BaseModel):
    """Ollama wire format only; stored and matching contracts continue using exact quotes."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    category: VacancyCategory
    specialization: Name | None
    role_evidence: Reference
    responsibilities: list[Reference] = Field(max_length=30)
    skills: list[_SkillReference] = Field(max_length=60)
    alternatives: list[_AlternativeReference] = Field(max_length=15)
    experience_evidence: Reference | None


def extraction_schema(document: dict[str, dict[str, str]]) -> dict:
    """Constrain generation to the references actually sent in this request."""
    schema = ReferencedExtraction.model_json_schema()
    references = [key for fragments in document.values() for key in fragments]
    schema["$defs"]["SourceReference"] = {"type": "string", "enum": references}
    reference = {"$ref": "#/$defs/SourceReference"}
    properties = schema["properties"]
    properties["role_evidence"] = dict(reference)
    properties["responsibilities"]["items"] = dict(reference)
    for field in ("skills", "alternatives"):
        definition = properties[field]["items"]["$ref"].rsplit("/", 1)[-1]
        schema["$defs"][definition]["properties"]["evidence"] = dict(reference)
    alternative_references = [
        key
        for fragments in document.values()
        for key, value in fragments.items()
        if OR.search(value)
    ]
    if alternative_references:
        definition = properties["alternatives"]["items"]["$ref"].rsplit("/", 1)[-1]
        schema["$defs"][definition]["properties"]["evidence"] = {
            "type": "string",
            "enum": alternative_references,
        }
    else:
        properties["alternatives"]["maxItems"] = 0
    description_references = list(document["description"])
    properties["experience_evidence"] = (
        {
            "anyOf": [
                {"type": "string", "enum": description_references},
                {"type": "null"},
            ]
        }
        if description_references
        else {"type": "null"}
    )
    return schema


def referenced_document(document: dict) -> dict[str, dict[str, str]]:
    """Preserve the whole source, splitting long lines so each stored quote fits its limit."""
    result = {}
    for field, prefix in (("title", "t"), ("description", "d"), ("source_skills", "s")):
        values = document[field] if field == "source_skills" else [document[field]]
        fragments = {}
        for value in values:
            lines = value.splitlines()
            if field == "description":
                lines = [
                    part
                    for line in lines
                    for part in re.split(r"(?<=[.!?;])\s+(?=[A-ZА-ЯЁ])", line)
                ]
            for line in lines:
                line = line.strip()
                while line:
                    end = len(line) if len(line) <= 800 else line.rfind(" ", 0, 801)
                    if end <= 0:
                        end = 800
                    fragments[f"{prefix}{len(fragments) + 1}"] = line[:end]
                    line = line[end:].lstrip()
        result[field] = fragments
    return result


class UnknownEvidence(ValueError):
    """Contains only a field path and a schema-validated reference, never source/model text."""


class UnconfirmedAlternative(ValueError):
    """An existing fragment does not support the proposed skill alternatives."""


def _supports_alternative(quote: str, options: list[str]) -> bool:
    keys = {skill_key(value) for value in options}
    found = {skill_key(value) for value in extract_skills(quote, vocabulary=options)}
    return len(keys) >= 2 and bool(OR.search(quote)) and keys <= found


def validate_alternatives(
    extraction: VacancyExtraction, document: dict[str, dict[str, str]]
) -> tuple[VacancyExtraction, list[str]]:
    """Correct a misplaced reference only when exactly one sent fragment supports it."""
    alternatives = []
    warnings = []
    sources = {value for fragments in document.values() for value in fragments.values()}
    for index, item in enumerate(extraction.alternatives):
        if _supports_alternative(item.evidence, item.options):
            alternatives.append(item)
            continue
        candidates = [quote for quote in sources if _supports_alternative(quote, item.options)]
        if len(candidates) != 1:
            raise UnconfirmedAlternative(
                f"alternatives[{index}].evidence: нет однозначного подтверждения вариантов навыков"
            )
        alternatives.append(item.model_copy(update={"evidence": candidates[0]}))
        warnings.append("Основание альтернативы уточнено по исходному тексту")
    return (
        extraction.model_copy(update={"alternatives": alternatives}),
        list(dict.fromkeys(warnings)),
    )


def resolve_extraction(
    extraction: ReferencedExtraction, document: dict[str, dict[str, str]]
) -> tuple[VacancyExtraction, list[str]]:
    sources = {key: value for fragments in document.values() for key, value in fragments.items()}

    def quote(reference: str, field: str) -> str:
        if reference not in sources:
            raise UnknownEvidence(f"{field}: нет исходного фрагмента {reference}")
        return sources[reference]

    data = extraction.model_dump()
    data["role_evidence"] = quote(extraction.role_evidence, "role_evidence")
    data["responsibilities"] = [
        quote(reference, f"responsibilities[{index}]")
        for index, reference in enumerate(extraction.responsibilities)
    ]
    for field in ("skills", "alternatives"):
        for index, item in enumerate(data[field]):
            item["evidence"] = quote(item["evidence"], f"{field}[{index}].evidence")
    warnings = []
    data["experience_min_years"] = None
    reference = extraction.experience_evidence
    if reference is not None:
        if reference not in document["description"]:
            data.update(experience_min_years=None, experience_evidence=None)
            warnings.append("Неподтверждённое требование опыта исключено")
        else:
            data["experience_evidence"] = document["description"][reference]
    return VacancyExtraction.model_validate(data), warnings
