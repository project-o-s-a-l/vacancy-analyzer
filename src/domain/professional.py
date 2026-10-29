"""Shared, lossless professional facts. Absence of evidence never creates a level."""

import hashlib
import json
from datetime import date
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, model_validator

FACT_SCHEMA = "professional-facts-v1"
PROFILE_SCHEMA = "candidate-profile-v2"
FACT_POLICY = "facts-conservative-v1"
Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Name = Annotated[str, Field(min_length=1, max_length=1000)]
Origin = Literal["user_confirmed", "machine_proposed", "source", "legacy_unknown"]


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def compatible_wire(value):
    """Keep hash-bound legacy review/benchmark payloads byte-semantically stable."""
    if isinstance(value, dict):
        return {
            key: compatible_wire(child)
            for key, child in value.items()
            if key != "professional_facts" or child is not None
        }
    if isinstance(value, list):
        return [compatible_wire(child) for child in value]
    return value


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Evidence(StrictModel):
    source_id: Name
    source_revision: Digest
    field: Name
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)
    quote: str | None = None
    section: str | None = None
    interpretation: (
        Literal["required", "preferred", "context", "not_required", "unspecified"] | None
    ) = None

    @model_validator(mode="after")
    def complete_span(self) -> Self:
        if any(v is not None for v in (self.start, self.end, self.quote)):
            if self.start is None or self.end is None or not self.quote:
                raise ValueError("Evidence needs start, end and quote together")
            if self.end <= self.start or self.end - self.start != len(self.quote):
                raise ValueError("Invalid Unicode evidence span")
        return self


class Claim[T](StrictModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)
    status: Literal["present", "missing", "unknown", "explicit_absent", "conflict"] = "missing"
    value: T | None = None
    origin: Origin = "legacy_unknown"
    evidence: list[Evidence] = Field(default_factory=list)
    alternatives: list[T] = Field(default_factory=list)
    legacy_value: JsonValue = None

    @model_validator(mode="after")
    def honest_state(self) -> Self:
        if (self.status == "present") != (self.value is not None):
            raise ValueError("Only present claims carry a resolved value")
        if self.status == "conflict":
            if len(self.alternatives) < 2:
                raise ValueError("Conflict needs competing values")
        elif self.alternatives:
            raise ValueError("Alternatives require conflict")
        if self.status in {"present", "explicit_absent", "conflict"}:
            if self.origin != "legacy_unknown" and not self.evidence:
                raise ValueError("An assertion needs evidence")
        return self


class NamedFact(StrictModel):
    id: Name
    name: Name
    assertion: Claim[bool] = Field(default_factory=Claim[bool])


class SkillFact(NamedFact):
    kind: Literal["skill"] = "skill"
    level: Claim[Literal["beginner", "intermediate", "advanced"]] = Field(default_factory=Claim)
    years: Claim[Annotated[float, Field(ge=0, allow_inf_nan=False)]] = Field(default_factory=Claim)
    last_used: Claim[date] = Field(default_factory=Claim)

    model_config = ConfigDict(extra="forbid", validate_default=True)


class RoleFact(NamedFact):
    kind: Literal["role"] = "role"
    context: Literal["held", "desired", "required", "unspecified"] = "unspecified"


class TaskFact(NamedFact):
    kind: Literal["task"] = "task"


class QualificationFact(NamedFact):
    kind: Literal["qualification"] = "qualification"
    valid_until: Claim[date] = Field(default_factory=Claim)
    model_config = ConfigDict(extra="forbid", validate_default=True)


class ValueFact(StrictModel):
    kind: Literal["preference", "condition", "experience_total", "seniority"]
    id: Name
    name: Name
    assertion: Claim[JsonValue]


ProfessionalFact = Annotated[
    SkillFact | RoleFact | TaskFact | QualificationFact | ValueFact, Field(discriminator="kind")
]


class ExperienceEpisode(StrictModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)
    id: Name
    kind: Literal["employment", "project", "unspecified"]
    context: Literal["work", "academic", "personal", "unspecified"] = "unspecified"
    started_precision: Literal["day", "month", "unknown"] = "unknown"
    ended_precision: Literal["day", "month", "unknown"] = "unknown"
    organization: Claim[str] = Field(default_factory=Claim)
    started: Claim[date] = Field(default_factory=Claim)
    ended: Claim[date] = Field(default_factory=Claim)
    ongoing: Claim[bool] = Field(default_factory=Claim)
    role_ids: list[str] = Field(default_factory=list)
    skill_ids: list[str] = Field(default_factory=list)
    task_ids: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.started.value and self.ended.value and self.started.value > self.ended.value:
            raise ValueError("Episode dates are reversed")
        if self.ongoing.value is True and self.ended.value is not None:
            raise ValueError("Ongoing episode cannot have a resolved end")
        return self


class Requirement(StrictModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)
    id: Name
    fact_id: Name
    scope: Literal["vacancy", "role", "skill", "episode"] = "vacancy"
    scope_id: str | None = None
    importance: Claim[Literal["required", "preferred", "context", "not_required"]] = Field(
        default_factory=Claim
    )
    years_min: Claim[Annotated[float, Field(ge=0, allow_inf_nan=False)]] = Field(
        default_factory=Claim
    )
    evidence: list[Evidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def scoped(self) -> Self:
        if (self.scope == "vacancy") != (self.scope_id is None):
            raise ValueError("A scoped requirement needs its subject")
        return self


class Alternative(StrictModel):
    id: Name
    operator: Literal["any_of"] = "any_of"
    requirement_ids: list[str] = Field(min_length=2)


class SourceReference(StrictModel):
    id: Name
    revision: Digest
    kind: Literal["profile", "resume", "vacancy", "extraction", "legacy"]


class LexicalObservation(StrictModel):
    """Candidate-independent phrase index; matching never reparses source text."""

    tokens: list[str]
    importance: Literal["required", "preferred", "context", "not_required", "unspecified"]
    ambiguous: bool = False
    evidence: Evidence


class ProfessionalFacts(StrictModel):
    schema_version: Literal["professional-facts-v1"] = FACT_SCHEMA
    document_id: Name
    subject: Literal["candidate", "vacancy"]
    owner_id: int | None = Field(default=None, gt=0)
    source_revision: Digest
    input_sha256: Digest
    extractor: Name
    extractor_version: Name
    extractor_contract: Digest
    model_id: str | None = None
    model_digest: str | None = None
    analysis_pipeline: str | None = None
    policy_version: Name = FACT_POLICY
    extracted_at: AwareDatetime
    reference_time: AwareDatetime
    input_scope: Literal["full", "partial", "legacy"]
    input_ranges: list[tuple[int, int]] = Field(default_factory=list)
    uncertainty_reasons: list[str] = Field(default_factory=list)
    sources: list[SourceReference]
    facts: list[ProfessionalFact] = Field(default_factory=list)
    episodes: list[ExperienceEpisode] = Field(default_factory=list)
    requirements: list[Requirement] = Field(default_factory=list)
    alternatives: list[Alternative] = Field(default_factory=list)
    lexical_observations: list[LexicalObservation] = Field(default_factory=list)
    # Calendar overlaps are never summed automatically into total/relevant tenure.
    overlap_policy: Literal["unaggregated", "calendar_union_v1"] = "unaggregated"

    @model_validator(mode="after")
    def references(self) -> Self:
        if (self.subject == "candidate") != (self.owner_id is not None):
            raise ValueError("Only candidate facts need an owner")
        if self.input_scope == "partial" and not self.input_ranges:
            raise ValueError("Partial input needs processed character ranges")
        if any(start < 0 or end <= start for start, end in self.input_ranges):
            raise ValueError("Invalid processed input range")
        facts = {f.id: f.kind for f in self.facts}
        episodes = {e.id for e in self.episodes}
        requirements = {r.id for r in self.requirements}
        if len(facts) != len(self.facts) or len(episodes) != len(self.episodes):
            raise ValueError("Duplicate fact/episode identity")
        if len(requirements) != len(self.requirements):
            raise ValueError("Duplicate requirement identity")
        for episode in self.episodes:
            for ids, kind in (
                (episode.role_ids, "role"),
                (episode.skill_ids, "skill"),
                (episode.task_ids, "task"),
            ):
                if any(facts.get(key) != kind for key in ids):
                    raise ValueError("Invalid episode reference")
        for requirement in self.requirements:
            if requirement.fact_id not in facts:
                raise ValueError("Invalid requirement reference")
            if requirement.scope != "vacancy":
                valid = (
                    requirement.scope_id in episodes
                    if requirement.scope == "episode"
                    else facts.get(requirement.scope_id) == requirement.scope
                )
                if not valid:
                    raise ValueError("Invalid requirement scope")
        for alternative in self.alternatives:
            if len(set(alternative.requirement_ids)) < 2 or any(
                key not in requirements for key in alternative.requirement_ids
            ):
                raise ValueError("Invalid alternative references")
        sources = {(s.id, s.revision) for s in self.sources}

        def inspect(value):
            if isinstance(value, Evidence):
                if (value.source_id, value.source_revision) not in sources:
                    raise ValueError("Evidence references an undeclared source")
            elif isinstance(value, BaseModel):
                for name in type(value).model_fields:
                    inspect(getattr(value, name))
            elif isinstance(value, list):
                for child in value:
                    inspect(child)

        for collection in (self.facts, self.episodes, self.requirements, self.lexical_observations):
            inspect(collection)
        return self

    @property
    def revision(self) -> str:
        return digest(self.model_dump(mode="json", exclude={"extracted_at"}))

    def compatibility(self, *, input_revision, extractor_contract, policy_version=FACT_POLICY):
        """Explicit reasons to re-extract/review; legacy provenance is never upgraded here."""
        return [
            name
            for name, current, expected in (
                ("input_revision", self.input_sha256, input_revision),
                ("extractor_contract", self.extractor_contract, extractor_contract),
                ("policy_version", self.policy_version, policy_version),
            )
            if current != expected
        ]

    def verify_text(self, source_id: str, revision: str, fields: dict[str, str]) -> None:
        """Verify exact spans at the extraction boundary, before persisting a fact set."""

        def inspect(value):
            if isinstance(value, Evidence):
                if value.source_id == source_id and value.source_revision == revision:
                    if value.start is not None:
                        text = fields.get(value.field)
                        if text is None or text[value.start : value.end] != value.quote:
                            raise ValueError("Evidence does not match source text")
            elif isinstance(value, BaseModel):
                for name in type(value).model_fields:
                    inspect(getattr(value, name))
            elif isinstance(value, list):
                for child in value:
                    inspect(child)

        inspect(self)
