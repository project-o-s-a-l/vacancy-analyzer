"""Versioned interpretation of a vacancy, separate from the source's own fields."""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from domain.vacancy import Vacancy

ANALYSIS_VERSION = "vacancy-analysis-v1"


class VacancyCategory(StrEnum):
    IT = "it"
    FINANCE = "finance_accounting"
    SALES = "sales"
    MARKETING = "marketing"
    DESIGN = "design_media"
    TRANSPORT = "transport_logistics"
    ENGINEERING = "engineering"
    PRODUCTION = "production"
    CONSTRUCTION = "construction"
    HEALTHCARE = "healthcare"
    EDUCATION = "education_science"
    HOSPITALITY = "hospitality"
    RETAIL = "retail"
    ADMINISTRATION = "hr_administration"
    LEGAL = "legal"
    AGRICULTURE = "agriculture"
    SECURITY = "security"
    SERVICES = "services"
    OTHER = "other"


CATEGORY_LABELS = {
    VacancyCategory.IT: "IT и данные",
    VacancyCategory.FINANCE: "Финансы и бухгалтерия",
    VacancyCategory.SALES: "Продажи",
    VacancyCategory.MARKETING: "Маркетинг",
    VacancyCategory.DESIGN: "Дизайн и медиа",
    VacancyCategory.TRANSPORT: "Транспорт и логистика",
    VacancyCategory.ENGINEERING: "Инженерия",
    VacancyCategory.PRODUCTION: "Производство",
    VacancyCategory.CONSTRUCTION: "Строительство",
    VacancyCategory.HEALTHCARE: "Медицина",
    VacancyCategory.EDUCATION: "Образование и наука",
    VacancyCategory.HOSPITALITY: "Гостиницы и общепит",
    VacancyCategory.RETAIL: "Розничная торговля",
    VacancyCategory.ADMINISTRATION: "Персонал и администрация",
    VacancyCategory.LEGAL: "Юриспруденция",
    VacancyCategory.AGRICULTURE: "Сельское хозяйство",
    VacancyCategory.SECURITY: "Безопасность",
    VacancyCategory.SERVICES: "Услуги",
    VacancyCategory.OTHER: "Другое / не определено",
}

Name = Annotated[str, Field(min_length=1, max_length=160)]
Quote = Annotated[str, Field(min_length=1, max_length=800)]
SkillImportance = Literal["required", "preferred", "unspecified", "context", "not_required"]


class ExtractedSkill(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: Name
    importance: SkillImportance
    evidence: Quote


class ExtractedAlternative(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    options: list[Name] = Field(min_length=2, max_length=10)
    importance: SkillImportance
    evidence: Quote


class VacancyExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    category: VacancyCategory
    specialization: Name | None
    role_evidence: Quote
    responsibilities: list[Quote] = Field(max_length=30)
    skills: list[ExtractedSkill] = Field(max_length=60)
    alternatives: list[ExtractedAlternative] = Field(max_length=15)
    experience_min_years: float | None = Field(ge=0, le=60, allow_inf_nan=False)
    experience_evidence: Quote | None

    @model_validator(mode="after")
    def experience_needs_evidence(self) -> Self:
        if self.experience_min_years is not None and self.experience_evidence is None:
            raise ValueError("An experience minimum requires evidence")
        return self


class VacancyAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal["vacancy-analysis-v1"] = ANALYSIS_VERSION
    provider: Literal["ollama"] = "ollama"
    model: str = Field(min_length=1)
    model_digest: str = Field(min_length=1)
    input_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    pipeline_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    analyzed_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    extraction: VacancyExtraction
    warnings: list[str] = Field(default_factory=list, max_length=5)
    input_scope: Literal["full", "work_sections"] = "full"
    requirements_source: Literal["llm", "rules"] = "llm"
    omitted_description_fragments: int = Field(default=0, ge=0)
    processing_status: Literal["complete", "partial"] = "complete"
    completed_chunks: int = Field(default=1, ge=0)
    total_chunks: int = Field(default=1, ge=1)
    failure_codes: list[str] = Field(default_factory=list)
    # Full validated chunk outputs avoid a lossy max-60-skills merge on long descriptions.
    chunks: list[VacancyExtraction] = Field(default_factory=list)


def vacancy_document(vacancy: "Vacancy") -> dict:
    from nlp.vacancy_document import professional_text

    return {
        "title": vacancy.title,
        "description": professional_text(vacancy.description or vacancy.description_html),
        "source_skills": sorted(set(vacancy.skills)),
        "source_experience": vacancy.experience,
    }


def analysis_input_hash(vacancy: "Vacancy") -> str:
    document = vacancy_document(vacancy)
    # Line wrapping, HTML block wrappers and repeated whitespace are presentation noise.
    document["description"] = " ".join(document["description"].split())
    payload = json.dumps(document, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def current_analysis(vacancy: "Vacancy") -> VacancyAnalysis | None:
    analysis = vacancy.analysis
    return analysis if analysis and analysis.input_hash == analysis_input_hash(vacancy) else None


def is_fresh(vacancy: "Vacancy", *, now: datetime, days: int | None = None) -> bool:
    """Check active status and dates, with an optional explicit publication window."""
    if now.tzinfo is None or now.utcoffset() is None or days is not None and days < 1:
        raise ValueError("Invalid freshness window")
    return (
        vacancy.published_at <= now
        and (days is None or now - timedelta(days=days) <= vacancy.published_at)
        and not vacancy.archived
        and (vacancy.valid_until is None or vacancy.valid_until > now)
    )
