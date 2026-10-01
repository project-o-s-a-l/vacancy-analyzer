"""Personal requirement coverage; this does not measure a person's abilities."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class SkillGapVacancy(BaseModel):
    id: UUID
    title: str
    skills: list[str]
    experience: str | None
    remote: bool | None
    published_at: datetime


class SampleScope(BaseModel):
    since: datetime
    until: datetime
    limit: int
    examined: int
    truncated: bool
    positions: list[str]
    vacancies: int
    with_skills: int
    note: str = (
        "Только собранные активные вакансии. Навыки — явно указанные в карточках; "
        "обязательность не определена. Лимит применяется до отбора направления. "
        "Даты публикаций — UTC. Отсутствие навыка в профиле не доказывает его отсутствия у вас."
    )


class SkillOpportunity(BaseModel):
    skill: str
    vacancies: int
    mean_coverage_gain_pp: float
    additional_fully_covered: int


class SkillGapReport(BaseModel):
    scope: SampleScope
    excluded_by_remote: int
    mean_skill_coverage: float | None
    fully_covered: int
    insufficient_experience: int
    unknown_experience: int
    opportunities: list[SkillOpportunity] = Field(default_factory=list)
    note: str = (
        "Симуляция добавления одного навыка: требования вакансий фиксированы. "
        "Покрытие касается только записанных навыков, не вероятности найма. "
        "Сначала проверьте, все ли ваши навыки отражены в профиле."
    )
