"""Personal requirement coverage; this does not measure a person's abilities."""

from datetime import datetime

from pydantic import BaseModel, Field


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
        "Только собранные активные вакансии. Используются общие факты и ограничения подбора. "
        "Лимит учитывает уникальные вакансии выбранного направления. "
        "Даты публикаций — UTC. Отсутствие навыка в профиле не доказывает его отсутствия у вас."
    )


class SkillOpportunity(BaseModel):
    skill: str
    vacancies: int
    mean_coverage_gain_pp: float
    additional_fully_covered: int


class SkillGapReport(BaseModel):
    scope: SampleScope
    excluded_by_constraints: int = 0
    comparison_version: str = "professional-comparison-v1"
    requirement_coverage: float | None = None
    uncertain_requirements: int = 0
    confirmed_deficits: int = 0
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
