from datetime import UTC, datetime, timedelta

from domain import CandidateProfile
from domain.exchange_rates import use_rates
from domain.skill_gap import SampleScope, SkillGapReport
from jobs.exchange_rates import rates_for
from jobs.serving_compute import compute
from scoring.skill_gap import skill_gap_report
from storage.repositories import VacancyRepository


async def analyze_skill_gap(
    repository: VacancyRepository,
    *,
    profile: CandidateProfile | None = None,
    release_read: bool = False,
    days: int = 30,
    limit: int = 2000,
    now: datetime | None = None,
) -> SkillGapReport:
    now = now or datetime.now(UTC)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must contain a timezone")
    if not 1 <= days <= 365 or not 1 <= limit <= 10_000:
        raise ValueError("Период должен быть 1–365 дней, лимит — 1–10000 вакансий.")
    if profile is None or not profile.skills or not profile.preferences.positions:
        raise ValueError("Для анализа укажите навыки и желаемую должность в профиле.")
    now = now.astimezone(UTC)
    since = now - timedelta(days=days)
    directions = profile.preferences.positions
    selected, truncated = await repository.skill_gap_sample(
        since=since, until=now, limit=limit, positions=directions
    )
    if release_read:
        await repository.session.close()
    scope = SampleScope(
        since=since,
        until=now,
        limit=limit,
        examined=len(selected),
        truncated=truncated,
        positions=directions,
        vacancies=len(selected),
        with_skills=0,
    )
    with use_rates(await rates_for(profile)):
        return await compute(skill_gap_report, selected, scope, profile)
