"""Measure uncached recommendation latency and top-five retention on a read-only catalog."""

import argparse
import asyncio
import json
import math
import random
from collections import defaultdict
from datetime import UTC, datetime
from statistics import median
from time import perf_counter

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker

from domain import CandidateProfile
from domain.retrieval import RetrievalParameters
from jobs.recommendations import recommend
from jobs.settings import Settings
from scoring.retrieval import profile_query, tokens
from storage.database import create_engine
from storage.models import VacancyModel
from storage.repositories import VacancyRepository
from storage.repositories.candidate_repository import CandidateRepository

PROFILES = {
    "Бэкенд-разработчик Python": ["Python", "FastAPI", "PostgreSQL", "Docker"],
    "Разработчик Java": ["Java", "Spring", "Kafka", "PostgreSQL"],
    "Фулстек-разработчик": ["TypeScript", "React", "Node.js", "SQL"],
    "Инженер по тестированию": ["pytest", "Postman", "SQL"],
    "Аналитик данных": ["SQL", "Python", "Power BI"],
    "Системный аналитик": ["BPMN", "UML", "SQL"],
    "Менеджер продукта": ["исследования пользователей", "продуктовая аналитика", "SQL"],
    "Преподаватель информатики": ["информатика", "Python", "педагогика"],
}

ADDITIONAL_SKILLS = {
    "Бэкенд-разработчик Python": ["Django", "SQL", "Git", "Linux", "REST API", "Redis", "pytest"],
    "Разработчик Java": ["Spring Boot", "Hibernate", "SQL", "Git", "Docker", "Linux", "REST API"],
    "Инженер по тестированию": [
        "тестирование ПО",
        "Python",
        "Selenium",
        "Playwright",
        "Git",
        "REST API",
        "тест-дизайн",
        "регрессионное тестирование",
    ],
    "Аналитик данных": [
        "Excel",
        "pandas",
        "NumPy",
        "статистика",
        "Tableau",
        "PostgreSQL",
        "визуализация данных",
        "A/B-тестирование",
    ],
}


def study_profiles(detailed):
    profiles = {}
    cases = ADDITIONAL_SKILLS if detailed else PROFILES
    for title in cases:
        skills = {name: {} for name in PROFILES[title]}
        preferences = {"positions": [title]}
        if detailed:
            skills = {
                name: {"level": "intermediate", "years": 3}
                for name in [*PROFILES[title], *ADDITIONAL_SKILLS[title]]
            }
            preferences.update(locations=["Москва"], remote="preferred")
        profiles[title] = CandidateProfile(
            preferences=preferences, skills=skills, experience_years=5
        )
    return profiles


async def measure(session, counts, repeats, at=None, detailed=False):
    await session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
    now = at or datetime.now(UTC)
    active = CandidateRepository._state_predicate(now, None, None)
    total, active_rows = (
        await session.execute(
            select(func.count(), func.count().filter(active)).select_from(VacancyModel)
        )
    ).one()
    if active_rows >= 5000:
        raise ValueError("The exhaustive reference needs a snapshot of fewer than 5000 active rows")
    if not active_rows:
        raise ValueError("The snapshot has no active vacancies")
    profiles = study_profiles(detailed)
    titles = (await session.scalars(select(VacancyModel.title).where(active).distinct())).all()
    indexed_titles = [tokens(title) for title in titles]
    full_title_matches = {}
    for title, profile in profiles.items():
        query = profile_query(profile)
        literal_title = query.titles[0]
        full_title_matches[title] = {
            "literal": sum(literal_title <= actual for actual in indexed_titles),
            "with_aliases": sum(
                any(v <= actual for v in query.titles_for(actual)) for actual in indexed_titles
            ),
        }
    reference = {}
    samples = defaultdict(list)
    retained = defaultdict(dict)

    async def run(title, k):
        trace, stats = {}, {}
        started = perf_counter()
        result = await recommend(
            VacancyRepository(session),
            profiles[title],
            now=now,
            trace=trace,
            diagnostics=stats,
            retrieval_parameters=RetrievalParameters(k=k),
            top=5,
            min_score=50,
        )
        elapsed = perf_counter() - started
        sample = {
            "total_s": elapsed,
            "retrieval_s": trace["retrieval_seconds"],
            "final_score_s": trace["filter_rank_seconds"],
            "scored": stats["scored_count"],
        }
        if stats["pool_budget_exhausted"] or stats["projection_budget_exhausted"]:
            raise RuntimeError("Selection budget exhausted; this catalog needs separate analysis")
        top_five = {
            "ranked": {v.id for v, _ in result.ranking[:5]},
            "displayed": {v.id for v, _ in result.displayed},
        }
        return top_five, sample, stats["source_exhausted"]

    # One exhaust-to-end run per profile supplies the reference and warms up the runtime/DB.
    reference_scored = {}
    for title in profiles:
        top_five, sample, exhausted = await run(title, active_rows + 1)
        if not exhausted:
            raise RuntimeError("The exhaustive run did not reach the end of the catalog")
        reference[title] = top_five
        reference_scored[title] = sample["scored"]

    rng = random.Random(20261121)
    cases = [(k, title) for k in counts for title in profiles]
    for _ in range(repeats):
        rng.shuffle(cases)
        for k, title in cases:
            top_five, sample, _ = await run(title, k)
            hits = {stage: len(ids & reference[title][stage]) for stage, ids in top_five.items()}
            if title in retained[k] and retained[k][title] != hits:
                raise RuntimeError("Retention changed inside the snapshot")
            retained[k][title] = hits
            samples[k].append(sample)
    rows = {}
    for k in sorted(samples):
        times = sorted(s["total_s"] for s in samples[k])
        rows[k] = {
            "requests": len(times),
            "total_median_s": round(median(times), 3),
            "total_p95_s": round(times[math.ceil(len(times) * 0.95) - 1], 3),
            "retrieval_median_s": round(median(s["retrieval_s"] for s in samples[k]), 3),
            "final_score_median_s": round(median(s["final_score_s"] for s in samples[k]), 3),
            "retained_ranked": sum(hits["ranked"] for hits in retained[k].values()),
            "retained_displayed": sum(hits["displayed"] for hits in retained[k].values()),
            "by_profile": {t: retained[k][t] for t in profiles},
        }
    return {
        "at": now.isoformat(),
        "profile_set": "detailed" if detailed else "basic",
        "total_rows": total,
        "active_rows": active_rows,
        "distinct_titles": len(titles),
        "full_title_matches": full_title_matches,
        "repeats": repeats,
        "reference_scored": reference_scored,
        "reference_ranked": {t: len(v["ranked"]) for t, v in reference.items()},
        "reference_displayed": {t: len(v["displayed"]) for t, v in reference.items()},
        "results": rows,
    }


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=int, nargs="+", default=[16, 32, 64, 96, 128, 192, 256])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--at", type=datetime.fromisoformat, help="Timezone-aware evaluation time")
    parser.add_argument("--detailed", action="store_true", help="Use four more detailed profiles")
    args = parser.parse_args()
    if args.repeats < 1 or any(not 1 <= k <= 5000 for k in args.counts):
        parser.error("repeats must be positive and counts must be between 1 and 5000")
    if args.at is not None and args.at.utcoffset() is None:
        parser.error("--at must include a UTC offset")
    engine = create_engine(Settings().database_url)
    try:
        async with async_sessionmaker(engine)() as session:
            result = await measure(
                session, sorted(set(args.counts)), args.repeats, args.at, args.detailed
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
