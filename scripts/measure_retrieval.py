"""Small PostgreSQL stand: 510 newer wrong-role cards and one older relevant card.

Creates and always drops its own schema. No HH, Telegram, Ollama or working profiles.
"""

import argparse
import asyncio
import json
import time
import tracemalloc
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from sqlalchemy import insert, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from domain import CandidateProfile, Vacancy
from domain.retrieval import RetrievalParameters
from domain.vacancy_analysis import analysis_input_hash
from domain.vacancy_duplicates import vacancy_duplicate_key
from jobs.candidates import retrieve_candidates
from jobs.settings import Settings
from scoring import rank_vacancies
from scoring.retrieval import document_terms, profile_query, search_document
from storage.models import Base, VacancyModel
from storage.payloads import payload_hash
from storage.repositories import VacancyRepository
from storage.repositories.candidate_repository import CandidateRepository


async def measure(output):
    if output.exists():
        raise ValueError("Выберите новый файл измерений")
    schema = "va05_measure_" + uuid4().hex
    url = Settings().database_url
    if not url.startswith("postgresql+asyncpg:"):
        raise ValueError("Стенду нужна PostgreSQL/asyncpg")
    admin = create_async_engine(url)
    engine = None
    created = False
    try:
        async with admin.begin() as connection:
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        created = True
        engine = create_async_engine(url, connect_args={"server_settings": {"search_path": schema}})
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        now = datetime.now(UTC)
        profile = CandidateProfile(
            skills={"java": {}, "spring": {}},
            experience_years=3,
            preferences={"positions": ["Java Developer"]},
        )
        cards = []
        for number in range(511):
            relevant = number == 510
            cards.append(
                Vacancy(
                    source="benchmark",
                    source_id=str(number),
                    url=f"https://example.invalid/vacancy/{number}",
                    company="Synthetic stand",
                    title="Java Developer" if relevant else f"Python Developer {number}",
                    description="Java Spring" if relevant else "Python API",
                    skills=["Java", "Spring"] if relevant else ["Python"],
                    published_at=now - timedelta(days=200 if relevant else 1),
                    raw_payload={"synthetic": number},
                )
            )
        values = []
        for vacancy in cards:
            document = search_document(vacancy.model_dump())
            values.append(
                vacancy.model_dump()
                | {
                    "url": str(vacancy.url),
                    "last_seen_at": vacancy.collected_at,
                    "raw_payload_hash": payload_hash(vacancy.raw_payload),
                    "duplicate_key": vacancy_duplicate_key(vacancy),
                    "search_version": document["version"],
                    "search_revision": analysis_input_hash(vacancy),
                    "content_revision": analysis_input_hash(vacancy),
                    "search_document": document,
                    "search_terms": document_terms(document),
                }
            )
        async with sessions.begin() as session:
            await session.execute(insert(VacancyModel), values)
            await session.execute(text("ANALYZE vacancies"))
        tracemalloc.start()
        started = time.perf_counter()
        async with sessions() as session:
            selected = await retrieve_candidates(VacancyRepository(session), profile, now=now)
            ranking = rank_vacancies(selected.items, profile, now=now)
        seconds = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        exhaustive = rank_vacancies(cards, profile, now=now)
        async with sessions() as session:
            predicate = CandidateRepository(session)._overlap(profile_query(profile).terms)
            sql = str(
                select(VacancyModel.id)
                .where(predicate)
                .compile(dialect=engine.dialect, compile_kwargs={"literal_binds": True})
            )
            natural_plan = list(await session.scalars(text("EXPLAIN " + sql)))
            await session.execute(text("SET LOCAL enable_seqscan = off"))
            index_plan = list(await session.scalars(text("EXPLAIN " + sql)))
        report = {
            "version": "va05-postgres-stand-v1",
            "rows": len(cards),
            "newer_wrong_role": 510,
            "parameters": RetrievalParameters().model_dump(),
            "scanned": selected.scanned,
            "selected": len(selected.items),
            "old_relevant_selected": cards[-1].id in {v.id for v in selected.items},
            "old_relevant_top1": ranking[0][0].id == cards[-1].id,
            "legacy_recent_500_misses_relevant": cards[-1].id
            not in {
                v.id
                for v in sorted(cards, key=lambda v: (-v.published_at.timestamp(), v.source_id))[
                    :500
                ]
            },
            "exhaustive_top5_recall": len(
                {v.id for v, _ in ranking[:5]} & {v.id for v, _ in exhaustive[:5]}
            )
            / 5,
            "seconds_selection_and_ranking": seconds,
            "python_peak_bytes": peak,
            "natural_plan": natural_plan,
            "index_capability_plan": index_plan,
            "memory_scope": (
                "Python allocations during one warm-index query; "
                "excludes PostgreSQL/RSS and index construction"
            ),
            "external_calls": 0,
        }
        assert report["old_relevant_selected"] and report["old_relevant_top1"]
        assert any("ix_vacancies_search_terms" in line for line in index_plan)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in (
                        "rows",
                        "selected",
                        "old_relevant_top1",
                        "seconds_selection_and_ranking",
                        "python_peak_bytes",
                    )
                },
                ensure_ascii=False,
            )
        )
    finally:
        if engine is not None:
            await engine.dispose()
        if created:
            assert schema.startswith("va05_measure_") and len(schema) == len("va05_measure_") + 32
            async with admin.begin() as connection:
                await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(measure(parser.parse_args().output))
