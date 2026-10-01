"""Sequential stand on frozen data in an isolated PostgreSQL schema.

No HH, Telegram, LLM or working profiles. Always drops only its own test schema.
"""

import argparse
import asyncio
import json
import statistics
import time
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from jobs.benchmark_data import load_bundle
from jobs.rank import rank_recent
from jobs.saved_selections import selection_contract
from jobs.settings import Settings
from storage.models import Base, SavedSelectionModel
from storage.repositories import VacancyRepository


def signature(ranking):
    return [(str(v.id), m.model_dump(mode="json")) for v, m in ranking]


async def measure(bundle, output):
    if output.exists():
        raise ValueError("Выберите новый файл измерений")
    catalog, cases, _, manifest = load_bundle(bundle)
    schema = "va07_measure_" + uuid4().hex
    url = Settings().database_url
    if not url.startswith("postgresql+asyncpg:"):
        raise ValueError("Стенду нужна PostgreSQL/asyncpg")
    admin, engine, created = create_async_engine(url), None, False
    try:
        async with admin.begin() as connection:
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        created = True
        engine = create_async_engine(url, connect_args={"server_settings": {"search_path": schema}})
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        async with sessions.begin() as session:
            for vacancy in catalog.vacancies:
                await VacancyRepository(session).save(vacancy)

        async def request(case, owner):
            stats = {}
            started = time.perf_counter()
            async with sessions.begin() as session:
                result = await rank_recent(
                    VacancyRepository(session),
                    case.profile,
                    cache_owner=owner,
                    cache_stats=stats,
                    now=catalog.reference_time,
                )
            return result, stats, time.perf_counter() - started

        rows = []
        # All held-out profiles, fixed comparison without tuning cache parameters.
        validation = [c for c in cases.cases if c.split == "validation"]
        for index, case in enumerate(validation, 1):
            _, cold, cold_seconds = await request(case, index)
            baseline_times, cache_times, count = [], [], {}
            for repeat in range(2):
                # Alternate order to reduce a systematic warm-connection advantage.
                order = (None, index) if repeat == 0 else (index, None)
                runs = {owner: await request(case, owner) for owner in order}
                baseline, _, baseline_time = runs[None]
                cached, count, cache_time = runs[index]
                assert signature(cached) == signature(baseline)
                assert count["recomputed"] == 0
                baseline_times.append(baseline_time)
                cache_times.append(cache_time)
            rows.append(
                {
                    "case_id": case.id,
                    "cold_seconds": cold_seconds,
                    "cold_computed": cold["recomputed"],
                    "warm": count,
                    "baseline_seconds": baseline_times,
                    "cached_seconds": cache_times,
                    "equal_to_full": True,
                }
            )

        # Separate allocation measurement; don't mix traced timings with normal timings.
        tracemalloc.start()
        await request(validation[0], 1)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        async with sessions() as session:
            max_entries = max(
                len(row.entries) for row in await session.scalars(select(SavedSelectionModel))
            )
            pool_rows = await session.scalar(select(func.count()).select_from(SavedSelectionModel))
        before = statistics.median(t for r in rows for t in r["baseline_seconds"])
        after = statistics.median(t for r in rows for t in r["cached_seconds"])
        report = {
            "version": "va07-postgres-stand-v1",
            "created_at": datetime.now(UTC).isoformat(),
            "contract": selection_contract(),
            "catalogue_rows": len(catalog.vacancies),
            "bundle_manifest": manifest,
            "profiles": len(rows),
            "rows": rows,
            "baseline_median_seconds": before,
            "cached_median_seconds": after,
            "speedup": before / after,
            "score_calls_before": sum(r["cold_computed"] for r in rows),
            "score_calls_after": sum(r["warm"]["recomputed"] for r in rows),
            "python_peak_bytes_one_cached_request": peak,
            "saved_owner_rows": pool_rows,
            "max_pool_entries": max_entries,
            "equal_to_full": True,
            "external_calls": 0,
            "parallel_requests": 1,
            "scope": "Whole rank_recent with PostgreSQL transactions, live retrieval, cache IO, "
            "baseline and rules ordering; excludes send-time checks, RSS, PG memory, "
            "model shadow predictions, cold imports and ingestion. Not a load test.",
        }
        assert after < before and report["score_calls_before"] > 0
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "profiles",
                        "baseline_median_seconds",
                        "cached_median_seconds",
                        "speedup",
                        "score_calls_before",
                        "score_calls_after",
                        "equal_to_full",
                    )
                },
                ensure_ascii=False,
            )
        )
    finally:
        if engine is not None:
            await engine.dispose()
        if created:
            assert schema.startswith("va07_measure_") and len(schema) == len("va07_measure_") + 32
            async with admin.begin() as connection:
                await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(measure(args.bundle, args.output))
