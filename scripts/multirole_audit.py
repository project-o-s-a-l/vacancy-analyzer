"""Developer runner; reads the working catalog but writes only local audit artifacts."""

import argparse
import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from jobs.matching_audit import audit_cases, catalog_snapshot, load_audit_cases
from jobs.settings import Settings
from storage.database import create_engine, session_factory
from storage.repositories import VacancyRepository


async def run(args):
    if args.output.exists():
        raise ValueError("Выберите новый каталог, чтобы не перезаписать разметку.")
    cases = load_audit_cases(args.cases)
    if args.corpus:
        snapshot = json.loads(args.corpus.read_text(encoding="utf-8"))
    else:
        engine = create_engine(Settings().database_url)
        try:
            async with session_factory(engine)() as session:
                now = datetime.now(UTC)
                vacancies, truncated = await VacancyRepository(session).matching_sample(
                    since=now - timedelta(days=30), until=now, limit=5000
                )
            if truncated:
                raise ValueError("Каталог больше лимита; выберите явно подготовленный снимок.")
            snapshot = catalog_snapshot(vacancies, now=now)
        finally:
            await engine.dispose()
    await audit_cases(cases, snapshot, args.output, features=args.features)


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Учебные резюме и реальный каталог без Telegram")
    parser.add_argument("--cases", type=Path, default=Path("config/evaluation/multirole.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--features", action="store_true")
    asyncio.run(run(parser.parse_args()))
