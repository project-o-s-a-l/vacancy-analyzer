"""Read-only local model check on explicitly selected frozen vacancy texts."""

import argparse
import asyncio
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from domain import Vacancy
from domain.vacancy_analysis import is_fresh
from jobs.settings import Settings
from jobs.vacancy_analysis import vacancy_analyzer
from nlp.ollama import VacancyAnalysisError


async def run(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise ValueError("Choose a new report path")
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    selected = set(args.vacancy_id)
    vacancies = [
        Vacancy.model_validate(item)
        for item in corpus["vacancies"]
        if item["source_id"] in selected
    ]
    if {item.source_id for item in vacancies} != selected:
        raise ValueError("Some selected IDs are absent from the corpus")
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "source": str(args.corpus),
        "database_written": False,
        "hh_requested": False,
        "independent_quality_evaluation": False,
        "results": [],
    }
    settings = Settings(
        vacancy_llm_enabled=True, **({"ollama_model": args.model} if args.model else {})
    )
    captured = {}

    async def capture(response):
        if response.request.url.path == "/api/chat":
            await response.aread()
            captured["response"] = response.json()

    async with vacancy_analyzer(settings) as analyzer:
        if args.capture_response:
            analyzer.client.event_hooks["response"].append(capture)
        await analyzer.ready()
        for vacancy in vacancies:
            if not is_fresh(vacancy, now=datetime.now(UTC), days=settings.vacancy_fresh_days):
                raise ValueError("Selected vacancy is outside the freshness window")
            started = time.monotonic()
            result = {"source_id": vacancy.source_id, "title": vacancy.title}
            try:
                parsed = await analyzer.analyze(vacancy)
                result["analysis"] = parsed.model_dump(mode="json")
            except VacancyAnalysisError as exc:
                result["error"] = str(exc)
            result["elapsed_seconds"] = round(time.monotonic() - started, 2)
            if args.capture_response:
                result["model_response"] = captured.pop("response", None)
            report["results"].append(result)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                vacancy.source_id,
                "ok" if "analysis" in result else result["error"],
                result["elapsed_seconds"],
                flush=True,
            )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--vacancy-id", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-response", action="store_true")
    parser.add_argument("--model")
    asyncio.run(run(parser.parse_args()))
