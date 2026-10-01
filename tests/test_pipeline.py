import json
from datetime import UTC, datetime

import pytest

from collectors.base import SearchQuery
from domain import Vacancy
from jobs.cli import load_profile, main, render
from jobs.collect import collect_vacancies
from jobs.rank import rank_recent
from scoring.baseline import rank_vacancies
from storage.repositories import VacancyRepository


async def test_source_to_storage_to_explained_ranking(sessions, session, sample_data, profile):
    class ExampleSource:
        async def search(self, query):
            return [
                Vacancy.model_validate(payload | {"published_at": datetime.now(UTC).isoformat()})
                for payload in sample_data["vacancies"]
            ]

    repository = VacancyRepository(session)
    source = ExampleSource()
    query = SearchQuery(text="python")
    assert await collect_vacancies(source, sessions, query) == 3
    assert await collect_vacancies(source, sessions, query) == 3
    ranking = await rank_recent(repository, profile)
    assert len(ranking) == 2
    assert all(match.eligible and match.eligibility for _, match in ranking)
    first, match = ranking[0]
    assert first.source_id == "demo-1"
    assert match.matched_skills == ["docker", "fastapi", "postgresql", "python"]
    assert match.reasons
    assert len(await repository.history(first.id)) == 1


def test_rank_json_contains_explanations_and_excludes_raw_source(sample_data, profile, now):
    vacancies = [Vacancy.model_validate(payload) for payload in sample_data["vacancies"]]
    results = json.loads(render(rank_vacancies(vacancies, profile, now=now), as_json=True))
    assert len(results) == 2
    assert results[0]["vacancy"]["source_id"] == "demo-1"
    assert "raw_payload" not in results[0]["vacancy"]
    assert "components" in results[0]["match"]


def test_example_profile_loads():
    from pathlib import Path

    profile = load_profile(Path(__file__).parents[1] / "config/profile.example.yaml")
    assert profile.skills["python"].years == 3


def test_cli_help_lists_real_operations(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["vacancy-analyzer", "--help"])
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 0
    help_text = capsys.readouterr().out
    assert all(
        command in help_text for command in ("bot", "collect", "rank", "skill-gap", "notify")
    )
    assert "demo" not in help_text and "telegram-preview" not in help_text
