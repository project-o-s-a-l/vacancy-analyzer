import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import requests
from sqlalchemy import func, select

from collectors.base import SearchQuery
from collectors.errors import CollectionStopped, VacancyParseError
from domain import CandidateProfile, Vacancy
from jobs.cli import parser
from jobs.collection_scheduler import (
    build_queries,
    collection_loop,
    load_profile_queries,
    profile_queries,
)
from jobs.users import UserService
from storage.models import VacancyModel
from storage.repositories.user_repository import RegisteredUser, UserRepository


def candidate(position, skill):
    return CandidateProfile.model_validate(
        {
            "skills": {skill: {}},
            "experience_years": 3,
            "preferences": {"positions": [position] if position else []},
        }
    )


async def store(sessions, user_id, profile, state=None):
    async with sessions.begin() as session:
        repository = UserRepository(session)
        user = await repository.register(user_id)
        await repository.save(RegisteredUser(user_id, profile, state, user.revision))


def test_default_mode_uses_profiles_and_manual_queries_can_repeat():
    default = parser().parse_args(["collect", "--watch"])
    assert default.query is None and default.days is None and default.search_field == "all"
    assert parser().parse_args(["collect", "--days", "14"]).days == 14
    assert parser().parse_args(["rank", "--profile", "profile.yaml"]).days is None
    args = parser().parse_args(["collect", "--query", "Python", "--query", "Java"])
    assert args.query == ["Python", "Java"]
    assert parser().parse_args(["collect", "--from-profiles"]).from_profiles
    with pytest.raises(SystemExit):
        parser().parse_args(["collect", "--from-profiles", "--query", "Java"])


def test_shared_positions_are_deduplicated_and_search_options_preserved():
    template = SearchQuery(text="unused", area="1", period=3, per_page=5, max_pages=2)
    profiles = [
        candidate("Python developer", "python"),
        candidate("  PYTHON   developer ", "python"),
        candidate("Java Developer", "java"),
    ]
    queries = profile_queries(profiles, template)
    assert [query.text for query in queries] == ["Python developer", "Java Developer"]
    assert all(query.search_field == "all" for query in queries)
    assert all(
        query.area == "1" and query.period == 3 and query.max_pages == 2 and query.per_page == 5
        for query in queries
    )
    assert template.text == "unused"


def test_skipped_positions_use_skills_and_empty_profiles_do_not_search():
    profiles = [candidate(None, "java"), CandidateProfile()]
    assert [q.text for q in profile_queries(profiles, SearchQuery(text="unused"))] == ["java"]
    assert profile_queries(profiles, SearchQuery(text="unused"))[0].search_field == "all"
    assert [
        q.text for q in build_queries(["", "  ", "C++", "c++", "C#"], SearchQuery(text="x"))
    ] == ["C++", "C#"]


def test_skill_and_position_searches_with_same_text_are_deduplicated():
    queries = profile_queries(
        [candidate("Java", "java"), candidate(None, "java")], SearchQuery(text="unused")
    )
    assert [(q.text.casefold(), q.search_field) for q in queries] == [
        ("java", "all"),
    ]


async def test_profile_plan_excludes_setup_and_includes_ready_profiles_being_edited(sessions):
    await store(sessions, 1, candidate("Python developer", "python"))
    await store(sessions, 2, candidate("Java Developer", "java"), "edit:salary")
    await store(sessions, 3, candidate("Go developer", "go"), "setup:remote")
    await store(sessions, 4, CandidateProfile())
    queries = await load_profile_queries(sessions, SearchQuery(text="unused"))
    assert [query.text for query in queries] == ["Python developer", "Java Developer"]


async def test_two_directions_reach_shared_catalog_and_personal_rankings(sessions):
    await store(sessions, 1, candidate("Python developer", "python"))
    await store(sessions, 2, candidate("Java developer", "java"))
    await store(sessions, 3, candidate("python developer", "python"))

    async def search(query):
        skill = query.text.split()[0].lower()
        return [
            Vacancy(
                source="hh",
                source_id=skill,
                url=f"https://example.org/jobs/{1 if skill == 'python' else 2}",
                title=query.text,
                company="Test",
                description=skill,
                skills=[skill],
                experience="1-3",
                published_at=datetime.now(UTC),
                raw_payload={"direction": skill},
            )
        ]

    source = SimpleNamespace(search=AsyncMock(side_effect=search))
    await collection_loop(source, sessions, SearchQuery(text="unused"), False, 3600)
    assert [call.args[0].text for call in source.search.await_args_list] == [
        "Python developer",
        "Java developer",
    ]
    async with sessions() as session:
        assert await session.scalar(select(func.count()).select_from(VacancyModel)) == 2
    users = UserService(sessions, min_score=0)
    python = await (await users.notifications(1)).pending()
    java = await (await users.notifications(2)).pending()
    assert python[0][0].source_id == "python" and python[0][1].matched_skills == ["python"]
    assert java[0][0].source_id == "java" and java[0][1].matched_skills == ["java"]


async def test_manual_queries_override_profile_plan(sessions):
    await store(sessions, 1, candidate("Java developer", "java"))
    source = SimpleNamespace(search=AsyncMock(return_value=[]))
    await collection_loop(
        source,
        sessions,
        SearchQuery(text="unused"),
        False,
        3600,
        query_texts=["Data analyst", "DATA   analyst", "QA engineer"],
    )
    assert [call.args[0].text for call in source.search.await_args_list] == [
        "Data analyst",
        "QA engineer",
    ]


async def test_empty_profiles_do_not_send_requests(sessions, capsys, caplog):
    caplog.set_level("INFO", logger="jobs.collection_scheduler")
    source = SimpleNamespace(search=AsyncMock(return_value=[]))
    await collection_loop(source, sessions, SearchQuery(text="unused"), False, 3600)
    source.search.assert_not_awaited()
    assert not capsys.readouterr().out
    assert any(r.event_code == "collection.no_profiles" for r in caplog.records)


async def test_watch_notices_first_completed_profile_without_restart(sessions):
    source = SimpleNamespace(search=AsyncMock(return_value=[]))
    sleeps = []
    now = [0]

    async def sleep(delay):
        sleeps.append(delay)
        now[0] += delay
        if len(sleeps) == 1:
            await store(sessions, 1, candidate("Java developer", "java"))
        else:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await collection_loop(
            source,
            sessions,
            SearchQuery(text="unused"),
            True,
            3600,
            clock=lambda: now[0],
            sleep=sleep,
        )
    assert [call.args[0].text for call in source.search.await_args_list] == ["Java developer"]
    assert sleeps == [60, 60]


async def test_watch_reads_new_profile_between_directions(sessions):
    await store(sessions, 1, candidate("Python developer", "python"))
    calls = []

    async def search(query):
        calls.append(query.text)
        if query.text == "Python developer":
            await store(sessions, 2, candidate("Java developer", "java"))
            return []
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await collection_loop(
            SimpleNamespace(search=search),
            sessions,
            SearchQuery(text="unused"),
            True,
            3600,
            clock=lambda: 0,
        )
    assert calls == ["Python developer", "Java developer"]


async def test_watch_drops_removed_directions(sessions):
    profile = candidate("Python developer", "python")
    profile.preferences.positions.append("Java developer")
    await store(sessions, 1, profile)
    calls = []

    async def search(query):
        calls.append(query.text)
        if len(calls) == 1:
            await store(sessions, 1, candidate("Go developer", "go"))
            return []
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await collection_loop(
            SimpleNamespace(search=search),
            sessions,
            SearchQuery(text="unused"),
            True,
            3600,
            clock=lambda: 0,
        )
    assert calls == ["Python developer", "Go developer"]


async def test_directions_are_not_recollected_on_every_profile_check(sessions):
    now = [0]
    calls = []

    async def search(query):
        calls.append((query.text, now[0]))
        if len(calls) == 3:
            raise asyncio.CancelledError
        return []

    async def sleep(delay):
        now[0] += delay

    with pytest.raises(asyncio.CancelledError):
        await collection_loop(
            SimpleNamespace(search=search),
            sessions,
            SearchQuery(text="unused"),
            True,
            100,
            query_texts=["Python", "Java"],
            clock=lambda: now[0],
            sleep=sleep,
        )
    assert calls == [("Python", 0), ("Java", 0), ("Python", 100)]


async def test_transport_failure_does_not_starve_other_directions(sessions):
    calls = []

    async def search(query):
        calls.append(query.text)
        if query.text == "Python":
            raise requests.Timeout
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await collection_loop(
            SimpleNamespace(search=search),
            sessions,
            SearchQuery(text="unused"),
            True,
            3600,
            query_texts=["Python", "Java"],
            clock=lambda: 0,
        )
    assert calls == ["Python", "Java"]


@pytest.mark.parametrize("error", [CollectionStopped("captcha"), VacancyParseError("bad date")])
async def test_block_and_parse_failures_stop_all_directions(sessions, error):
    source = SimpleNamespace(search=AsyncMock(side_effect=error))
    with pytest.raises(type(error)):
        await collection_loop(
            source, sessions, SearchQuery(text="unused"), True, 3600, query_texts=["Python", "Java"]
        )
    assert source.search.await_count == 1


@pytest.mark.postgres
async def test_profile_queries_work_with_postgres_jsonb(postgres_sessions):
    await store(postgres_sessions, 1, candidate("Python developer", "python"))
    await store(postgres_sessions, 2, candidate("Java developer", "java"))
    queries = await load_profile_queries(postgres_sessions, SearchQuery(text="unused"))
    assert [query.text for query in queries] == ["Python developer", "Java developer"]
