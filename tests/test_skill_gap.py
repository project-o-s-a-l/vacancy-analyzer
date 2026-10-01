from datetime import timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from aiogram import Bot
from aiogram.methods import SendMessage
from sqlalchemy import event
from test_bot import TEST_TOKEN, RecordingSession, command_update
from test_users import store_profile

from bot.cards import split_message
from bot.handlers import build_dispatcher
from domain import CandidateProfile
from jobs.cli import parser, run_command
from jobs.skill_gap import analyze_skill_gap
from jobs.users import UserService
from scoring.skill_gap import render_skill_gap
from storage.repositories import VacancyRepository


def candidate(position="бухгалтер", skills=("Excel",), **preferences):
    return CandidateProfile.model_validate(
        {
            "skills": dict.fromkeys(skills, {}),
            "experience_years": 2,
            "preferences": {"positions": [position], **preferences},
        }
    )


def test_analysis_message_chunks_preserve_emoji_and_text():
    text = "Текст\n" + "😀" * 4000 + "\nконец"
    chunks = split_message(text)
    assert "".join(chunks) == text
    assert all(len(part.encode("utf-16-le")) // 2 <= 3500 for part in chunks)


async def seed(repository, vacancy, now, **values):
    changes = {
        "id": uuid4(),
        "source_id": uuid4().hex,
        "published_at": now - timedelta(days=1),
        "title": "Бухгалтер",
        "skills": ["Excel", "Расчёт зарплат"],
        "salary_from": None,
        "salary_to": None,
        "experience": "1-3",
        "remote": True,
        "valid_until": None,
    }
    changes.update(values)
    result = vacancy.model_copy(update=changes)
    await repository.save(result)
    return result


async def test_sample_filters_before_limit_and_does_not_select_raw(session, vacancy, now):
    repo = VacancyRepository(session)
    await seed(repo, vacancy, now, archived=True, published_at=now)
    await seed(repo, vacancy, now, valid_until=now)
    await seed(repo, vacancy, now, published_at=now + timedelta(seconds=1))
    await seed(repo, vacancy, now, published_at=now - timedelta(days=31))
    first = await seed(repo, vacancy, now, published_at=now)
    second = await seed(repo, vacancy, now, published_at=now - timedelta(days=30))
    queries = []
    engine = session.bind.sync_engine

    def record(connection, cursor, statement, parameters, context, executemany):
        queries.append(statement)

    event.listen(engine, "before_cursor_execute", record)
    try:
        sample, truncated = await repo.skill_gap_sample(
            since=now - timedelta(days=30), until=now, limit=1
        )
    finally:
        event.remove(engine, "before_cursor_execute", record)
    assert [v.id for v in sample] == [first.id]
    assert truncated
    assert sample[0].published_at.tzinfo is not None
    assert "raw_payload" not in queries[0] and "description_html" not in queries[0]
    sample, truncated = await repo.skill_gap_sample(
        since=now - timedelta(days=30), until=now, limit=2
    )
    assert {v.id for v in sample} == {first.id, second.id}
    assert not truncated


async def test_analysis_window_uses_utc_for_expiry_with_local_offset(session, vacancy, now):
    repo = VacancyRepository(session)
    await seed(repo, vacancy, now, valid_until=now + timedelta(minutes=1))
    local = now.astimezone(timezone(timedelta(hours=11)))
    sample, _ = await repo.skill_gap_sample(since=local - timedelta(days=30), until=local)
    assert len(sample) == 1


async def test_skill_gap_unknown_experience_is_not_zero(session, vacancy, now):
    repo = VacancyRepository(session)
    await seed(repo, vacancy, now, experience="1-3")
    for years, insufficient, unknown in ((0, 1, 0), (None, 0, 1)):
        profile = candidate().model_copy(update={"experience_years": years})
        report = await analyze_skill_gap(repo, now=now, profile=profile)
        assert (report.insufficient_experience, report.unknown_experience) == (
            insufficient,
            unknown,
        )


@pytest.mark.parametrize(
    "position,title",
    [
        ("бухгалтер", "Помощник бухгалтера"),
        ("водитель", "Водителя автомобиля"),
        ("повар", "Помощник повара"),
        ("медсестра", "Медсестра процедурная"),
        ("Java Developer", "Senior Java разработчик"),
        ("менеджер по продажам", "Менеджер по продажам оборудования"),
    ],
)
async def test_directions_are_not_limited_to_it(session, vacancy, now, position, title):
    repo = VacancyRepository(session)
    await seed(repo, vacancy, now, title=title)
    await seed(repo, vacancy, now, title="Python Developer")
    report = await analyze_skill_gap(repo, now=now, profile=candidate(position))
    assert report.scope.vacancies == 1


async def test_skill_gap_fixed_requirements_simulation_and_separate_experience(
    session, vacancy, now
):
    repo = VacancyRepository(session)
    await seed(repo, vacancy, now, skills=["Excel", "Расчёт зарплат"], experience="3-6")
    await seed(repo, vacancy, now, skills=["Excel", "Расчёт зарплат", "Аудит"], experience=None)
    await seed(repo, vacancy, now, skills=["Excel"])
    await seed(repo, vacancy, now, skills=[])
    await seed(repo, vacancy, now, title="Java Developer", skills=["Java"])
    profile = candidate()
    original = profile.model_dump()
    report = await analyze_skill_gap(repo, now=now, profile=profile)
    assert report.scope.vacancies == 4 and report.scope.with_skills == 3
    assert report.mean_skill_coverage == pytest.approx((0.5 + 1 / 3 + 1) / 3)
    assert report.fully_covered == 1
    gap = report.opportunities[0]
    assert gap.skill == "расчёт зарплат" and gap.vacancies == 2
    assert gap.mean_coverage_gain_pp == 27.78 and gap.additional_fully_covered == 1
    assert report.insufficient_experience == 1 and report.unknown_experience == 1
    assert profile.model_dump() == original
    assert "не вероятности найма" in render_skill_gap(report)


async def test_gap_counts_known_skills_using_morphology_and_aliases(session, vacancy, now):
    repo = VacancyRepository(session)
    await seed(repo, vacancy, now, skills=["Расчёт зарплаты", "EXCEL"])
    report = await analyze_skill_gap(
        repo, now=now, profile=candidate(skills=("расчёт зарплат", "Excel"))
    )
    assert report.mean_skill_coverage == 1 and report.fully_covered == 1
    assert not report.opportunities


async def test_remote_required_excludes_unknown_but_preferred_keeps_it(session, vacancy, now):
    repo = VacancyRepository(session)
    for remote in (True, False, None):
        await seed(repo, vacancy, now, remote=remote)
    for mode, size in (("required", 1), ("preferred", 3), ("office", 3)):
        report = await analyze_skill_gap(repo, now=now, profile=candidate(remote=mode))
        assert report.scope.vacancies == size
        assert report.excluded_by_remote == 3 - size


async def test_unknown_skills_and_empty_selection_do_not_count_as_covered(session, vacancy, now):
    repo = VacancyRepository(session)
    await seed(repo, vacancy, now, skills=[])
    report = await analyze_skill_gap(repo, now=now, profile=candidate())
    assert report.mean_skill_coverage is None and report.fully_covered == 0
    assert "недостаточно" in render_skill_gap(report)
    empty = await analyze_skill_gap(repo, now=now, profile=candidate("повар"))
    assert empty.scope.vacancies == 0
    assert "недостаточно" in render_skill_gap(empty)


async def test_global_sample_truncation_is_visible(session, vacancy, now):
    repo = VacancyRepository(session)
    await seed(repo, vacancy, now, title="Бухгалтер", published_at=now - timedelta(days=2))
    await seed(repo, vacancy, now, title="Водитель")
    report = await analyze_skill_gap(repo, now=now, profile=candidate(), limit=1)
    assert report.scope.truncated and report.scope.examined == 1
    assert report.scope.vacancies == 0
    assert "ограничена" in render_skill_gap(report)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"days": 0},
        {"days": 366},
        {"limit": 0},
        {"limit": 10001},
        {},
        {"profile": CandidateProfile()},
        {"profile": CandidateProfile(skills={"Excel": {}})},
    ],
)
async def test_invalid_analysis_parameters(session, kwargs):
    with pytest.raises(ValueError):
        await analyze_skill_gap(VacancyRepository(session), **kwargs)


async def test_naive_analysis_time_rejected(session, now):
    with pytest.raises(ValueError):
        await analyze_skill_gap(VacancyRepository(session), now=now.replace(tzinfo=None))


async def test_telegram_reports_are_personal_and_require_registration(sessions, vacancy):
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    async with sessions.begin() as session:
        await seed(VacancyRepository(session), vacancy, now)
        await seed(
            VacancyRepository(session), vacancy, now, title="Java Developer", skills=["Spring"]
        )
    await store_profile(sessions, 42, candidate())
    await store_profile(sessions, 99, candidate("Java Developer", ("Java",)))
    transport = RecordingSession()
    bot = Bot(TEST_TOKEN, session=transport)
    dispatcher = build_dispatcher(UserService(sessions))
    for user_id, command in ((42, "/skillgap"), (99, "/skillgap"), (100, "/skillgap")):
        await dispatcher.feed_update(bot, command_update(command, user_id=user_id, chat_id=user_id))
    messages = [v for v in transport.requests if isinstance(v, SendMessage)]
    first = "\n".join(v.text for v in messages if v.chat_id == 42)
    second = "\n".join(v.text for v in messages if v.chat_id == 99)
    assert "бухгалтер" in first and "spring" not in first
    assert "spring" in second and "расчёт зарплат" not in second
    assert "/start" in next(v.text for v in messages if v.chat_id == 100)
    assert all(len(v.text) <= 3500 for v in messages)


async def test_cli_analysis_uses_same_calculation_and_json(
    session, vacancy, now, monkeypatch, capsys
):
    import jobs.cli as cli

    await seed(VacancyRepository(session), vacancy, now)
    await session.commit()
    monkeypatch.setattr(cli, "Settings", lambda: SimpleNamespace(database_url="unused"))
    monkeypatch.setattr(cli, "create_engine", lambda _: session.bind)
    monkeypatch.setattr(cli, "load_profile", lambda _: candidate())
    await run_command(parser().parse_args(["skill-gap", "--profile", "unused.yaml", "--json"]))
    import json

    result = json.loads(capsys.readouterr().out)
    assert result["scope"]["vacancies"] == 1
    assert result["opportunities"][0]["skill"] == "расчёт зарплат"


@pytest.mark.postgres
async def test_postgres_analysis_projection_and_gap(postgres_sessions, vacancy, now):
    async with postgres_sessions.begin() as session:
        repo = VacancyRepository(session)
        await seed(repo, vacancy, now)
        await seed(repo, vacancy, now, archived=True)
    async with postgres_sessions() as session:
        report = await analyze_skill_gap(VacancyRepository(session), now=now, profile=candidate())
    assert report.scope.vacancies == 1
    assert report.opportunities[0].skill == "расчёт зарплат"
