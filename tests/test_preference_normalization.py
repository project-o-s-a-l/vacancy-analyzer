from dataclasses import replace

import pytest
from aiogram import Bot
from test_bot import TEST_TOKEN, RecordingSession, command_update
from test_users import callback_update

from bot.handlers import build_dispatcher
from bot.resume_cards import render_resume, render_resume_details
from domain import CandidateProfile
from jobs.users import UserService
from nlp.professional_facts import candidate_facts
from nlp.resume_document import extract_document
from storage.repositories.user_repository import UserRepository

TEXT = (
    "Желаемая должность: Бухгалтер\nНавыки: Excel, 1С\nОбщий опыт: 3 года\n"
    "Обязательные условия: только Москва; зарплата от 60000 RUB в месяц до налога.\n"
)


def test_explicit_preferences_have_exact_evidence_and_no_fake_skills():
    extraction = extract_document(TEXT)
    items = {p.group: p for p in extraction.preferences}
    assert items["locations"].values.locations == ["Москва"]
    assert items["locations"].values.locations_required
    assert items["salary"].values.salary_min == 60000
    assert items["salary"].values.salary_required
    assert items["salary"].values.salary_gross
    assert (
        TEXT[items["salary"].start : items["salary"].end]
        == "зарплата от 60000 RUB в месяц до налога"
    )
    assert set(extraction.skills) == {"excel", "1с"}
    assert not any("Обязательные" in o.name for o in extraction.observations)


@pytest.mark.parametrize(
    "suffix",
    [
        "Проживаю в Москве. Раньше зарплата 80000 рублей.",
        "Зарплата: от 60000 RUB",  # No pay period or gross/net: do not invent comparability.
        "Обязательные условия: не только Москва; зарплата обсуждается.",
        "Опыт работы\nКомпания: Москва\nЗарплата: 100000 RUB",
    ],
)
def test_history_and_ambiguous_conditions_are_not_constraints(suffix):
    result = extract_document("Желаемая должность: Бухгалтер\nНавыки: Excel\n" + suffix)
    assert result.preferences == []


def test_conflicting_preferences_require_review():
    result = extract_document(TEXT + "Обязательные условия: только Казань.\n")
    assert [p.group for p in result.preferences] == ["salary"]
    assert any("Противоречивые" in note for note in result.notes)


@pytest.mark.parametrize(
    "suffix,expected",
    [
        ("Обязательное условие: только удалённая работа.", "required"),
        ("Пожелания: удаленная работа.", "preferred"),
    ],
)
def test_remote_explicit_import(suffix, expected):
    result = extract_document("Желаемая должность: Разработчик\n" + suffix)
    assert result.preferences[0].values.remote == expected


@pytest.fixture(params=["sessions", pytest.param("postgres_sessions", marks=pytest.mark.postgres)])
def preference_store(request):
    return request.getfixturevalue(request.param)


async def test_import_review_confirmation_edit_and_owner_isolation(preference_store):
    sessions = preference_store
    users = UserService(sessions)
    await users.register(42)
    await users.register(43)
    recording = RecordingSession()
    bot = Bot(TEST_TOKEN, session=recording)
    dispatcher = build_dispatcher(users)
    await dispatcher.feed_update(bot, command_update("/resume " + TEXT))
    draft = await users.resumes.pending(42)
    assert draft
    assert (await users.get(42)).profile.preferences.locations == []
    assert "60000" in render_resume(draft, (await users.get(42)).profile)
    assert "зарплата от 60000" in render_resume_details(draft)
    await dispatcher.feed_update(bot, callback_update("resume:confirm:" + draft.token))
    profile = (await users.get(42)).profile
    assert profile.preferences.locations_required and profile.preferences.salary_required
    fact = next(f for f in profile.professional_facts.facts if f.id == "preference:salary_min")
    assert fact.assertion.origin == "user_confirmed" and fact.assertion.evidence[0].quote
    assert (await users.get(43)).profile.preferences.salary_min is None
    # New imports preserve the atomic old salary/location group; explicit review can change it.
    draft = await users.resumes.prepare(
        42, TEXT.replace("Москва", "Казань").replace("60000", "90000")
    )
    assert "60000" in render_resume(draft, profile)
    draft = await users.resumes.revise(42, draft.token, "locations", "только Казань")
    draft = await users.resumes.revise(42, draft.token, "salary", "?")
    await users.resumes.confirm(42, draft.token)
    profile = (await users.get(42)).profile
    assert profile.preferences.locations == ["Казань"]
    assert profile.preferences.locations_required
    assert profile.preferences.salary_min is None and not profile.preferences.salary_required
    fact = next(f for f in profile.professional_facts.facts if f.id == "preference:locations")
    assert fact.assertion.value == ["Казань"]
    assert fact.assertion.evidence[0].quote is None


async def test_preexisting_nondefault_and_confirmed_empty_preferences_survive_merge(sessions):
    users = UserService(sessions)
    await users.register(42)
    profile = CandidateProfile(preferences={"locations": ["Казань"]})
    profile.professional_facts = candidate_facts(profile, 42, confirmed_fields={"salary"})
    async with sessions.begin() as session:
        repo = UserRepository(session)
        user = await repo.get(42)
        await repo.save(replace(user, profile=profile))
    draft = await users.resumes.prepare(42, TEXT)
    await users.resumes.confirm(42, draft.token)
    profile = (await users.get(42)).profile
    assert profile.preferences.locations == ["Казань"]
    assert not profile.preferences.locations_required
    assert profile.preferences.salary_min is None
    assert not profile.preferences.salary_required
