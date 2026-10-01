import hashlib
import json
from datetime import UTC, datetime, timedelta

import pytest
from aiogram import Bot
from aiogram.methods import SendMessage
from pydantic import ValidationError
from test_bot import TEST_TOKEN, RecordingSession, command_update
from test_users import callback_update, store_profile

from bot.cards import render_card, render_profile
from bot.handlers import build_dispatcher
from domain import CandidateProfile
from domain.qualifications import Qualification
from domain.vacancy_analysis import VacancyAnalysis, VacancyExtraction, analysis_input_hash
from jobs.cli import parser, render
from jobs.profiles import update_field
from jobs.rank import rank_recent
from jobs.resumes import profile_hash
from jobs.users import UserService
from nlp.experience_requirement import experience_requirement
from scoring import check_eligibility, rank_vacancies, score_vacancy
from storage.repositories import VacancyRepository


def job(vacancy, now, **updates):
    return vacancy.model_copy(
        update={
            "title": "Специалист",
            "description": "Требования: Python, FastAPI, SQL.",
            "description_html": None,
            "skills": ["Python", "FastAPI", "SQL"],
            "experience": None,
            "remote": True,
            "location": "Москва",
            "published_at": now,
            **updates,
        }
    )


def candidate(years=0, **preferences):
    return CandidateProfile(
        skills={"python": {}, "fastapi": {}, "sql": {}},
        experience_years=years,
        preferences={"positions": ["Python Backend Developer"], **preferences},
    )


def codes(result):
    return {item.code for item in result.exclusions}


def test_senior_role_is_blocked_even_when_score_passes(vacancy, now):
    source = "Для учебного примера требуется опыт коммерческой разработки на Python."
    v = job(
        vacancy,
        now,
        source_id="demo-senior",
        title="Senior Python Backend Developer",
        description="Требования:\nPython, FastAPI, SQL.\n" + source,
        experience="3-6",
    )
    v.analysis = VacancyAnalysis(
        model="qwen3:4b",
        model_digest="frozen-test",
        input_hash=analysis_input_hash(v),
        extraction=VacancyExtraction(
            category="it",
            specialization="Python backend",
            role_evidence=v.title,
            responsibilities=[],
            skills=[
                {"name": name, "importance": "required", "evidence": "Python, FastAPI, SQL."}
                for name in v.skills
            ],
            alternatives=[],
            experience_min_years=None,
            experience_evidence=source,
        ),
    )
    profile = candidate(remote="required")
    result = score_vacancy(v, profile, now=now)
    assert result.score >= 50
    assert not result.eligible
    assert {"experience_required", "seniority_mismatch"} <= codes(result.eligibility)
    assert result.eligibility.experience.minimum_years == 3
    assert {e.source for e in result.eligibility.experience.evidence} == {"hh", "description"}
    assert not rank_vacancies([v], profile, now=now)
    diagnostic = rank_vacancies([v], profile, now=now, include_ineligible=True)
    assert len(diagnostic) == 1 and "Не соответствует" in render(diagnostic, as_json=False)
    assert "Не соответствует" in render_card(v, result)


@pytest.mark.parametrize(
    "years,minimum,eligible",
    [
        (0, 3, False),
        (2, 3, True),
        (1, 3, False),
        (3, 3, True),
        (None, 3, True),
        (0, 0, True),
    ],
)
def test_experience_policy_and_one_year_tolerance(vacancy, now, years, minimum, eligible):
    text = f"Требования:\nОпыт работы от {minimum} лет." if minimum else "Без опыта работы."
    v = job(vacancy, now, description=text)
    result = check_eligibility(v, candidate(years), now=now)
    assert result.eligible is eligible
    if years is None:
        assert any(i.code == "candidate_experience_unknown" for i in result.uncertainties)
    if years == 2:
        assert any(i.code == "experience_gap" for i in result.uncertainties)


@pytest.mark.parametrize(
    "text,required,minimum",
    [
        ("Требования:\nКоммерческий опыт разработки.", True, None),
        ("Требования:\nОпыт работы от 3 лет.", True, 3),
        ("Requirements:\nAt least 2 years of work experience.", True, 2),
        ("Будет плюсом:\nОпыт работы от 3 лет.", None, None),
        ("О компании:\nНаш опыт работы составляет 20 лет.", None, None),
        ("У нас более 20 лет опыта работы.", None, None),
        ("Требования:\nБез опыта работы с Docker.", None, None),
        ("Требования:\nОпыт работы с Docker не требуется.", None, None),
        ("Требования:\nОпыт работы с Docker.", None, None),
        ("Нужен Python 3.12. Компания работает с 2001 года.", None, None),
    ],
)
def test_requirements_keep_origin_and_do_not_invent_tenure(vacancy, now, text, required, minimum):
    result = experience_requirement(job(vacancy, now, description=text))
    assert result.required is required and result.minimum_years == minimum


def test_hh_conflict_does_not_award_no_experience_score(vacancy, now):
    v = job(vacancy, now, experience="3-6", description="Без опыта работы.")
    result = score_vacancy(v, candidate(), now=now)
    assert result.eligible and result.eligibility.experience.conflict
    assert result.components["experience"] is None
    assert any(i.code == "experience_conflict" for i in result.eligibility.uncertainties)


@pytest.mark.parametrize("hh,conflict", [("1-3", False), ("3-6", True)])
def test_hh_range_is_not_conflicting_with_a_supported_inner_bound(vacancy, now, hh, conflict):
    v = job(vacancy, now, experience=hh, description="Требования:\nОпыт работы от 2 лет.")
    requirement = experience_requirement(v)
    assert requirement.minimum_years == 2 and requirement.conflict is conflict


def test_independent_technology_durations_do_not_become_one_total(vacancy, now):
    v = job(
        vacancy, now, description="Требования:\nОпыт работы с WMS от 3 лет, с Axelot от 1,5 лет."
    )
    requirement = experience_requirement(v)
    assert requirement.minimum_years is None and requirement.conflict
    assert requirement.evidence[0].source == "description"


@pytest.mark.parametrize(
    "title,excluded",
    [
        ("Senior Python Developer", True),
        ("Ведущий бухгалтер", True),
        ("Старший продавец", True),
        ("Руководитель отдела", True),
        ("Junior Python Developer", False),
        ("Staff Nurse", False),
        ("Senior Care Assistant", False),
        ("Ведущий мероприятий", False),
        ("Lead Generation Specialist", False),
    ],
)
def test_seniority_is_a_separate_role_signal_not_a_number(vacancy, now, title, excluded):
    result = check_eligibility(job(vacancy, now, title=title), candidate(), now=now)
    assert ("seniority_mismatch" in codes(result)) is excluded
    assert result.experience.minimum_years is None


@pytest.mark.parametrize(
    "mode,remote,eligible",
    [
        ("required", True, True),
        ("required", False, False),
        ("required", None, False),
        ("office", False, True),
        ("office", True, False),
        ("office", None, False),
        ("preferred", False, True),
        ("preferred", None, True),
        ("any", None, True),
    ],
)
def test_work_format_is_explicit(vacancy, now, mode, remote, eligible):
    assert (
        check_eligibility(
            job(vacancy, now, remote=remote), candidate(remote=mode), now=now
        ).eligible
        is eligible
    )


@pytest.mark.parametrize(
    "strict,location,remote,eligible",
    [
        (True, "Казань", False, False),
        (True, None, False, False),
        (False, "Казань", False, True),
        (False, None, False, True),
        (True, "г. МОСКВА", False, True),
        (True, "Казань", True, True),
    ],
)
def test_location_strictness_and_remote_exception(vacancy, now, strict, location, remote, eligible):
    profile = candidate(locations=["Москва"], locations_required=strict)
    assert (
        check_eligibility(
            job(vacancy, now, location=location, remote=remote), profile, now=now
        ).eligible
        is eligible
    )


@pytest.mark.parametrize(
    "strict,updates,eligible",
    [
        (False, {"salary_to": 100000, "salary_from": 90000}, True),
        (True, {"salary_to": 100000, "salary_from": 90000}, False),
        (True, {"salary_to": 200000, "salary_from": 100000}, True),
        (True, {"salary_to": None, "salary_from": 160000}, True),
        (True, {"salary_to": None, "salary_from": 100000}, False),
        (True, {"salary_gross": None}, False),
        (False, {"salary_gross": None}, True),
        (True, {"currency": "USD"}, False),
        (True, {"currency": "rub"}, True),
        (True, {"currency": "RUR"}, True),
        (True, {"salary_period": "HOUR"}, False),
        (True, {"salary_to": None, "salary_from": None}, False),
    ],
)
def test_salary_does_not_guess_tax_currency_or_upper_bound(vacancy, now, strict, updates, eligible):
    profile = candidate(salary_min=150000, salary_required=strict)
    v = job(vacancy, now, **updates)
    assert check_eligibility(v, profile, now=now).eligible is eligible


@pytest.mark.parametrize(
    "importance,status,expired,eligible",
    [
        ("Требования", "absent", False, False),
        ("Требования", "present", True, False),
        ("Требования", "present", False, True),
        ("Требования", None, False, True),
        ("Будет плюсом", "absent", False, True),
    ],
)
def test_required_documents_distinguish_missing_and_explicit_absence(
    vacancy, now, importance, status, expired, eligible
):
    v = job(
        vacancy,
        now,
        title="Повар",
        description=importance + ":\nМедкнижка обязательна."
        if importance == "Требования"
        else importance + ":\nМедкнижка.",
    )
    profile = candidate()
    if status:
        profile.qualifications = [
            Qualification(
                name="медкнижка",
                status=status,
                valid_until=now.date() - timedelta(days=1) if expired else None,
            )
        ]
    assert check_eligibility(v, profile, now=now).eligible is eligible


def test_preference_modes_clear_safely_and_legacy_profiles_load():
    profile = candidate(salary_min=150000, locations=["Москва"])
    assert not profile.preferences.salary_required and not profile.preferences.locations_required
    profile = update_field(profile, "constraints", "both")
    assert profile.preferences.salary_required and profile.preferences.locations_required
    assert "зарплата, города" in render_profile(profile)
    profile = update_field(profile, "salary", "пропустить")
    assert not profile.preferences.salary_required and profile.preferences.locations_required
    profile = update_field(profile, "locations", "пропустить")
    assert not profile.preferences.locations_required
    with pytest.raises(ValidationError):
        CandidateProfile(preferences={"salary_required": True})
    with pytest.raises(ValueError):
        update_field(CandidateProfile(), "constraints", "both")


def test_legacy_resume_profile_hash_survives_default_preferences():
    profile = candidate(salary_min=150000, locations=["Москва"])
    previous = profile.model_dump(mode="json")
    previous["preferences"].pop("salary_required")
    previous["preferences"].pop("locations_required")
    expected = hashlib.sha256(
        json.dumps(previous, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    assert profile_hash(profile) == expected
    strict = update_field(profile, "constraints", "both")
    assert profile_hash(strict) != expected


@pytest.mark.postgres
async def test_postgres_persists_constraints_and_excludes_ineligible_jobs(
    postgres_sessions, vacancy
):
    now = datetime.now(UTC)
    users = UserService(postgres_sessions, min_score=0)
    await users.register(42)
    for field, value in [
        ("skills", "Python, FastAPI, SQL"),
        ("experience", "0"),
        ("positions", "Python Backend Developer"),
        ("salary", "150000 RUB до налогов"),
        ("locations", "Москва"),
        ("constraints", "both"),
    ]:
        await users.begin_edit(42, field)
        await users.accept_answer(42, value)
    senior = job(
        vacancy,
        now,
        source_id="demo-senior",
        title="Senior Python Backend Developer",
        experience="3-6",
    )
    async with postgres_sessions.begin() as session:
        await VacancyRepository(session).save(senior)
    saved = await UserService(postgres_sessions).get(42)
    assert (
        saved.profile.preferences.salary_required and saved.profile.preferences.locations_required
    )
    assert not await (await users.notifications(42)).pending()


async def test_telegram_constraint_choice_is_persisted_and_user_isolated(sessions):
    users = UserService(sessions)
    await store_profile(sessions, 42, candidate(salary_min=150000, locations=["Москва"]))
    await store_profile(sessions, 99, candidate())
    transport = RecordingSession()
    bot = Bot(TEST_TOKEN, session=transport)
    dispatcher = build_dispatcher(users)
    await dispatcher.feed_update(bot, callback_update("profile:constraints"))
    await dispatcher.feed_update(bot, callback_update("answer:constraints:both"))
    saved = await UserService(sessions).get(42)
    assert (
        saved.profile.preferences.salary_required and saved.profile.preferences.locations_required
    )
    assert not (await users.get(99)).profile.preferences.salary_required
    assert any(
        isinstance(m, SendMessage) and "Обязательные условия: зарплата, города" in m.text
        for m in transport.requests
    )


async def test_telegram_and_cli_exclude_same_ineligible_vacancy(sessions, vacancy):
    now = datetime.now(UTC)
    v = job(vacancy, now, title="Senior Python Backend Developer", experience="3-6")
    profile = candidate()
    async with sessions.begin() as session:
        await VacancyRepository(session).save(v)
    users = UserService(sessions, min_score=0)
    await store_profile(sessions, 42, profile)
    transport = RecordingSession()
    bot = Bot(TEST_TOKEN, session=transport)
    await build_dispatcher(users).feed_update(bot, command_update("/latest"))
    assert not (await (await users.notifications(42)).pending())
    async with sessions() as session:
        assert not await rank_recent(VacancyRepository(session), profile)
        diagnostic = await rank_recent(VacancyRepository(session), profile, include_ineligible=True)
    assert len(diagnostic) == 1 and not diagnostic[0][1].eligible
    assert (
        parser()
        .parse_args(["rank", "--profile", "profile.yaml", "--include-ineligible"])
        .include_ineligible
    )
    assert not any(isinstance(m, SendMessage) and v.title in m.text for m in transport.requests)
