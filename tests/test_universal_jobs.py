from datetime import UTC, datetime

import pytest

from bot.cards import render_profile
from bot.keyboards import form_keyboard
from collectors.base import SearchQuery
from domain import CandidateProfile, Vacancy
from jobs.collection_scheduler import profile_queries
from jobs.notify import NotificationService
from jobs.profiles import update_field
from nlp.extraction import canonical_skill, extract_skills
from nlp.skills import SKILLS
from nlp.titles import title_tokens
from scoring.baseline import score_vacancy
from scoring.skill_score import match_skills
from storage.repositories import VacancyRepository
from storage.repositories.user_repository import RegisteredUser, UserRepository

CASES = [
    (
        "Бухгалтер",
        ["ведение бухгалтерского учёта", "Excel"],
        "Опыт ведения бухгалтерского учета. Excel.",
    ),
    (
        "Водитель",
        ["перевозка грузов", "ведение путевых листов"],
        "Опыт перевозки грузов и ведения путевых листов.",
    ),
    (
        "Повар",
        ["приготовление блюд", "соблюдение санитарных норм"],
        "Навыки приготовления блюд и соблюдения санитарных норм.",
    ),
    (
        "Врач-терапевт",
        ["ведение медицинской документации", "приём пациентов"],
        "Опыт ведения медицинской документации и приема пациентов.",
    ),
    (
        "Менеджер по продажам",
        ["работа с клиентами", "ведение переговоров"],
        "Опыт работы с клиентами и ведения переговоров.",
    ),
    (
        "Реставратор музыкальных инструментов",
        ["реставрация струнных инструментов", "настройка роялей"],
        "Опыт реставрации струнных инструментов и настройки роялей.",
    ),
]


def candidate(position, skills):
    profile = update_field(CandidateProfile(), "skills", ", ".join(skills))
    profile = update_field(profile, "positions", position)
    profile = update_field(profile, "experience", "3")
    return update_field(profile, "remote", "на месте работы")


def sample_vacancy(source_id, position, description, *, skills=()):
    return Vacancy(
        source="sample",
        source_id=str(source_id),
        url=f"https://example.org/jobs/{source_id}",
        title=position,
        company="Пример работодателя",
        description=description,
        skills=list(skills),
        experience="1-3",
        remote=False,
        published_at=datetime.now(UTC),
        raw_payload={},
    )


@pytest.mark.parametrize("position,skills,description", CASES, ids=[case[0] for case in CASES])
def test_any_profession_matches_profile_terms_without_registering_dictionary(
    position, skills, description
):
    assert all(canonical_skill(skill) not in SKILLS for skill in skills)
    profile = candidate(position, skills)
    queries = profile_queries([profile], SearchQuery(text="template"))
    assert len(queries) == 1 and queries[0].text == position and queries[0].search_field == "all"
    vacancy = sample_vacancy("123", position, description)
    match = score_vacancy(vacancy, profile)
    assert match.components["skills"] == 1
    assert set(match.matched_skills) == {canonical_skill(skill) for skill in skills}
    assert match.components["title"] is None and match.score >= 50
    assert any("полный список требований не подтверждён" in warning for warning in match.warnings)
    unrelated = vacancy.model_copy(
        update={"title": "Другая специальность", "description": "Совсем другие задачи"}
    )
    different = score_vacancy(unrelated, profile)
    assert different.matched_skills == [] and different.score < match.score


def test_explicit_requirements_in_different_cases_match_without_losing_unknown_requirements():
    result = match_skills(["ведение учёта"], ["ведения учета", "кадровое делопроизводство"])
    assert result.score == 0.5
    assert result.matched == ("ведения учета",)
    assert result.missing == ("кадровое делопроизводство",)


def test_arbitrary_skill_phrases_obey_word_boundaries_and_escape_punctuation():
    assert extract_skills("Магазин и Excellent", vocabulary=["газ", "Excel"]) == []
    assert extract_skills("Работа в 1С:Предприятие", vocabulary=["1С:Предприятие"]) == [
        "1с:предприятие"
    ]
    assert match_skills(["###"], ["!!!"]).score == 0


@pytest.mark.parametrize(
    "left,right",
    [
        ("водителя", "водитель"),
        ("бухгалтера", "бухгалтер"),
        ("повара", "повар"),
        ("врача-терапевта", "врач-терапевт"),
    ],
)
def test_titles_use_generic_russian_inflections(left, right):
    assert title_tokens(left) == title_tokens(right)


def test_job_format_is_not_limited_to_office_workers():
    profile = update_field(CandidateProfile(), "remote", "на месте работы")
    assert profile.preferences.remote == "office"
    assert "на месте работы" in render_profile(profile)
    assert update_field(profile, "remote", "офис").preferences.remote == "office"
    assert any(
        button.text == "На месте работы"
        for row in form_keyboard("remote").inline_keyboard
        for button in row
    )


@pytest.mark.postgres
async def test_six_professions_share_storage_and_receive_their_own_recommendations(
    postgres_sessions,
):
    async with postgres_sessions.begin() as session:
        users = UserRepository(session)
        vacancies = VacancyRepository(session)
        for index, (position, skills, description) in enumerate(CASES, start=1):
            profile = candidate(position, skills)
            await users.register(index)
            await users.save(RegisteredUser(index, profile, None))
            vacancy = sample_vacancy(index, position, description, skills=skills)
            await vacancies.save(vacancy)
    for index, (position, skills, _) in enumerate(CASES, start=1):
        service = NotificationService(
            postgres_sessions, candidate(position, skills), chat_id=index, min_score=50, top=1
        )
        pending = await service.pending()
        assert len(pending) == 1
        assert pending[0][0].title == position
        assert pending[0][1].components["skills"] == 1
