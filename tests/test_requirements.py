import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from aiogram import Bot
from aiogram.methods import SendMessage
from test_bot import TEST_TOKEN, RecordingSession, command_update
from test_users import store_profile

from bot.cards import render_card
from bot.handlers import build_dispatcher
from domain import CandidateProfile
from jobs.notify import NotificationService
from jobs.users import UserService
from nlp.requirements import RULES_VERSION, analyze_requirements
from scoring.baseline import MatchResult, score_vacancy
from scoring.skill_score import match_skills
from storage.models import NotificationModel
from storage.repositories import VacancyRepository


def statuses(text, explicit=(), vocabulary=()):
    result = analyze_requirements(text, explicit, vocabulary=vocabulary)
    return {v.skill: v.importance for v in result.skills}


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Требования:\nJava\nБудет плюсом:\nDocker", {"java": "required", "docker": "preferred"}),
        (
            "<h3>Requirements</h3><ul><li>Java</li></ul><h3>Nice to have</h3><p>Docker</p>",
            {"java": "required", "docker": "preferred"},
        ),
        ("Java обязателен, Docker будет плюсом", {"java": "required", "docker": "preferred"}),
        ("Python, Java обязательны", {"python": "required", "java": "required"}),
        ("Желательно Python, Docker", {"python": "preferred", "docker": "preferred"}),
        ("Python is required; Docker is a plus", {"python": "required", "docker": "preferred"}),
        ("Java is essential. Docker preferred.", {"java": "required", "docker": "preferred"}),
        (
            "Требования: Python. Будет плюсом: Docker.",
            {"python": "required", "docker": "preferred"},
        ),
        (
            "Требования:\nPython\nУсловия:\nОбучение Docker",
            {"python": "required", "docker": "context"},
        ),
        (
            "Обязанности:\nРабота с Java\nМы предлагаем:\nОбучение Docker обязательно",
            {"java": "context", "docker": "context"},
        ),
        (
            "Требования:\nPython\nДополнительная информация:\nJava",
            {"python": "required", "java": "unspecified"},
        ),
        ("Наш стек:\nJava, Docker", {"java": "unspecified", "docker": "unspecified"}),
        ("Python обязателен\nDocker", {"python": "required", "docker": "unspecified"}),
        ("Обязательные навыки: Java\nSQL", {"java": "required", "sql": "required"}),
        ("Preferred qualifications: Java\nSQL", {"java": "preferred", "sql": "preferred"}),
    ],
)
def test_headers_inline_cues_and_boundaries(text, expected):
    assert statuses(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Java не требуется",
        "Знание Java не обязательно",
        "Java необязателен",
        "Без знания Java",
        "Опыт Java не нужен",
        "Java is not required",
        "No Java required",
        "We do not require Java",
        "We don't require Java",
        "No experience with Java required",
        "Java is not mandatory",
    ],
)
def test_negated_mentions_are_not_missing_requirements_even_with_tag(text):
    assert statuses(text, ["Java"]) == {"java": "not_required"}


def test_not_only_is_not_negation():
    assert statuses("Требования: не только Python, но и Java") == {
        "python": "required",
        "java": "required",
    }


@pytest.mark.parametrize(
    "text",
    [
        "Работа с Java при необходимости",
        "В обязательства компании входит обучение Java",
    ],
)
def test_similar_nouns_are_not_mandatory_cues(text):
    assert statuses(text) == {"java": "unspecified"}


def test_alternatives_do_not_turn_context_or_negation_into_requirements():
    assert statuses("Условия: обучение Python или Java") == {"python": "context", "java": "context"}
    assert statuses("Python или Java не требуются") == {
        "python": "not_required",
        "java": "not_required",
    }
    result = analyze_requirements("Требования: Python или другой язык")
    assert result.names("unspecified") == ["python"] and result.warnings


def test_unknown_html_headings_end_previous_section_and_keep_inline_markup():
    text = (
        "<h3>Требования</h3><p>Java</p><p><strong>Другой раздел без двоеточия</strong></p>"
        "<p>Docker</p><p><strong>Технический стек (обязательно)</strong></p>"
        "<p>PostgreSQL</p><p><strong>Что не входит в зону ответственности</strong></p>"
        "<p>Бизнес-анализ</p><h4>Условия</h4><p>Fast<b>API</b></p>"
    )
    assert statuses(text, ["Бизнес-анализ"]) == {
        "java": "required",
        "docker": "unspecified",
        "postgresql": "required",
        "бизнес-анализ": "not_required",
        "fastapi": "context",
    }


def test_candidate_requirements_and_technical_stack_headers():
    assert statuses(
        "Nice to Have:\nJava\nTechnical Stack:\nDocker\nCandidate Requirements:\nSQL"
    ) == {
        "java": "preferred",
        "docker": "unspecified",
        "sql": "required",
    }


def test_unlabelled_tags_stay_unknown_and_context_does_not_promote_them():
    result = analyze_requirements("Условия: обучим Java", ["Java", "Новый редкий инструмент"])
    assert result.names("unspecified") == ["java", "новый редкий инструмент"]
    java = result.skills[0]
    assert {v.origin for v in java.evidence} == {"tag", "description"}
    assert any(v.importance == "context" for v in java.evidence)


@pytest.mark.parametrize(
    "text",
    [
        "Требования: Java или другой язык",
        "Java required or Kotlin preferred",
        "Java обязателен и не требуется",
        "Java обязателен\nJava не нужен",
        "Java required and preferred",
        "Java обязателен, но не требуется",
        "Java не обязателен, но будет плюсом",
    ],
)
def test_alternatives_and_conflicts_remain_uncertain(text):
    result = analyze_requirements(text, ["Kotlin"] if "Kotlin" in text else [])
    assert all(v.importance == "unspecified" for v in result.skills)
    assert result.warnings


@pytest.mark.parametrize(
    "skill,phrase",
    [
        ("расчёт зарплат", "Требования: опыт расчёта зарплат"),
        ("ведение путевых листов", "Требования: ведение путевых листов"),
        ("приготовление блюд", "Обязательно: навыки приготовления блюд"),
        ("приём пациентов", "Необходим приём пациентов"),
        ("ведение переговоров", "Требования: навыки ведения переговоров"),
        ("реставрация струнных инструментов", "Обязательна реставрация струнных инструментов"),
    ],
)
def test_any_profession_and_arbitrary_terms_use_same_rules(skill, phrase):
    assert statuses(phrase, vocabulary=[skill]) == {skill: "required"}
    assert statuses("Будет плюсом: " + skill, [skill]) == {skill: "preferred"}


def test_skill_aliases_and_inflections_share_evidence():
    result = analyze_requirements(
        "Обязательно: опыт ведения учёта\nЖелательно: postgres", ["ведение учета", "PSQL"]
    )
    assert len(result.skills) == 2
    assert len(result.skills[1].evidence) == 2
    assert statuses("Требования: JavaScript, MySQL", ["Java"]) == {
        "java": "unspecified",
        "javascript": "required",
        "mysql": "required",
    }


def test_required_wins_over_preferred_and_weights_are_three_two_one():
    match = match_skills(["Java"], ["Java"], ["Docker", "Java"], unspecified=["SQL", "Docker"])
    assert match.score == pytest.approx(3 / 7)
    assert match.missing == ("docker", "sql")
    result = analyze_requirements("Желательно: Java\nОбязательно: Java")
    assert result.names("required") == ["java"]


def job(vacancy, now, description, explicit=()):
    return vacancy.model_copy(
        update={
            "id": uuid4(),
            "source_id": uuid4().hex,
            "description": description,
            "description_html": None,
            "skills": list(explicit),
            "published_at": now,
            "archived": False,
            "valid_until": None,
        }
    )


def test_priority_changes_score_and_provides_evidence_without_filtering(vacancy, now):
    profile = CandidateProfile(skills={"java": {}})
    required_java = job(vacancy, now, "Требования: Java\nБудет плюсом: Docker")
    required_docker = job(vacancy, now, "Требования: Docker\nБудет плюсом: Java")
    a = score_vacancy(required_java, profile, now=now)
    b = score_vacancy(required_docker, profile, now=now)
    assert a.components["skills"] == 0.75 and b.components["skills"] == 0.25
    assert a.score - b.score == 20
    assert a.missing_preferred == ["docker"] and not a.missing_required
    assert b.missing_required == ["docker"] and not b.missing_preferred
    assert a.eligible and b.eligible  # An incomplete profile is not proof of lacking the skill.
    assert a.requirements_version == RULES_VERSION
    assert any(v.section == "Требования" for v in a.requirements[1].evidence)
    assert "Желательно, нет в профиле: docker" in render_card(required_java, a)
    assert "Обязательно, нет в профиле: docker" in render_card(required_docker, b)


def test_unknown_priority_stays_unknown_and_negated_context_gives_no_points(vacancy, now):
    profile = CandidateProfile(skills={"java": {}})
    for text in ("Обязанности: Java", "Java не требуется"):
        match = score_vacancy(job(vacancy, now, text), profile, now=now)
        assert match.components["skills"] is None
        assert match.matched_skills == [] and match.missing_skills == []
    unknown = score_vacancy(job(vacancy, now, "Java, Docker"), profile, now=now)
    assert unknown.components["skills"] == 0.5
    assert unknown.missing_unspecified == ["docker"]
    assert "Важность не указана" in render_card(vacancy, unknown)


def test_new_match_fields_allow_old_stored_context(vacancy, profile, now):
    match = score_vacancy(vacancy, profile, now=now).model_dump()
    for field in (
        "requirements",
        "missing_required",
        "missing_preferred",
        "missing_unspecified",
        "requirements_version",
    ):
        del match[field]
    old = MatchResult.model_validate(match)
    assert old.requirements == [] and old.requirements_version is None


async def test_delivered_card_and_durable_context_include_requirement_evidence(sessions, vacancy):
    now = datetime.now(UTC)
    profile = CandidateProfile(skills={"java": {}})
    sample = job(vacancy, now, "Требования: Java\nБудет плюсом: Docker")
    await store_profile(sessions, 42, profile)
    async with sessions.begin() as session:
        await VacancyRepository(session).save(sample)
    transport = RecordingSession()
    bot = Bot(TEST_TOKEN, session=transport)
    service = UserService(sessions, min_score=0, top=1)
    await build_dispatcher(service).feed_update(bot, command_update("/latest"))
    cards = [v.text for v in transport.requests if isinstance(v, SendMessage)]
    assert any("Желательно, нет в профиле: docker" in text for text in cards)
    async with sessions() as session:
        notification = await session.get(NotificationModel, (42, sample.id))
        context = notification.exposure_context["match"]
        assert context["requirements_version"] == RULES_VERSION
        assert context["missing_preferred"] == ["docker"]
        assert context["requirements"]


@pytest.mark.postgres
async def test_postgres_profile_terms_and_original_description_remain_unchanged(
    postgres_sessions, vacancy
):
    now = datetime.now(UTC)
    profile = CandidateProfile(skills={"ведение учёта": {}})
    sample = job(
        vacancy,
        now,
        "Требования: ведение учёта\nБудет плюсом: кадровое делопроизводство",
        explicit=["ведение учёта", "кадровое делопроизводство"],
    )
    original = sample.model_dump(mode="json", exclude={"professional_facts"})
    async with postgres_sessions.begin() as session:
        await VacancyRepository(session).save(sample)
    service = NotificationService(postgres_sessions, profile, chat_id=42, min_score=0)
    items = await service.pending()
    assert len(items) == 1
    saved, match = items[0]
    assert saved.model_dump(mode="json", exclude={"raw_payload", "professional_facts"}) == {
        k: v for k, v in original.items() if k != "raw_payload"
    }
    async with postgres_sessions() as session:
        assert (await VacancyRepository(session).get_by_id(saved.id)).model_dump(
            mode="json", exclude={"professional_facts"}
        ) == original
    assert match.missing_preferred == ["кадровое делопроизводство"]
    assert json.loads(match.model_dump_json())["requirements_version"] == RULES_VERSION
