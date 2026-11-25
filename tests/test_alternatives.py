from datetime import UTC, datetime

import pytest
from aiogram import Bot
from aiogram.methods import SendMessage
from test_bot import TEST_TOKEN, RecordingSession, command_update
from test_requirements import job
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


def match(text, known, explicit=()):
    requirements = analyze_requirements(text, explicit, vocabulary=known)
    return requirements, match_skills(
        known,
        requirements.names("required"),
        requirements.names("preferred"),
        unspecified=requirements.names("unspecified"),
        alternatives=requirements.alternatives,
    )


@pytest.mark.parametrize(
    "text",
    [
        "Требования: Java или Kotlin",
        "Требуется Java либо Kotlin",
        "Requirements: Java or Kotlin",
        "Java required or Kotlin required",
        "Требования: опыт работы с Java или Kotlin",
        "Required: either Java or Kotlin",
    ],
)
def test_required_alternative_wordings(text):
    analysis = analyze_requirements(text, ["Java", "Kotlin"])
    assert len(analysis.alternatives) == 1
    assert analysis.alternatives[0].options == ["java", "kotlin"]
    assert analysis.alternatives[0].importance == "required"
    assert not analysis.names("required") and not analysis.names("unspecified")
    assert all(not item.counted_separately for item in analysis.skills)


@pytest.mark.parametrize("known", [("Java",), ("Kotlin",), ("Java", "Kotlin")])
def test_one_option_is_sufficient_without_double_weight(known):
    _, result = match("Требования: Java или Kotlin", known, ["Java", "Kotlin"])
    assert result.score == 1
    assert result.missing == ()
    assert result.alternatives[0].satisfied


def test_missing_group_is_one_condition_not_two_individual_skills():
    _, result = match("Требования: Java или Kotlin", ["SQL"])
    assert result.score == 0 and not result.missing and not result.matched
    group = result.alternatives[0]
    assert not group.satisfied and not group.matched_options
    assert group.options == ["java", "kotlin"]


def test_multiple_groups_weights_and_no_bonus_for_second_option():
    text = "Требования: Java или Kotlin\nБудет плюсом: Docker или Kubernetes"
    for known, expected in [(["Java"], 0.75), (["Java", "Kotlin"], 0.75), (["Java", "Docker"], 1)]:
        _, result = match(text, known)
        assert result.score == expected
        assert len(result.alternatives) == 2


@pytest.mark.parametrize("separator", ["; ", "\n", ", "])
def test_standalone_condition_is_preserved_next_to_group(separator):
    text = "Требования: Java или Kotlin" + separator + "SQL обязателен"
    analysis, result = match(text, ["Kotlin"])
    assert analysis.names("required") == ["sql"]
    assert result.score == 0.5 and result.missing == ("sql",)


def test_two_groups_can_be_separated_by_comma():
    analysis, result = match("Требования: Java или Kotlin, SQL или PostgreSQL", ["Kotlin", "SQL"])
    assert len(analysis.alternatives) == 2
    assert result.score == 1


@pytest.mark.parametrize(
    "text",
    [
        "Требования: Java или другой язык",
        "Требования: Java или НеизвестныйФреймворк",
        "Требования: Java или Kotlin и SQL",
        "Требования: Java и SQL или Kotlin",
        "Требования: Java или (Kotlin и SQL)",
        "Требования: Java, Kotlin или Python",
        "Требования: НеизвестныйФреймворк, Java или Kotlin",
        "Требования: Java или Kotlin, SQL",
        "Java required or Kotlin preferred",
    ],
)
def test_incomplete_or_ambiguous_syntax_is_not_falsely_satisfied(text):
    analysis, result = match(text, ["Java"])
    assert not analysis.alternatives and not result.alternatives
    assert analysis.warnings


def test_negative_and_context_sections_do_not_create_active_groups():
    for text in ("Не требуется Java или Kotlin", "Обязанности: работа с Java или Kotlin"):
        analysis, result = match(text, ["Java"])
        assert not analysis.alternatives and result.score is None


def test_unknown_importance_is_kept_for_bare_choice():
    analysis, result = match("Java или Kotlin", ["Kotlin"])
    assert analysis.alternatives[0].importance == "unspecified"
    assert result.score == 1


def test_aliases_collapse_to_single_skill_not_two_distinct_options():
    analysis, result = match("Требования: PostgreSQL или postgres", ["PSQL"])
    assert not analysis.alternatives
    assert analysis.names("required") == ["postgresql"]
    assert result.score == 1 and not analysis.warnings


def test_identical_groups_merge_evidence_and_keep_stronger_importance():
    analysis, result = match("Желательно: Kotlin или Java\nТребования: Java или Kotlin", ["Java"])
    assert len(analysis.alternatives) == 1 and len(result.alternatives) == 1
    group = analysis.alternatives[0]
    assert group.importance == "required" and len(group.evidence) == 2
    assert result.score == 1
    assert {item.alternative_id for item in group.evidence} == {group.id}


def test_tag_with_choice_does_not_add_a_spurious_standalone_requirement():
    analysis, result = match("Требования: Java или Kotlin", ["Java"], ["Java or Kotlin"])
    assert result.score == 1 and not result.missing
    assert not analysis.names("unspecified")


def test_simple_tag_choice_has_unknown_importance_and_respects_explicit_negation():
    analysis, result = match("", ["Java"], ["Java or Kotlin"])
    assert analysis.alternatives[0].importance == "unspecified" and result.score == 1
    analysis, result = match("Java или Kotlin не требуются", ["Java"], ["Java or Kotlin"])
    assert analysis.alternatives[0].importance == "not_required"
    assert result.score is None and not result.alternatives[0].counted_in_score


@pytest.mark.parametrize(
    "known,expected", [(["Java"], 1), (["Kotlin"], 0), (["Java", "Kotlin"], 1)]
)
def test_independent_required_skill_is_not_replaced_by_choice(known, expected):
    analysis, result = match("Java обязателен\nТребования: Java или Kotlin", known)
    assert analysis.names("required") == ["java"]
    assert result.score == expected
    assert not result.alternatives[0].counted_in_score


@pytest.mark.parametrize(
    "skills",
    [
        ("расчёт зарплат", "кадровое делопроизводство"),
        ("ведение путевых листов", "перевозка пассажиров"),
        ("приготовление блюд", "приготовление десертов"),
        ("реставрация струнных инструментов", "настройка роялей"),
    ],
)
def test_professional_phrases_outside_seed_dictionary_are_alternatives(skills):
    text = "Требования: " + " или ".join(skills)
    _, result = match(text, [skills[1]], skills)
    assert result.score == 1 and not result.missing
    assert len(result.alternatives[0].options) == 2


def test_contradictory_evidence_downgrades_group_importance():
    analysis, _ = match("Требования: Java или Kotlin\nJava не требуется", ["Kotlin"])
    assert analysis.alternatives[0].importance == "unspecified"
    assert analysis.warnings


def test_score_and_card_explain_group_without_missing_unchosen_option(vacancy, now):
    profile = CandidateProfile(skills={"Kotlin": {}})
    original = profile.model_dump()
    sample = job(vacancy, now, "Требования: Java или Kotlin", ["Java", "Kotlin"])
    result = score_vacancy(sample, profile, now=now)
    assert result.components["skills"] == 1
    assert not result.missing_required and not result.missing_unspecified
    assert result.matched_skills == ["kotlin"]
    assert result.alternatives[0].satisfied
    card = render_card(sample, result)
    assert "Нужен один из навыков: Java или Kotlin — в профиле есть Kotlin" in card
    assert not any("java" in value for value in result.warnings)
    missing = score_vacancy(sample, CandidateProfile(skills={"SQL": {}}), now=now)
    assert "Нужен один из навыков: Java или Kotlin — стоит уточнить" in render_card(sample, missing)
    assert missing.components["skills"] == 0
    assert profile.model_dump() == original


def test_alternative_card_with_large_labels_respects_utf16_limit(vacancy, now):
    sample = job(vacancy, now, "Требования: Java или Kotlin")
    result = score_vacancy(sample, CandidateProfile(), now=now)
    result.alternatives[0].options = ["😀" * 10000, "X" * 10000]
    result.matched_skills = ["😀" * 10000]
    result.missing_required = result.missing_preferred = result.missing_unspecified = ["😀" * 10000]
    result.reasons = result.warnings = ["😀" * 10000] * 10
    vacancy.title = vacancy.company = vacancy.location = "😀" * 10000
    assert len(render_card(vacancy, result).encode("utf-16-le")) // 2 < 4096


def test_v1_context_still_loads_with_defaults(vacancy, now):
    result = score_vacancy(job(vacancy, now, "Java обязателен"), CandidateProfile(), now=now)
    payload = result.model_dump()
    payload.pop("alternatives")
    payload["requirements_version"] = "requirements-v1"
    for item in payload["requirements"]:
        item.pop("counted_separately")
        for evidence in item["evidence"]:
            evidence.pop("alternative_id")
    loaded = MatchResult.model_validate(payload)
    assert loaded.alternatives == [] and loaded.requirements[0].counted_separately
    assert loaded.requirements[0].evidence[0].alternative_id is None


async def test_telegram_choice_and_delivery_context(sessions, vacancy):
    profile = CandidateProfile(skills={"Kotlin": {}})
    sample = job(vacancy, datetime.now(UTC), "Требования: Java или Kotlin")
    await store_profile(sessions, 42, profile)
    async with sessions.begin() as session:
        await VacancyRepository(session).save(sample)
    transport = RecordingSession()
    bot = Bot(TEST_TOKEN, session=transport)
    await build_dispatcher(UserService(sessions, min_score=0)).feed_update(
        bot, command_update("/latest")
    )
    assert any(
        "в профиле есть Kotlin" in v.text for v in transport.requests if isinstance(v, SendMessage)
    )
    async with sessions() as session:
        context = (await session.get(NotificationModel, (42, sample.id))).exposure_context["match"]
        assert context["requirements_version"] == RULES_VERSION
        assert context["alternatives"][0]["satisfied"]
        assert context["alternatives"][0]["matched_options"] == ["kotlin"]


@pytest.mark.postgres
async def test_postgres_choice_uses_current_profile_and_preserves_source(
    postgres_sessions, vacancy
):
    sample = job(vacancy, datetime.now(UTC), "Требования: Java или Kotlin", ["Java", "Kotlin"])
    original = sample.model_dump(mode="json", exclude={"professional_facts"})
    async with postgres_sessions.begin() as session:
        await VacancyRepository(session).save(sample)
    for known in ("Java", "Kotlin"):
        service = NotificationService(
            postgres_sessions, CandidateProfile(skills={known: {}}), chat_id=42, min_score=0
        )
        [(saved, result)] = await service.pending()
        assert saved.model_dump(mode="json", exclude={"raw_payload", "professional_facts"}) == {
            k: v for k, v in original.items() if k != "raw_payload"
        }
        async with postgres_sessions() as session:
            assert (await VacancyRepository(session).get_by_id(saved.id)).model_dump(
                mode="json", exclude={"professional_facts"}
            ) == original
        assert result.components["skills"] == 1 and not result.missing_skills
