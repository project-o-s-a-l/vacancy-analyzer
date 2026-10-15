from datetime import UTC, date, datetime, timedelta

import pytest
from aiogram import Bot
from aiogram.methods import SendMessage
from pydantic import ValidationError
from test_bot import TEST_TOKEN, RecordingSession, command_update
from test_requirements import job
from test_users import callback_update, store_profile, text_update

from bot.cards import render_card, render_profile
from bot.handlers import build_dispatcher
from domain import CandidateProfile
from domain.qualifications import Qualification
from jobs.notify import NotificationService
from jobs.profiles import update_field
from jobs.users import UserService
from nlp.qualifications import compare_qualifications
from scoring.baseline import MatchResult, score_vacancy
from storage.models import NotificationModel
from storage.repositories import VacancyRepository

TODAY = date(2026, 10, 17)


def compare(text, *names, explicit=()):
    return compare_qualifications(
        text, [Qualification(name=name) for name in names], today=TODAY, explicit=explicit
    )[0]


@pytest.mark.parametrize(
    "text,name",
    [
        ("Требования: водительские права категории B", "права B"),
        ("Требования: водительское удостоверение категории В", "права B"),
        ("Требования: права B", "водительские права категории В"),
        ("Requirements: driving licence category B", "права B"),
        ("Требования: медицинская книжка", "медкнижка"),
        ("Обязательно: удостоверение стропальщика", "Удостоверение стропальщика"),
        (
            "Требования: допуск по электробезопасности III группы",
            "допуск по электробезопасности III группы",
        ),
        ("Required: certificate AWS Solutions Architect", "certificate AWS Solutions Architect"),
    ],
)
def test_explicit_documents_match_self_report_without_becoming_verified(text, name):
    [result] = compare(text, name)
    assert result.status == "declared_present" and result.importance == "required"
    assert result.evidence
    assert all("qualification_" not in evidence.text for evidence in result.evidence)


@pytest.mark.parametrize("name", ["права C", "права BE", "права B1"])
def test_categories_do_not_imply_other_categories(name):
    [result] = compare("Требования: водительское удостоверение категории B", name)
    assert result.status == "unknown"


def test_missing_document_is_unknown_not_absent():
    [result] = compare("Требования: медкнижка")
    assert result.status == "unknown" and result.profile_name is None


@pytest.mark.parametrize(
    "status,expires,expected",
    [
        ("present", None, "declared_present"),
        ("present", TODAY, "declared_present"),
        ("present", TODAY - timedelta(days=1), "expired"),
        ("present", TODAY + timedelta(days=1), "declared_present"),
        ("absent", None, "declared_absent"),
    ],
)
def test_expiry_and_explicit_absence_are_separate(status, expires, expected):
    [result], _ = compare_qualifications(
        "Требования: медкнижка",
        [Qualification(name="медкнижка", status=status, valid_until=expires)],
        today=TODAY,
    )
    assert result.status == expected


@pytest.mark.parametrize(
    "text,importance,status",
    [
        ("Медкнижка не требуется", "not_required", "not_required"),
        ("Обязанности: проверять медицинские книжки", "context", "context"),
        ("Будет плюсом: медкнижка", "preferred", "unknown"),
        ("Требования: медкнижка\nУсловия: поможем оформить сертификат AWS", "required", "unknown"),
    ],
)
def test_importance_negation_and_context(text, importance, status):
    results = compare(text)
    result = next(value for value in results if value.kind == "medical_book")
    assert (result.importance, result.status) == (importance, status)


@pytest.mark.parametrize(
    "text",
    [
        "Требования: водительские права категории B, C",
        "Требования: права B, C",
        "Требования: водительское удостоверение",
        "Требования: медкнижка или сертификат AWS",
        "Требования: сертификат",
    ],
)
def test_ambiguous_qualifications_require_manual_review(text):
    results = compare(text, "права B", "права C", "медкнижка", "сертификат AWS")
    assert results and all(result.status == "uncertain" for result in results)


def test_generic_certificate_does_not_match_unrelated_certificate():
    [result] = compare("Требования: сертификат AWS Solutions Architect", "сертификат Azure")
    assert result.status == "unknown"


@pytest.mark.parametrize(
    "text",
    [
        "Допускается работа без опыта",
        "Водительские правила нужно знать",
        "Сертификатная программа обучения",
        "Employees are permitted to work remotely",
    ],
)
def test_similar_words_are_not_documents(text):
    assert not compare(text)


def test_engineering_tolerances_remain_skills_and_not_work_permits(vacancy, now):
    profile = CandidateProfile(skills={"расчёт допусков": {}, "допуски и посадки": {}})
    sample = job(vacancy, now, "Требования: расчёт допусков; допуски и посадки")
    result = score_vacancy(sample, profile, now=now)
    assert not result.qualifications and result.components["skills"] == 1


@pytest.mark.parametrize(
    "text",
    [
        "Лицензии на IDE - компенсируем",
        "Подарочные сертификаты на услуги красоты",
        "Возможность получения профессиональных сертификатов за счёт компании",
        "Мы поможем получить сертификат AWS",
    ],
)
def test_benefits_and_software_licenses_are_not_personal_qualification_requirements(text):
    results = compare(text)
    assert results and all(result.status == "context" for result in results)


def test_employer_benefit_does_not_erase_explicit_document_requirement():
    [result] = compare("Сертификат AWS обязателен, обучение за счёт компании")
    assert result.importance == "required" and result.status == "unknown"


def test_tag_is_not_mandatory_and_does_not_overwrite_description():
    [result] = compare("", "медкнижка", explicit=["медицинская книжка"])
    assert result.importance == "unspecified" and result.status == "declared_present"
    assert result.evidence[0].origin == "tag"
    [result] = compare("Требования: медкнижка", "медкнижка", explicit=["медицинская книжка"])
    assert result.importance == "required" and result.status == "declared_present"
    assert {value.origin for value in result.evidence} == {"tag", "description"}


def test_conflicting_declarations_and_requirement_evidence_are_not_accepted():
    entries = [
        Qualification(name="медкнижка"),
        Qualification(name="медицинская книжка", status="absent"),
    ]
    [result], _ = compare_qualifications("Требования: медкнижка", entries, today=TODAY)
    assert result.status == "uncertain"
    [result] = compare("Требования: медкнижка\nМедкнижка не требуется", "медкнижка")
    assert result.status == "uncertain" and result.importance == "unspecified"


def test_more_than_ten_mentions_preserve_original_evidence():
    text = "Требования:\n" + "\n".join(
        f"Удостоверение специальности {index}" for index in range(12)
    )
    results = compare(text)
    assert len(results) == 12
    for result in results:
        assert result.name.casefold() in result.evidence[0].text.casefold()
        assert "qualification_" not in result.evidence[0].text


def test_profile_form_has_present_absent_and_optional_dates():
    profile = update_field(
        CandidateProfile(),
        "qualifications",
        "права В; медкнижка до 31.12.2027; нет: удостоверение стропальщика",
    )
    assert profile.qualifications[0].name == "водительские права категории B"
    assert profile.qualifications[1].valid_until == date(2027, 12, 31)
    assert profile.qualifications[2].status == "absent"
    text = render_profile(profile)
    assert "Документы и допуски" in text and "с ваших слов" in text and "указано: нет" in text
    assert not update_field(profile, "qualifications", "пропустить").qualifications


def test_quantity_in_qualification_name_is_not_misread_as_expiry():
    profile = update_field(CandidateProfile(), "qualifications", "допуск к установкам до 1000В")
    assert profile.qualifications[0].valid_until is None
    [result] = compare("Требования: допуск к установкам до 1000В", profile.qualifications[0].name)
    assert result.status == "declared_present"


@pytest.mark.parametrize(
    "text",
    [
        "нет:",
        ";;",
        "права B; водительское удостоверение категории В",
        "медкнижка; медицинская книжка",
        "медкнижка до 31.02.2027",
        "медкнижка до завтра",
        "нет: медкнижка до 2027-01-01",
        "X" * 161,
    ],
)
def test_invalid_or_duplicate_documents_do_not_silently_change_profile(text):
    profile = CandidateProfile()
    with pytest.raises((ValueError, ValidationError)):
        update_field(profile, "qualifications", text)
    assert not profile.qualifications


def test_legacy_profile_and_result_load_without_documents(vacancy, now):
    profile = CandidateProfile.model_validate({"skills": {"Java": {}}})
    assert not profile.qualifications
    payload = score_vacancy(vacancy, profile, now=now).model_dump()
    payload.pop("qualifications")
    payload.pop("qualifications_version")
    assert not MatchResult.model_validate(payload).qualifications


def test_document_is_not_proved_by_skill_and_does_not_change_eligibility(vacancy, now):
    sample = job(vacancy, now, "Требования: сертификат AWS", ["сертификат AWS"])
    profile = CandidateProfile(skills={"AWS": {}, "сертификат AWS": {}})
    original = profile.model_dump()
    result = score_vacancy(sample, profile, now=now)
    assert result.components["skills"] is None
    assert result.qualifications[0].status == "unknown" and result.eligible
    profile.qualifications = [Qualification(name="сертификат AWS")]
    declared = score_vacancy(sample, profile, now=now)
    assert declared.score == result.score and declared.eligible
    assert declared.qualifications[0].status == "declared_present"
    assert "без проверки" in render_card(sample, declared)
    assert original["qualifications"] == []


def test_full_card_and_profile_with_documents_remain_bounded(vacancy, now):
    profile = CandidateProfile(skills={"Java": {}})
    sample = job(
        vacancy,
        now,
        "Требования: Java или Kotlin; медкнижка; удостоверение стропальщика; "
        "допуск к работе на высоте",
    )
    result = score_vacancy(sample, profile, now=now)
    assert len(result.qualifications) == 3 and result.alternatives
    for document in result.qualifications:
        document.name = "😀" * 10000
    result.matched_skills = ["😀" * 10000]
    result.missing_required = result.missing_preferred = result.missing_unspecified = ["😀" * 10000]
    result.reasons = result.warnings = ["😀" * 10000] * 10
    sample.title = sample.company = sample.location = "😀" * 10000
    assert len(render_card(sample, result).encode("utf-16-le")) // 2 < 4096
    profile.qualifications = [Qualification(name="😀" * 160) for _ in range(20)]
    profile.skills = {"😀" * 2000: {}}
    profile.preferences.positions = profile.preferences.locations = ["😀" * 10000]
    assert len(render_profile(profile).encode("utf-16-le")) // 2 < 4096


async def test_edit_documents_persists_and_isolates_users_without_extending_setup(sessions):
    users = UserService(sessions)
    await store_profile(sessions, 42, CandidateProfile(skills={"Java": {}}))
    await store_profile(sessions, 99, CandidateProfile(skills={"Python": {}}))
    await users.begin_edit(42, "qualifications")
    await users.accept_answer(42, "медкнижка до 2027-12-31")
    saved = await UserService(sessions).get(42)
    assert saved.form_state is None and saved.profile.qualifications[0].valid_until == date(
        2027, 12, 31
    )
    assert not (await users.get(99)).profile.qualifications
    assert set(saved.profile.skills) == {"Java"}


async def test_bot_documents_button_and_delivery_context(sessions, vacancy):
    users = UserService(sessions, min_score=0)
    await store_profile(sessions, 42, CandidateProfile(skills={"Java": {}}))
    transport = RecordingSession()
    bot = Bot(TEST_TOKEN, session=transport)
    dispatcher = build_dispatcher(users)
    await dispatcher.feed_update(bot, callback_update("profile:qualifications"))
    await dispatcher.feed_update(bot, text_update("права B"))
    assert (await users.get(42)).profile.qualifications
    sample = job(vacancy, datetime.now(UTC), "Требования: права B\nJava")
    async with sessions.begin() as session:
        await VacancyRepository(session).save(sample)
    await dispatcher.feed_update(bot, command_update("/latest"))
    assert any(
        "есть со слов пользователя" in item.text
        for item in transport.requests
        if isinstance(item, SendMessage)
    )
    async with sessions() as session:
        context = (await session.get(NotificationModel, (42, sample.id))).exposure_context
        assert context["profile"]["qualifications"][0]["name"] == "водительские права категории B"
        assert context["match"]["qualifications_version"] == "qualifications-v1"
        assert context["match"]["qualifications"][0]["status"] == "declared_present"


@pytest.mark.postgres
async def test_postgres_document_round_trip_and_read_only_matching(postgres_sessions, vacancy):
    profile = update_field(
        CandidateProfile(skills={"Java": {}}), "qualifications", "права B; нет: медкнижка"
    )
    await store_profile(postgres_sessions, 42, profile)
    sample = job(vacancy, datetime.now(UTC), "Требования: права B; медкнижка\nJava")
    async with postgres_sessions.begin() as session:
        await VacancyRepository(session).save(sample)
    stored = await UserService(postgres_sessions).get(42)
    assert stored.profile.model_copy(update={"professional_facts": None}) == profile
    assert (
        await NotificationService(
            postgres_sessions, stored.profile, chat_id=42, min_score=0
        ).pending()
        == []
    )
    # rejects an explicitly absent mandatory medical document in delivery.
    result = score_vacancy(sample, stored.profile)
    assert result.eligible is False
    assert {item.status for item in result.qualifications} == {
        "declared_present",
        "declared_absent",
    }
    assert (await UserService(postgres_sessions).get(42)).profile.model_copy(
        update={"professional_facts": None}
    ) == profile
