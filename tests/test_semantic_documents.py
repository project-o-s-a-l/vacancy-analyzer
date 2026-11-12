"""text contracts; these are engineering checks, not encoder quality labels."""

from datetime import UTC, datetime

import pytest

from domain import CandidateProfile
from domain.candidate import Skill
from domain.professional import Claim
from domain.resume import RESUME_VERSION, ResumeDraft
from jobs.resumes import profile_hash, proposed_profile
from jobs.semantic_documents import profile_document, vacancy_document
from nlp.professional_facts import (
    candidate_facts,
    confirm_resume_facts,
    resume_facts,
    vacancy_facts,
)
from nlp.resume_document import extract_document

AT = datetime(2026, 9, 1, tzinfo=UTC)


def confirmed(text):
    initial = CandidateProfile()
    extraction = extract_document(text, reference_time=AT)
    draft = ResumeDraft(
        token="0" * 32,
        text=text,
        format="text",
        extraction=extraction,
        profile_hash=profile_hash(initial),
        professional_facts=resume_facts(text, extraction, 42, version=RESUME_VERSION),
    )
    profile = proposed_profile(initial, draft)
    profile.professional_facts = confirm_resume_facts(profile, draft, 42)
    return profile


def card(vacancy, text):
    return vacancy.model_copy(
        update={
            "description": text,
            "description_html": None,
            "skills": [],
            "experience": None,
            "professional_facts": None,
        }
    )


def test_confirmed_profile_retains_tasks_roles_and_scope_without_contacts():
    profile = confirmed(
        "Имя: Private Person\nПочта: private@example.com\n"
        "Желаемая должность: Data engineer\nНавыки: Python — 3 года advanced, SQL\n"
        "Опыт работы\n01.2020 — 12.2021\nДолжность: Аналитик\n"
        "Задачи: строить хранилище данных\n"
        "Учебные проекты\n01.2022 — 12.2022\nПроект: прогнозирование спроса"
    )
    document = profile_document(profile, owner_id=42)
    assert document.kind == "query" and document.owner_id == 42
    assert "Data engineer" in document.text and "Аналитик" in document.text
    assert "строить хранилище данных" in document.text
    assert "Candidate experience (work)" in document.text
    assert "academic project" in document.text
    assert "years of use: 3.0" in document.text
    assert "Candidate desired role" in document.text and "Held role" in document.text
    for forbidden in ("Private", "example.com", "Имя", "Почта"):
        assert forbidden not in document.text


def test_flat_confirmed_profile_is_supported_without_inventing_skill_tenure():
    profile = CandidateProfile.model_validate(
        {
            "skills": {"Python": {"years": 4, "level": "advanced"}},
            "experience_years": 7,
            "preferences": {"positions": ["Data engineer"], "locations": ["Private City"]},
        }
    )
    document = profile_document(profile, owner_id=42)
    assert "Python" in document.text and "Data engineer" in document.text
    assert "all roles): 7.0 years" in document.text
    assert "years of use" not in document.text and "advanced" not in document.text
    assert "Private City" not in document.text
    assert document == profile_document(profile, owner_id=42)


def test_unconfirmed_and_conflicting_facts_never_become_asserted_experience():
    profile = confirmed("Навыки: Python, SQL\nЖелаемая должность: Инженер")
    for fact in profile.professional_facts.facts:
        if fact.kind == "skill" and fact.name == "python":
            fact.assertion.origin = "machine_proposed"
        if fact.kind == "skill" and fact.name == "sql":
            fact.assertion = Claim[bool](status="conflict", alternatives=[True, False])
    document = profile_document(profile)
    assert "python" not in document.text and "sql" not in document.text
    assert "unconfirmed_or_unresolved_fact" in document.exclusions
    with pytest.raises(ValueError, match="owner"):
        profile_document(profile, owner_id=43)


def test_source_edit_cannot_reuse_stale_rich_profile_tasks():
    profile = confirmed(
        "Навыки: Python\nОпыт работы\n01.2020 — 12.2021\n"
        "Должность: Инженер\nЗадачи: секретная старая задача"
    )
    old = profile_document(profile)
    profile.skills = {"Rust": Skill()}
    current = profile_document(CandidateProfile.model_validate(profile.model_dump()))
    assert "старая задача" in old.text and "старая задача" not in current.text
    assert "Rust" in current.text and old.input_revision != current.input_revision
    assert "stale_rich_facts_omitted" in current.exclusions


def test_vacancy_preserves_tasks_scoped_required_preferred_and_alternatives(vacancy):
    vacancy = card(
        vacancy,
        "Требования:\nPython или Java обязательны.\nОпыт с Python от 2 лет.\n"
        "Желательно:\nОпыт с SQL от 1 года.\n"
        "Обязанности:\nРазрабатывать аналитическое хранилище.\n",
    )
    document = vacancy_document(vacancy)
    assert document.kind == "passage" and document.owner_id is None
    assert "Vacancy task to perform: Разрабатывать аналитическое хранилище" in document.text
    assert "Vacancy alternatives (any one)" in document.text and " OR " in document.text
    assert "minimum 2.0 years in this skill" in document.text
    assert "Preferred skill: sql" in document.text
    assert "Candidate" not in document.text


def test_unknown_not_required_and_conflict_do_not_reenter_through_source_statements(vacancy):
    vacancy = card(
        vacancy,
        "Требования:\nPython обязателен.\nPython не требуется.\n"
        "Java не требуется.\nSQL обязателен.\n",
    )
    vacancy.title = "Инженер данных"
    document = vacancy_document(vacancy)
    assert "python" not in document.text.lower()
    assert "java" not in document.text.lower()
    assert "sql" in document.text.lower()
    assert "unresolved_or_not_required_requirement" in document.exclusions


def test_rare_requirement_phrase_and_english_task_survive_without_fixed_skill_alias(vacancy):
    vacancy = card(
        vacancy,
        "Requirements:\nFluxCapacitor knowledge required.\n"
        "Responsibilities:\nMaintain distributed event ingestion pipelines.\n",
    )
    document = vacancy_document(vacancy)
    assert "FluxCapacitor" in document.text
    assert "distributed event ingestion pipelines" in document.text


def test_long_task_evidence_tail_is_not_silently_truncated(vacancy):
    task = "Обрабатывать " + "очень подробные профессиональные данные " * 40 + "TAIL_SKILL."
    vacancy = card(vacancy, "Обязанности:\n" + task)
    vacancy.professional_facts = vacancy_facts(vacancy)
    task_fact = next(f for f in vacancy.professional_facts.facts if f.kind == "task")
    assert "TAIL_SKILL" not in task_fact.name
    document = vacancy_document(vacancy)
    assert "TAIL_SKILL" in document.text
    assert len(document.text) > 1000


def test_advertising_contacts_and_format_do_not_change_encoder_input(vacancy):
    vacancy = card(
        vacancy,
        "О компании:\nРекламный текст.\nОбязанности:\nРазрабатывать хранилище.\n"
        "Телефон: +7 (999) 123-45-67\n"
        "Требования:\nFluxCapacitor обязателен.\n"
        "Условия:\nДМС и бонусы.\n",
    )
    first = vacancy_document(vacancy)
    vacancy.description = vacancy.description.replace("Рекламный текст", "Новая реклама")
    second = vacancy_document(vacancy)
    assert first.input_revision == second.input_revision
    assert first.source_revision != second.source_revision
    assert "999" not in first.text and "Рекламный" not in first.text and "ДМС" not in first.text
    vacancy.description += "\nТребования:\nQuantumSQL обязателен."
    third = vacancy_document(vacancy)
    assert third.input_revision != second.input_revision


def test_fact_only_change_invalidates_meaning_but_timestamp_does_not(profile):
    profile.professional_facts = candidate_facts(profile, 42, confirmed_fields={"skills"})
    first = profile_document(profile)
    facts = profile.professional_facts
    facts.extracted_at = AT
    assert profile_document(profile).input_revision == first.input_revision
    fact = next(f for f in facts.facts if f.kind == "skill")
    fact.assertion = Claim[bool](status="unknown")
    assert profile_document(profile).input_revision != first.input_revision
