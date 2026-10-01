import json
from copy import deepcopy

import pytest

from nlp.ollama_evidence import referenced_document
from nlp.ollama_input import InputTooLarge, prepare_input


def test_short_input_preserves_all_fragments_and_has_full_scope():
    document = {
        "title": "Бухгалтер",
        "description": "О компании\nТребования:\nЗнание Excel.",
        "source_skills": ["Excel"],
    }
    before = deepcopy(document)
    prepared = prepare_input(document, max_chars=6000)
    assert prepared.references == referenced_document(document)
    assert json.loads(prepared.content) == prepared.references
    assert prepared.scope == "full" and prepared.omitted_fragments == 0
    assert document == before


@pytest.mark.parametrize(
    "title,headings,requirement",
    [
        ("Бухгалтер", ("Обязанности", "Требования", "Будет плюсом"), "Ведение учёта, без опыта"),
        (
            "Водитель",
            ("Ваши задачи", "Требования к кандидату", "Документы и допуски"),
            "Перевозка грузов; права категории C",
        ),
        (
            "Медицинская сестра",
            ("Должностные обязанности", "Профессиональные навыки", "Желательно"),
            "Уход за пациентами",
        ),
        (
            "Accountant",
            ("Key Responsibilities", "Requirements", "Nice to Have"),
            "Excel; no prior experience required",
        ),
    ],
)
def test_scoped_input_keeps_complete_work_sections_for_different_professions(
    title, headings, requirement
):
    description = (
        "О компании:\n"
        + "Описание работодателя. " * 500
        + "\n"
        + headings[0]
        + "\nРабочие задачи.\n"
        + headings[1]
        + "\n"
        + requirement
        + "\n"
        + headings[2]
        + "\nДополнительный навык не обязателен.\n"
        + "Условия:\n"
        + "Условия и преимущества. " * 500
    )
    document = {"title": title, "description": description, "source_skills": []}
    before = deepcopy(document)
    prepared = prepare_input(document, max_chars=6000)
    selected = "\n".join(prepared.references["description"].values())
    assert prepared.scope == "work_sections" and prepared.omitted_fragments > 0
    assert len(prepared.content) <= 6000
    assert requirement in selected
    assert "Дополнительный навык не обязателен." in selected
    assert all(heading in selected for heading in headings)
    assert "Описание работодателя" not in selected and "Условия и преимущества" not in selected
    assert document == before


def test_both_language_requirement_blocks_are_kept_and_references_are_not_renumbered():
    document = {
        "title": "Developer",
        "description": "About us\n"
        + "Company overview. " * 500
        + "\nRequirements\n5+ years of experience.\nNice to Have\nSQL is preferred.\n"
        + "What we offer\nRemote work.\nТребования\nОпыт работы от 5 лет.\n"
        + "Будет плюсом\nSQL желательно.\nМы предлагаем\nУдалённая работа.",
        "source_skills": ["SQL"],
    }
    full = referenced_document(document)
    prepared = prepare_input(document, max_chars=6000)
    selected = prepared.references["description"]
    assert "5+ years of experience." in selected.values()
    assert "Опыт работы от 5 лет." in selected.values()
    assert "SQL is preferred." in selected.values() and "SQL желательно." in selected.values()
    assert all(full["description"][key] == value for key, value in selected.items())
    assert prepared.references["source_skills"] == full["source_skills"]
    assert prepared.omitted_fragments == len(full["description"]) - len(selected)


def test_unknown_heading_ends_selected_section_and_coverage_remains_partial():
    document = {
        "title": "Бухгалтер",
        "description": "Компания. " * 1000
        + "\nТребования:\nExcel обязателен.\n"
        + "Дополнительные критерии:\nSQL обязателен.",
        "source_skills": ["Excel", "SQL"],
    }
    prepared = prepare_input(document, max_chars=6000)
    assert prepared.scope == "work_sections"
    assert "Excel обязателен." in prepared.references["description"].values()
    assert "SQL обязателен." not in prepared.references["description"].values()


@pytest.mark.parametrize("header", ["Требования", "Requirements", ""])
def test_source_is_not_cut_when_even_work_sections_exceed_the_budget(header):
    document = {
        "title": "Developer",
        "description": header + "\n" + "Обязательное требование. " * 500 + "\nSQL не требуется.",
        "source_skills": [],
    }
    before = deepcopy(document)
    with pytest.raises(InputTooLarge, match="не обрезан"):
        prepare_input(document, max_chars=6000)
    assert document == before
