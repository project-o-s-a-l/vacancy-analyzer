import asyncio
import json
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from aiogram import Bot
from aiogram.methods import SendMessage
from aiogram.types import Document
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from sqlalchemy import func, select
from test_bot import TEST_TOKEN, RecordingSession, command_update
from test_users import callback_update, store_profile

from bot.delivery import TelegramSender
from bot.handlers import build_dispatcher
from bot.resume_cards import render_resume
from domain.resume import ResumeExtraction
from jobs.resumes import ResumeService
from jobs.users import UserService
from nlp.ollama_resume import ResumeAnalysisError
from nlp.resume import extract_resume
from processing.resume import MAX_RESUME_BYTES, ResumeBuffer, ResumeError, read_resume
from storage.models import NotificationModel, ResumeModel
from storage.repositories import VacancyRepository
from storage.repositories.user_repository import UserRepository

TEXT = (
    "Желаемая должность: Java Developer\nОбщий опыт: 2 года 6 месяцев\n"
    "Ключевые навыки: Java, SQL\nОбразование\nУниверситет"
)
SUMMARY = "backend python developer. опыт работы 0 лет. студент"


@pytest.mark.parametrize("existing", [False, True])
async def test_plain_resume_summary_is_reviewed_before_it_changes_profile(sessions, existing):
    extracted = ResumeExtraction(
        skills=["python"], positions=["backend python developer"], experience_years=0
    )
    analyzer = SimpleNamespace(analyze=AsyncMock(return_value=extracted))
    users = UserService(sessions, resume_analyzer=analyzer)
    await users.register(42)
    if existing:
        from domain import CandidateProfile

        await store_profile(
            sessions,
            42,
            CandidateProfile.model_validate(
                {"skills": {SUMMARY: {}}, "preferences": {"locations": ["Санкт-Петербург"]}}
            ),
        )
    before = await users.get(42)
    recording = RecordingSession()
    bot = Bot(TEST_TOKEN, session=recording)
    dispatcher = build_dispatcher(users)
    await dispatcher.feed_update(bot, command_update(SUMMARY))
    draft = await users.resumes.pending(42)
    assert draft and draft.version == "resume-v5-ollama"
    assert (await users.get(42)).profile == before.profile
    analyzer.analyze.assert_awaited_once_with(SUMMARY)
    assert "Предварительный профиль" in recording.requests[-1].text
    await dispatcher.feed_update(bot, callback_update("resume:confirm:" + draft.token))
    user = await users.get(42)
    assert "python" in user.profile.skills
    if existing:
        assert SUMMARY in user.profile.skills
    assert user.profile.preferences.positions == ["backend python developer"]
    assert user.profile.experience_years == 0 and user.form_state is None
    assert user.profile.preferences.locations == before.profile.preferences.locations
    async with sessions() as session:
        assert (await session.get(ResumeModel, 42)).text == SUMMARY


async def test_manual_skill_list_and_explicit_resume_do_not_need_model(sessions):
    analyzer = SimpleNamespace(analyze=AsyncMock())
    users = UserService(sessions, resume_analyzer=analyzer)
    await users.register(42)
    bot = Bot(TEST_TOKEN, session=RecordingSession())
    await build_dispatcher(users).feed_update(bot, command_update("Python, SQL"))
    assert set((await users.get(42)).profile.skills) == {"python", "sql"}
    assert (await users.get(42)).form_state == "setup:experience"
    await users.resumes.prepare(
        42, "Желаемая должность: Java Developer\nНавыки: Java, SQL\nОбщий опыт: 2 года"
    )
    analyzer.analyze.assert_not_awaited()


async def test_model_failure_keeps_active_profile_and_creates_reviewable_rules_draft(
    sessions, profile
):
    await store_profile(sessions, 42, profile)
    analyzer = SimpleNamespace(analyze=AsyncMock(side_effect=ResumeAnalysisError("unavailable")))
    users = UserService(sessions, resume_analyzer=analyzer)
    draft = await users.resumes.prepare(42, SUMMARY)
    assert (await users.get(42)).profile.model_copy(update={"professional_facts": None}) == profile
    assert any("Локальный разбор недоступен" in note for note in draft.extraction.notes)
    assert "Показаны прежние значения" in render_resume(draft, profile)
    analyzer.analyze.assert_awaited_once()


async def test_unregistered_user_does_not_trigger_inference(sessions):
    analyzer = SimpleNamespace(analyze=AsyncMock())
    with pytest.raises(ValueError, match="/start"):
        await ResumeService(sessions, analyzer=analyzer).prepare(42, SUMMARY)
    analyzer.analyze.assert_not_awaited()


def test_metadata_and_desired_role_in_skill_section_do_not_become_skills():
    extracted = extract_resume(
        "Желаемая должность: Backend Python Developer\n"
        "Навыки: Python, Backend Python Developer, опыт работы 0 лет, студент\n"
        "Общий опыт: 0 лет"
    )
    assert extracted.skills == ["python"]
    assert extracted.positions == ["Backend Python Developer"] and extracted.experience_years == 0


async def test_rule_fields_and_warnings_are_merged_without_false_missing_field_notes(sessions):
    analyzer = SimpleNamespace(
        analyze=AsyncMock(return_value=ResumeExtraction(positions=["Бухгалтер"]))
    )
    users = UserService(sessions, resume_analyzer=analyzer)
    await users.register(42)
    draft = await users.resumes.prepare(42, "Бухгалтер\nНавыки: Excel\nОбщий опыт: 0 лет")
    assert draft.extraction.skills == ["excel"] and draft.extraction.positions == ["Бухгалтер"]
    assert draft.extraction.experience_years == 0
    assert not any("не определ" in note for note in draft.extraction.notes)


async def test_conflicting_explicit_totals_are_not_overwritten_by_model_guess(sessions):
    analyzer = SimpleNamespace(
        analyze=AsyncMock(
            return_value=ResumeExtraction(positions=["Бухгалтер"], experience_years=5)
        )
    )
    users = UserService(sessions, resume_analyzer=analyzer)
    await users.register(42)
    draft = await users.resumes.prepare(42, "Навыки: Excel\nОбщий опыт: 2 года\nОбщий опыт: 5 лет")
    assert draft.extraction.experience_years is None
    assert any("несколько разных" in value for value in draft.extraction.notes)


def docx(text):
    from html import escape

    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        paragraphs = "".join(
            f"<w:p><w:r><w:t>{escape(line)}</w:t></w:r></w:p>" for line in text.splitlines()
        )
        archive.writestr(
            "word/document.xml",
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
            + paragraphs
            + "</w:body></w:document>",
        )
    return buffer.getvalue()


def pdf(*, pages=1, text=True, encrypted=False):
    writer = PdfWriter()
    for _ in range(pages):
        page = writer.add_blank_page(width=600, height=800)
        if text:
            page[NameObject("/Resources")] = DictionaryObject(
                {
                    NameObject("/Font"): DictionaryObject(
                        {
                            NameObject("/F1"): DictionaryObject(
                                {
                                    NameObject("/Type"): NameObject("/Font"),
                                    NameObject("/Subtype"): NameObject("/Type1"),
                                    NameObject("/BaseFont"): NameObject("/Helvetica"),
                                }
                            )
                        }
                    )
                }
            )
            stream = DecodedStreamObject()
            stream.set_data(
                b"BT /F1 12 Tf 14 TL 40 700 Td (Desired position: Accountant) Tj "
                b"T* (Skills: Excel, Audit) Tj ET"
            )
            page[NameObject("/Contents")] = writer._add_object(stream)
    if encrypted:
        writer.encrypt("private-test-password")
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


@pytest.mark.parametrize("kind", ["txt", "docx", "pdf"])
def test_read_all_resume_formats_without_writing_files(kind):
    data = TEXT.encode() if kind == "txt" else docx(TEXT) if kind == "docx" else pdf()
    text, actual = read_resume(data, "RESUME." + kind.upper())
    assert actual == kind
    assert "Java" in text if kind != "pdf" else "Accountant" in text
    assert extract_resume(text).skills


def test_windows_text_and_bounded_buffer():
    assert read_resume(TEXT.encode("cp1251"), "resume.txt")[0] == TEXT
    stream = ResumeBuffer()
    with pytest.raises(ResumeError, match="2 МБ"):
        stream.write(b"x" * (MAX_RESUME_BYTES + 1))
    assert stream.getvalue() == b""


@pytest.mark.parametrize(
    "data,name",
    [
        (b"", "resume.txt"),
        (b"a" * (MAX_RESUME_BYTES + 1), "resume.txt"),
        (b"not a pdf", "resume.pdf"),
        (b"not a docx", "resume.docx"),
        (b"\x00" * 100, "resume.txt"),
        (b"12345", "resume.txt"),
        (b"text", "resume.doc"),
        (b"text", "resume.docm"),
        (b"text", None),
    ],
    ids=[
        "empty",
        "too-large",
        "invalid-pdf",
        "invalid-docx",
        "binary",
        "no-text",
        "old-doc",
        "macro-doc",
        "no-name",
    ],
)
def test_invalid_uploads_have_fixed_actionable_errors(data, name):
    with pytest.raises(ResumeError):
        read_resume(data, name)


@pytest.mark.parametrize(
    "options,match",
    [
        ({"pages": 21}, "20 страниц"),
        ({"text": False}, "скана"),
        ({"encrypted": True}, "паролем"),
    ],
)
def test_pdf_page_text_and_encryption_limits(options, match):
    with pytest.raises(ResumeError, match=match):
        read_resume(pdf(**options), "resume.pdf")


def test_docx_rejects_expansion_and_entities():
    data = BytesIO()
    with ZipFile(data, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", b"x" * (10 * 1024 * 1024 + 1))
    with pytest.raises(ResumeError, match="распаковки"):
        read_resume(data.getvalue(), "resume.docx")
    data = BytesIO()
    with ZipFile(data, "w") as archive:
        archive.writestr(
            "word/document.xml", b'<!DOCTYPE a [<!ENTITY x SYSTEM "file:///private">]><a>&x;</a>'
        )
    with pytest.raises(ResumeError, match="структуру"):
        read_resume(data.getvalue(), "resume.docx")


@pytest.mark.parametrize(
    "title,skills",
    [
        ("Бухгалтер", ["расчёт зарплат", "Excel"]),
        ("Повар", ["разделка рыбы", "HACCP"]),
        ("Водитель", ["перевозка грузов", "ведение путевых листов"]),
        ("Java Developer", ["Java", "SQL"]),
    ],
)
def test_explicit_sections_are_universal(title, skills):
    extraction = extract_resume(
        f"Желаемая должность\n{title}\nНавыки:\n" + ", ".join(skills) + "\nОбразование\nВуз"
    )
    assert extraction.positions == [title]
    assert len(extraction.skills) == 2
    assert extraction.experience_years is None
    assert "вуз" not in extraction.skills


@pytest.mark.parametrize(
    "line,expected",
    [
        ("Общий опыт: 0 лет", 0),
        ("Опыт работы — 5 лет 3 месяца", 5.25),
        ("Total experience: 2 years 6 months", 2.5),
        ("Общий опыт: 2,5 года", 2.5),
        ("Опыт работы: 18 месяцев", 1.5),
        ("Общий опыт: без опыта", 0),
        ("Общий опыт: 90 лет", None),
        ("Общий опыт: 1 год 15 месяцев", None),
    ],
)
def test_only_explicit_total_experience(line, expected):
    assert extract_resume(line).experience_years == expected


def test_no_guessing_from_employment_dates_contacts_or_conflicting_totals():
    text = (
        "Python Developer\n2019–2025 бухгалтер\nExcel использовал в работе\n"
        "email@example.org\n+7 999 1234567"
    )
    extraction = extract_resume(text)
    assert not extraction.skills and not extraction.positions
    assert extraction.experience_years is None
    assert extract_resume("Общий опыт: 2 года\nОбщий опыт: 5 лет").experience_years is None


@pytest.mark.parametrize("kind", ["txt", "docx"])
def test_word_per_line_resume_preserves_position_zero_experience_and_skill_phrases(kind):
    original = (
        "Желаемая должность: Python Backend Developer\nУровень: стажёр / Junior\n"
        "Ключевые навыки: Python, FastAPI, REST API, SQL\nОбщий опыт: 0 лет\n"
        "Коммерческого опыта разработки нет.\nУчебные проекты:\nAPI для управления задачами"
    )
    wrapped = "\n\n".join(original.split())
    data = wrapped.encode() if kind == "txt" else docx(wrapped)
    text, actual = read_resume(data, "resume." + kind)
    extraction = extract_resume(text)
    assert actual == kind
    assert extraction.positions == ["Python Backend Developer"]
    assert extraction.experience_years == 0
    assert set(extraction.skills) == {"python", "fastapi", "rest api", "sql"}
    assert not extraction.notes
    assert "Желаемая\n\nдолжность:" in text  # The saved source text is unchanged.


def test_wrapped_english_fields_and_experience_months():
    text = (
        "Desired\nposition:\nPython\nBackend\nDeveloper\nLevel:\nJunior\n"
        "Skills:\nPython,\nREST\nAPI,\nSQL\nTotal\nwork\nexperience:\n0\nyears\n6\nmonths"
    )
    extraction = extract_resume(text)
    assert extraction.positions == ["Python Backend Developer"]
    assert extraction.experience_years == 0.5
    assert set(extraction.skills) == {"python", "rest api", "sql"}


def test_split_total_header_does_not_merge_separate_skill_list_items():
    extraction = extract_resume("Навыки:\nPython\nSQL\nОбщий\nопыт:\n0\nлет")
    assert extraction.skills == ["python", "sql"]
    assert extraction.experience_years == 0


def test_wrapped_employment_dates_do_not_create_total_experience():
    text = "\n".join(
        "Желаемая должность: Водитель Уровень: Начинающий Навыки: Вождение, "
        "перевозка грузов Образование: Учебные проекты 2020 2026".split()
    )
    extraction = extract_resume(text)
    assert extraction.positions == ["Водитель"]
    assert extraction.experience_years is None
    assert "общий" not in extraction.skills


def test_skill_aliases_limits_and_section_end():
    extraction = extract_resume(
        "Навыки: Postgres, PostgreSQL, Расчёт зарплаты, расчёт зарплат\n"
        "Контакты:\nprivate@example.org"
    )
    assert len(extraction.skills) == 2
    assert not any("@" in skill for skill in extraction.skills)
    assert len(extract_resume("Skills: " + ",".join(f"tool{i}" for i in range(100))).skills) == 50
    assert extract_resume("Skills: Excel, private@example.org, +7 999 1234567").skills == ["excel"]


async def test_draft_confirmation_is_atomic_persistent_and_preserves_preferences(sessions, profile):
    await store_profile(sessions, 42, profile)
    users = UserService(sessions)
    draft = await users.resumes.prepare(42, TEXT)
    assert (await users.get(42)).profile.model_copy(update={"professional_facts": None}) == profile
    restarted = ResumeService(sessions)
    assert await restarted.pending(42) == draft
    preview = render_resume(draft, profile)
    assert "Java Developer" in preview and "ещё не применён" in preview
    user = await restarted.confirm(42, draft.token)
    assert set(user.profile.skills) == {*profile.skills, "java", "sql"}
    assert user.profile.experience_years == 2.5
    assert user.profile.preferences.positions == list(
        dict.fromkeys([*profile.preferences.positions, "Java Developer"])
    )
    assert user.profile.preferences.salary_min == profile.preferences.salary_min
    assert user.profile.qualifications == profile.qualifications
    assert user.form_state is None
    assert await restarted.pending(42) is None
    async with sessions() as session:
        row = await session.get(ResumeModel, 42)
        assert row.text == TEXT and row.parser_version == "resume-v5"
    with pytest.raises(ValueError):
        await restarted.confirm(42, draft.token)


async def test_stale_or_foreign_drafts_cannot_overwrite_profiles(sessions, profile):
    users = UserService(sessions)
    for user_id in (42, 99):
        await store_profile(sessions, user_id, profile)
    first = await users.resumes.prepare(42, TEXT)
    second = await users.resumes.prepare(42, TEXT + "\nNew resume")
    with pytest.raises(ValueError):
        await users.resumes.confirm(42, first.token)
    with pytest.raises(ValueError):
        await users.resumes.confirm(99, second.token)
    await users.begin_edit(42, "experience")
    await users.accept_answer(42, "7")
    with pytest.raises(ValueError, match="Профиль изменился"):
        await users.resumes.confirm(42, second.token)
    assert (await users.get(42)).profile.experience_years == 7
    assert (await users.get(99)).profile.model_copy(update={"professional_facts": None}) == profile


async def test_cancel_delete_and_empty_extraction_do_not_erase_profile(sessions, profile):
    await store_profile(sessions, 42, profile)
    resumes = ResumeService(sessions)
    draft = await resumes.prepare(42, "Текст резюме с описанием работы без явных разделов.")
    user = await resumes.confirm(42, draft.token)
    assert user.profile.model_copy(update={"professional_facts": None}) == profile
    new = await resumes.prepare(42, TEXT)
    await resumes.cancel(42, new.token)
    async with sessions() as session:
        assert (await session.get(ResumeModel, 42)).text != TEXT
    await resumes.delete(42)
    async with sessions() as session:
        assert await session.get(ResumeModel, 42) is None
    assert (await UserService(sessions).get(42)).profile.model_copy(
        update={"professional_facts": None}
    ) == profile


async def test_empty_profile_needs_skills_after_confirmation(sessions):
    users = UserService(sessions)
    await users.register(42)
    draft = await users.resumes.prepare(
        42, "Резюме с описанием работы, без списка навыков и стажа."
    )
    assert (await users.resumes.confirm(42, draft.token)).form_state == "edit:skills"


async def test_confirmed_resume_drives_real_catalog_ranking(sessions, vacancy):
    users = UserService(sessions)
    await users.register(42)
    draft = await users.resumes.prepare(42, TEXT)
    async with sessions() as session:
        assert await UserRepository(session).search_profiles() == []
    await users.resumes.confirm(42, draft.token)
    async with sessions.begin() as session:
        await VacancyRepository(session).save(
            vacancy.model_copy(
                update={
                    "title": "Java Developer",
                    "skills": ["Java", "SQL"],
                    "description": "Java, SQL",
                    "description_html": None,
                    "experience": "1-3",
                }
            )
        )
    result = await (await users.notifications(42)).pending()
    assert result and set(result[0][1].matched_skills) == {"java", "sql"}
    async with sessions() as session:
        assert (await UserRepository(session).search_profiles())[0].preferences.positions == [
            "Java Developer"
        ]
    bot = Bot(TEST_TOKEN, session=RecordingSession())
    assert (await (await users.notifications(42)).deliver(TelegramSender(bot))).sent == 1
    async with sessions() as session:
        context = (await session.get(NotificationModel, (42, vacancy.id))).exposure_context
        assert TEXT not in json.dumps(context, ensure_ascii=False)
    replacement = await users.resumes.prepare(42, TEXT + "\nНовая версия резюме.")
    await users.resumes.confirm(42, replacement.token)
    assert await (await users.notifications(42)).pending() == []


def document_update(data=None, user_id=42, name="resume.txt"):
    data = TEXT.encode() if data is None else data
    update = command_update("/resume", user_id=user_id, chat_id=user_id)
    return update.model_copy(
        update={
            "message": update.message.model_copy(
                update={
                    "text": None,
                    "entities": [],
                    "document": Document(
                        file_id="test-file",
                        file_unique_id="unique",
                        file_name=name,
                        file_size=len(data),
                    ),
                }
            )
        }
    )


async def test_telegram_file_review_confirm_delete_with_fake_download(sessions, monkeypatch):
    users = UserService(sessions)
    await users.register(42)
    download = AsyncMock(
        side_effect=lambda *args, **kwargs: kwargs["destination"].write(TEXT.encode())
    )
    monkeypatch.setattr(Bot, "download", download)
    transport = RecordingSession()
    bot = Bot(TEST_TOKEN, session=transport)
    dispatcher = build_dispatcher(users)
    await dispatcher.feed_update(bot, document_update())
    assert not (await users.get(42)).profile.skills
    draft = await users.resumes.pending(42)
    messages = [r for r in transport.requests if isinstance(r, SendMessage)]
    assert "Предварительный" in messages[-1].text
    assert messages[-1].reply_markup.inline_keyboard[0][0].callback_data.endswith(draft.token)
    await dispatcher.feed_update(bot, callback_update("resume:confirm:" + draft.token))
    assert set((await users.get(42)).profile.skills) == {"java", "sql"}
    await dispatcher.feed_update(bot, command_update("/resume_delete"))
    async with sessions() as session:
        assert await session.get(ResumeModel, 42) is None


async def test_unregistered_and_wrong_type_uploads_not_downloaded(sessions, monkeypatch):
    users = UserService(sessions)
    download = AsyncMock()
    monkeypatch.setattr(Bot, "download", download)
    bot = Bot(TEST_TOKEN, session=RecordingSession())
    dispatcher = build_dispatcher(users)
    await dispatcher.feed_update(bot, document_update())
    await users.register(42)
    await dispatcher.feed_update(bot, document_update(name="resume.exe"))
    await dispatcher.feed_update(bot, document_update(data=b"x" * (MAX_RESUME_BYTES + 1)))
    download.assert_not_awaited()


async def test_resume_text_button_and_cancel_via_real_dispatcher(sessions):
    users = UserService(sessions)
    await users.register(42)
    transport = RecordingSession()
    bot = Bot(TEST_TOKEN, session=transport)
    dispatcher = build_dispatcher(users)
    await dispatcher.feed_update(bot, callback_update("resume:upload"))
    assert "PDF" in transport.requests[-1].text
    await dispatcher.feed_update(bot, command_update("/resume " + TEXT))
    assert await users.resumes.pending(42)
    await dispatcher.feed_update(bot, command_update("/cancel"))
    assert await users.resumes.pending(42) is None
    assert (await users.get(42)).profile.skills == {}


async def test_failed_download_keeps_previous_draft(sessions, monkeypatch):
    users = UserService(sessions)
    await users.register(42)
    original = await users.resumes.prepare(42, TEXT)

    def oversized(*args, **kwargs):
        kwargs["destination"].write(b"x" * (MAX_RESUME_BYTES + 1))

    monkeypatch.setattr(Bot, "download", AsyncMock(side_effect=oversized))
    transport = RecordingSession()
    await build_dispatcher(users).feed_update(Bot(TEST_TOKEN, session=transport), document_update())
    assert "2 МБ" in transport.requests[-1].text
    assert await users.resumes.pending(42) == original


@pytest.mark.postgres
async def test_postgres_resume_confirmation_isolation_and_concurrency(postgres_sessions, profile):
    users = UserService(postgres_sessions)
    for user_id in (42, 99):
        await store_profile(postgres_sessions, user_id, profile)
    draft = await users.resumes.prepare(42, TEXT)
    results = await asyncio.gather(
        users.resumes.confirm(42, draft.token),
        users.resumes.confirm(42, draft.token),
        return_exceptions=True,
    )
    assert sum(isinstance(value, ValueError) for value in results) == 1
    assert (await users.get(99)).profile.model_copy(update={"professional_facts": None}) == profile
    async with postgres_sessions() as session:
        assert await session.scalar(select(func.count()).select_from(ResumeModel)) == 1
