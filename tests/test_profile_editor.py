"""real Telegram transport, durable writes, import fences and production matching."""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from aiogram import Bot
from aiogram.methods import AnswerCallbackQuery, SendMessage
from sqlalchemy import select, update
from test_bot import TEST_TOKEN, RecordingSession
from test_comparison import job, skill_requirement
from test_users import callback_update, text_update

from bot.handlers import build_dispatcher
from bot.profile_editor import screen
from jobs.profile_editor import ProfileEditor
from jobs.rank import rank_recent
from jobs.resumes import StaleResumeWork
from jobs.users import UserService
from nlp.professional_facts import profile_payload
from scoring import score_vacancy
from scoring.comparison import compare_facts
from scoring.serving_features import profile_revision
from storage.models import (
    FactRevisionModel,
    ProfileEditModel,
    SavedSelectionModel,
    TelegramUserModel,
)
from storage.repositories import VacancyRepository
from storage.repositories.user_repository import StaleProfileWrite, UserRepository


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def editor_sessions(request):
    return request.getfixturevalue(
        "postgres_sessions" if request.param == "postgres" else "sessions"
    )


class Chat:
    def __init__(self, service, owner=42):
        self.service, self.owner = service, owner
        self.transport = RecordingSession()
        self.bot = Bot(TEST_TOKEN, session=self.transport)
        self.dispatcher = build_dispatcher(service)

    async def send(self, text):
        await self.dispatcher.feed_update(self.bot, text_update(text, self.owner))

    def data(self, label):
        message = next(
            r
            for r in reversed(self.transport.requests)
            if isinstance(r, SendMessage) and r.reply_markup
        )
        buttons = [b for row in message.reply_markup.inline_keyboard for b in row]
        return next(b.callback_data for b in buttons if b.text == label)

    async def click(self, label):
        data = self.data(label)
        await self.dispatcher.feed_update(self.bot, callback_update(data, user_id=self.owner))
        return data

    async def field(self, label, value):
        await self.click(label)
        await self.send(value)


async def change(editor, owner, edit, kind, key, field, text):
    return await editor.act(owner, edit.token, ["set", kind, key, field, text])


async def seed_profile(service, owner=42, role="Разработчик", task="Разрабатывать отчёты"):
    await service.register(owner)
    editor = service.editor
    edit = await editor.open(owner)
    edit = await change(editor, owner, edit, "skill", "", "new", "Python")
    edit = await change(editor, owner, edit, "basic", "", "experience", "5")
    edit = await change(editor, owner, edit, "basic", "", "positions", "Разработчик")
    edit = await change(editor, owner, edit, "episode", "", "new", "employment")
    key = edit.view[2]
    for field, value in {
        "roles": role,
        "tasks": task,
        "started": "2020-01",
        "ended": "2023-12",
        "skills": "Python",
    }.items():
        edit = await change(editor, owner, edit, "episode", key, field, value)
    await editor.act(owner, edit.token, ["save"])
    return (await service.get(owner)).profile, key


async def test_manual_resume_through_real_bot_without_llm_and_restart(editor_sessions):
    users = UserService(editor_sessions)
    chat = Chat(users)
    await chat.send("/start")
    await chat.send("/profile")
    await chat.click("Навыки и их применение")
    await chat.click("+ Добавить")
    await chat.send("Python")
    await chat.field("Лет применения навыка", "2.5")
    await chat.click("Уровень навыка")
    await chat.click("Средний")
    await chat.field("Последнее использование", "15.06.2025")
    await chat.click("← Назад")
    await chat.click("← Назад")
    await chat.click("Опыт и проекты (0)")
    await chat.click("+ Место работы")
    await chat.field("Организация", "<Работа & компания> 😀")
    await chat.field("Занимаемые должности", "Разработчик")
    await chat.field("Начало", "2020-01")
    await chat.field("Окончание", "2022-12")
    await chat.field("Задачи", "Разрабатывать отчёты; Автоматизировать обработку")
    await chat.field("Навыки этого опыта", "Python")
    await chat.click("← Назад")
    await chat.click("+ Место работы")
    await chat.field("Название записи / проекта", "Вторая работа")
    await chat.field("Начало", "2022-01")  # overlaps intentionally
    await chat.click("Продолжается сейчас")
    await chat.click("Да")
    await chat.click("← Назад")
    await chat.click("+ Проект")
    await chat.field("Название записи / проекта", "Учебная система")
    await chat.click("Вид опыта")
    await chat.click("Учебный проект")
    await chat.click("Начало")
    await chat.click("Не знаю / очистить")
    await chat.field("Задачи", "Обработка данных")
    await chat.click("Связь с местом работы")
    await chat.click("Выбрать работу")
    await chat.click("<Работа & компания> 😀")
    # Nothing from the editor is used before its explicit save.
    assert not (await users.get(42)).profile.skills
    pending = await ProfileEditor(editor_sessions).get(42)
    assert len(pending.profile.professional_facts.episodes) == 3
    await chat.click("Сохранить правки")
    restarted = UserService(editor_sessions)
    user = await restarted.get(42)
    assert user.form_state is None
    assert user.profile == pending.profile
    episodes = user.profile.professional_facts.episodes
    assert episodes[1].ongoing.value is True and episodes[1].ended.value is None
    assert episodes[2].context == "academic" and episodes[2].started.value is None
    assert episodes[2].parent_id == episodes[0].id
    skill = next(f for f in user.profile.professional_facts.facts if f.kind == "skill")
    assert skill.years.value == 2.5 and skill.level.value == "intermediate"
    assert skill.last_used.value.isoformat() == "2025-06-15"
    assert skill.years.origin == "user_confirmed"
    assert skill.years.evidence[0].quote is None
    await chat.send("/profile")
    await chat.click("Опыт и проекты")
    assert "<Работа & компания> 😀" in chat.transport.requests[-1].text
    for request in chat.transport.requests:
        if isinstance(request, SendMessage):
            assert request.parse_mode is None
            assert len(request.text.encode("utf-16-le")) // 2 <= 3500
            if request.reply_markup:
                assert all(
                    len(b.callback_data.encode()) <= 64
                    for row in request.reply_markup.inline_keyboard
                    for b in row
                    if b.callback_data
                )
    await chat.bot.session.close()


async def test_delete_cascade_unknown_clears_and_cancel_are_atomic(editor_sessions):
    users = UserService(editor_sessions)
    original, key = await seed_profile(users)
    edit = await users.editor.open(42)
    edit = await change(users.editor, 42, edit, "episode", "", "new", "project")
    project = edit.view[2]
    edit = await change(users.editor, 42, edit, "episode", project, "parent", key)
    edit = await change(users.editor, 42, edit, "episode", key, "delete", "yes")
    assert not any(f.kind in {"task", "role"} for f in edit.profile.professional_facts.facts)
    assert edit.profile.professional_facts.episodes[0].parent_id is None
    assert edit.profile.skills
    await users.editor.act(42, edit.token, ["cancel"])
    assert (await users.get(42)).profile == original
    edit = await users.editor.open(42)
    skill = next(f for f in edit.profile.professional_facts.facts if f.kind == "skill")
    edit = await change(users.editor, 42, edit, "skill", skill.id, "years", "0")
    assert (
        next(f for f in edit.profile.professional_facts.facts if f.kind == "skill").years.value == 0
    )
    edit = await change(users.editor, 42, edit, "skill", skill.id, "years", "?")
    edit = await change(users.editor, 42, edit, "basic", "", "experience", "не знаю")
    edit = await change(users.editor, 42, edit, "skill", skill.id, "delete", "yes")
    await users.editor.act(42, edit.token, ["save"])
    # A later ordinary field save must not revive the skill through its episode link.
    await users.begin_edit(42, "locations")
    user = await users.accept_answer(42, "Казань")
    assert not user.profile.skills
    assert not any(f.kind == "skill" for f in user.profile.professional_facts.facts)
    assert user.profile.professional_facts.episodes[0].skill_ids == []
    assert user.profile.experience_years is None
    assert (
        next(
            f for f in user.profile.professional_facts.facts if f.id == "experience_total"
        ).assertion.value
        is None
    )
    async with editor_sessions() as session:
        assert len(list(await session.scalars(select(FactRevisionModel)))) >= 3


async def test_stale_foreign_repeated_callbacks_and_back(editor_sessions):
    users = UserService(editor_sessions)
    await seed_profile(users)
    await users.register(99)
    chat = Chat(users)
    await chat.send("/profile")
    await chat.click("Навыки и их применение")
    old = await chat.click("1. python")
    before = await users.editor.get(42)
    await chat.dispatcher.feed_update(chat.bot, callback_update(old))
    assert (await users.editor.get(42)) == before
    assert isinstance(chat.transport.requests[-1], AnswerCallbackQuery)
    assert "устарела" in chat.transport.requests[-1].text
    current = chat.data("Лет применения навыка")
    await chat.dispatcher.feed_update(chat.bot, callback_update(current, user_id=99))
    assert await users.editor.get(99) is None
    await chat.click("Лет применения навыка")
    await chat.click("← Назад")
    assert (await users.editor.get(42)).profile == before.profile
    await chat.click("Удалить…")
    assert (await users.get(42)).profile.skills
    await chat.click("← Назад")
    assert (await users.editor.get(42)).profile.skills
    data = await chat.click("Сохранить правки")
    revision = (await users.get(42)).revision
    await chat.dispatcher.feed_update(chat.bot, callback_update(data))
    assert (await users.get(42)).revision == revision
    await chat.bot.session.close()


async def test_import_review_uses_same_editor_and_preserves_manual_deletions(editor_sessions):
    users = UserService(editor_sessions)
    await users.register(42)
    text = (
        "Навыки: Python, SQL\nОпыт работы\n01.2020 — 12.2022\n"
        "Должность: Разработчик\nЗадачи: Разрабатывать отчёты"
    )
    draft = await users.resumes.prepare(42, text)
    edit = await users.editor.open(42, resume_token=draft.token)
    assert edit.profile.professional_facts.episodes
    key = edit.profile.professional_facts.episodes[0].id
    edit = await change(
        users.editor, 42, edit, "episode", key, "organization", "Исправленная организация"
    )
    edit = await change(users.editor, 42, edit, "episode", key, "tasks", "Проверять расчёты")
    await users.editor.act(42, edit.token, ["save"])
    with pytest.raises(ValueError):
        await users.resumes.confirm(42, draft.token)
    draft = await users.resumes.pending(42)
    saved = await users.resumes.confirm(42, draft.token)
    assert (
        saved.profile.professional_facts.episodes[0].organization.value
        == "Исправленная организация"
    )
    edit = await users.editor.open(42)
    edit = await change(users.editor, 42, edit, "episode", key, "delete", "yes")
    skill = next(f for f in edit.profile.professional_facts.facts if f.kind == "skill")
    edit = await change(users.editor, 42, edit, "skill", skill.id, "delete", "yes")
    await users.editor.act(42, edit.token, ["save"])
    removed = (await users.get(42)).profile
    # Neither an old confirmed draft nor a fresh merge revives manually cleared sections.
    with pytest.raises(ValueError):
        await users.resumes.confirm(42, draft.token)
    repeated = await users.resumes.prepare(42, text)
    user = await users.resumes.confirm(42, repeated.token)
    assert not user.profile.professional_facts.episodes
    assert user.profile.skills == removed.skills
    # An explicit full replacement has a separate preview and can restore imported data.
    replacement = await users.resumes.prepare(42, text)
    replacement = await users.resumes.revise(42, replacement.token, "mode", "replace_all")
    replaced = await users.resumes.confirm(42, replacement.token)
    assert replaced.profile.professional_facts.episodes
    assert set(replaced.profile.skills) == {"python", "sql"}


async def test_profile_change_rejects_old_editor_and_old_import(editor_sessions):
    users = UserService(editor_sessions)
    await seed_profile(users)
    draft = await users.resumes.prepare(42, "Навыки: SQL\nДолжность: Аналитик данных")
    edit = await users.editor.open(42)
    await users.begin_edit(42, "experience")
    await users.accept_answer(42, "7")
    with pytest.raises(ValueError, match="Профиль изменился"):
        await users.editor.act(42, edit.token, ["save"])
    await users.editor.cancel(42)
    with pytest.raises(ValueError, match="Профиль изменился"):
        await users.resumes.confirm(42, draft.token)
    assert (await users.get(42)).profile.experience_years == 7


async def test_failed_edit_and_cancel_leave_profile_and_revision_intact(editor_sessions):
    users = UserService(editor_sessions)
    _, key = await seed_profile(users)
    original = await users.get(42)
    edit = await users.editor.open(42)
    with pytest.raises(ValueError):
        await change(users.editor, 42, edit, "episode", key, "started", "2025-01")
    assert (await users.editor.get(42)).token == edit.token
    assert await users.get(42) == original
    await users.editor.act(42, edit.token, ["cancel"])
    assert await users.get(42) == original


async def test_read_old_writer_mismatch_preserves_career_and_prunes_removed_skill(editor_sessions):
    users = UserService(editor_sessions)
    profile, _ = await seed_profile(users)
    legacy = profile_payload(profile)
    legacy["skills"] = {}
    async with editor_sessions.begin() as session:
        await session.execute(
            update(TelegramUserModel).where(TelegramUserModel.user_id == 42).values(profile=legacy)
        )
    user = await users.get(42)
    assert user.profile.professional_facts.episodes
    assert not user.profile.professional_facts.episodes[0].skill_ids
    assert not any(f.kind == "skill" for f in user.profile.professional_facts.facts)
    async with editor_sessions.begin() as session:
        user = await UserRepository(session).save(user)
    assert (await users.get(42)).profile == user.profile


async def test_legacy_write_without_revision_rejects_stale_editor_and_repository(editor_sessions):
    users = UserService(editor_sessions)
    profile, _ = await seed_profile(users)
    old = await users.get(42)
    edit = await users.editor.open(42)
    values = profile_payload(profile)
    values["preferences"]["remote"] = "required"
    async with editor_sessions.begin() as session:
        await session.execute(
            update(TelegramUserModel).where(TelegramUserModel.user_id == 42).values(profile=values)
        )
    with pytest.raises(ValueError, match="Профиль изменился"):
        await users.editor.act(42, edit.token, ["save"])
    async with editor_sessions.begin() as session:
        with pytest.raises(StaleProfileWrite):
            await UserRepository(session).save(old)
    assert (await users.get(42)).profile.preferences.remote == "required"


async def test_repository_rejects_foreign_facts_before_regeneration(editor_sessions):
    users = UserService(editor_sessions)
    await seed_profile(users)
    await seed_profile(users, 99, role="Чужая роль", task="Чужая задача")
    user = await users.get(42)
    other = (await users.get(99)).profile
    async with editor_sessions.begin() as session:
        with pytest.raises(ValueError, match="another owner"):
            await UserRepository(session).save(
                replace(user, profile=other), confirmed_fields={"remote"}
            )
    assert (await users.get(42)).profile == user.profile


async def test_partial_import_scope_survives_manual_preference_edit(editor_sessions):
    users = UserService(editor_sessions)
    await users.register(42)
    draft = await users.resumes.prepare(
        42, "Навыки: Python\nЗадачи: Обрабатывать данные и готовить отчёты."
    )
    await users.resumes.confirm(42, draft.token)
    edit = await users.editor.open(42)
    assert edit.profile.professional_facts.input_scope == "partial"
    ranges = edit.profile.professional_facts.input_ranges
    edit = await change(users.editor, 42, edit, "basic", "", "remote", "required")
    await users.editor.act(42, edit.token, ["save"])
    facts = (await users.get(42)).profile.professional_facts
    assert facts.input_scope == "partial" and facts.input_ranges == ranges


async def test_calendar_total_cannot_survive_changed_dates_or_reappear_on_merge(editor_sessions):
    users = UserService(editor_sessions)
    await users.register(42)
    text = "Навыки: Python\nОпыт работы\n01.2020 — 12.2022\nДолжность: Инженер"
    draft = await users.resumes.prepare(42, text)
    user = await users.resumes.confirm(42, draft.token)
    assert user.profile.experience_years == 3
    key = user.profile.professional_facts.episodes[0].id
    edit = await users.editor.open(42)
    edit = await change(users.editor, 42, edit, "basic", "", "remote", "required")
    edit = await change(users.editor, 42, edit, "episode", key, "ended", "2021-12")
    await users.editor.act(42, edit.token, ["save"])
    assert (await users.get(42)).profile.experience_years is None
    draft = await users.resumes.prepare(42, text)
    user = await users.resumes.confirm(42, draft.token)
    assert user.profile.experience_years is None
    assert user.profile.professional_facts.episodes[0].ended.value.year == 2021


async def test_comparison_and_real_retrieval_receive_manual_facts(editor_sessions, vacancy):
    users = UserService(editor_sessions)
    now = datetime.now(UTC)
    a, key = await seed_profile(users)
    b, _ = await seed_profile(users, 99, role="Бухгалтер", task="Сверять счета")
    assert a.skills == b.skills and a.experience_years == b.experience_years
    v = job(vacancy, now, "Требования: Python обязателен.\nОбязанности:\nРазрабатывать отчёты.")
    async with editor_sessions.begin() as session:
        await VacancyRepository(session).save(v)
    results = []
    for owner in (42, 99):
        async with editor_sessions() as session:
            current = await UserRepository(session).get(owner)
            pairs = await rank_recent(VacancyRepository(session), current.profile, now=now)
            assert len(pairs) == 1
            results.append(pairs[0][1])
    assert results[0].comparison.features["task_match"] == 1
    assert results[1].comparison.features["task_match"] == 0
    assert results[0].comparison.features["role_match"] == 1
    assert results[1].comparison.features["role_match"] == 0
    edit = await users.editor.open(42)
    edit = await change(users.editor, 42, edit, "episode", key, "tasks", "?")
    edit = await change(users.editor, 42, edit, "episode", key, "roles", "?")
    await users.editor.act(42, edit.token, ["save"])
    result = compare_facts(v, (await users.get(42)).profile, now=now)
    assert result.features["task_match"] is None and result.features["role_match"] is None
    # Explicit tenure differs from unknown even with identical total experience.
    skill_requirement(v)
    unknown = score_vacancy(v, (await users.get(42)).profile, now=now)
    edit = await users.editor.open(42)
    skill = next(f for f in edit.profile.professional_facts.facts if f.kind == "skill")
    edit = await change(users.editor, 42, edit, "skill", skill.id, "years", "1")
    await users.editor.act(42, edit.token, ["save"])
    known = score_vacancy(v, (await users.get(42)).profile, now=now)
    assert unknown.eligible and not known.eligible
    assert unknown.comparison.features["skill_years_fit"] is None
    assert known.comparison.features["skill_years_fit"] == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    "kind,field,value",
    [
        ("episode", "roles", "Бухгалтер"),
        ("episode", "tasks", "Сверять счета"),
        ("episode", "ended", "2022-01"),
        ("skill", "years", "2"),
    ],
)
async def test_fact_only_changes_invalidate_actual_cache(
    editor_sessions, vacancy, kind, field, value
):
    users = UserService(editor_sessions)
    original, key = await seed_profile(users)
    now = datetime.now(UTC)
    v = job(vacancy, now)
    async with editor_sessions.begin() as session:
        await VacancyRepository(session).save(v)

    async def rank(profile):
        stats = {}
        async with editor_sessions.begin() as session:
            await rank_recent(
                VacancyRepository(session), profile, cache_owner=42, now=now, diagnostics=stats
            )
        return stats

    assert (await rank(original))["candidate_cache_hit"] == 0
    assert (await rank(original))["candidate_cache_hit"] == 1
    edit = await users.editor.open(42)
    if kind == "skill":
        key = next(f.id for f in edit.profile.professional_facts.facts if f.kind == "skill")
    edit = await change(users.editor, 42, edit, kind, key, field, value)
    await users.editor.act(42, edit.token, ["save"])
    current = (await users.get(42)).profile
    assert set(original.skills) == set(current.skills)
    assert original.experience_years == current.experience_years
    assert profile_revision(current) != profile_revision(original)
    async with editor_sessions() as session:
        assert await session.get(SavedSelectionModel, 42) is None
    assert (await rank(current))["candidate_cache_hit"] == 0


async def test_mandatory_preference_reaches_eligibility_through_bot(editor_sessions, vacancy):
    users = UserService(editor_sessions)
    await seed_profile(users)
    v = job(vacancy, datetime.now(UTC), remote=False)
    assert score_vacancy(v, (await users.get(42)).profile).eligible
    chat = Chat(users)
    await chat.send("/profile")
    await chat.click("Формат работы")
    await chat.click("Только удалённо")
    await chat.click("Сохранить правки")
    assert not score_vacancy(v, (await users.get(42)).profile).eligible
    await chat.send("/profile")
    await chat.click("Формат работы")
    await chat.click("Не знаю / очистить")
    await chat.click("Сохранить правки")
    assert score_vacancy(v, (await users.get(42)).profile).eligible
    await chat.bot.session.close()


@pytest.mark.postgres
async def test_postgres_one_winner_for_duplicate_step_and_atomic_profile_cas(postgres_sessions):
    users = UserService(postgres_sessions)
    await seed_profile(users)
    edit = await users.editor.open(42)
    results = await asyncio.gather(
        *[
            users.editor.act(42, edit.token, ["set", "basic", "", "experience", value])
            for value in ("2", "3")
        ],
        return_exceptions=True,
    )
    assert sum(isinstance(r, ValueError) for r in results) == 1
    edit = await users.editor.get(42)
    results = await asyncio.gather(
        *[users.editor.act(42, edit.token, ["save"]) for _ in range(2)], return_exceptions=True
    )
    assert sum(isinstance(r, ValueError) for r in results) == 1
    user = await users.get(42)

    async def write(years):
        async with postgres_sessions.begin() as session:
            profile = user.profile.model_copy(update={"experience_years": years})
            return await UserRepository(session).save(
                replace(user, profile=profile), confirmed_fields={"experience"}
            )

    results = await asyncio.gather(write(4), write(6), return_exceptions=True)
    assert sum(isinstance(r, StaleProfileWrite) for r in results) == 1


async def test_pagination_full_text_and_resume_delete_cleans_private_editor(editor_sessions):
    users = UserService(editor_sessions)
    await users.register(42)
    edit = await users.editor.open(42)
    for index in range(9):
        edit = await change(
            users.editor, 42, edit, "task", "", "new", f"Задача {index} " + "😀" * 150
        )
    edit = await users.editor.act(42, edit.token, ["view", "list", "task", 0])
    text, actions = screen(edit)
    assert "Задача 0" in text and "Задача 8" not in text
    action = next(a for label, a in actions if label == "Следующие →")
    edit = await users.editor.act(42, edit.token, action)
    assert "Задача 8" in screen(edit)[0]
    await users.editor.cancel(42)
    draft = await users.resumes.prepare(42, "Навыки: Python\nОпыт работы: 3 года")
    await users.editor.open(42, resume_token=draft.token)
    await users.resumes.delete(42)
    assert await users.editor.get(42) is None
    async with editor_sessions() as session:
        assert await session.get(ProfileEditModel, 42) is None


async def test_calendar_union_projects_unknown_and_ongoing_are_conservative(
    editor_sessions, vacancy
):
    users = UserService(editor_sessions)
    original, first = await seed_profile(users)
    now = datetime.now(UTC)
    v = job(vacancy, now, "Требования: Python\nОбязанности:\nРазрабатывать отчёты.")
    initial = compare_facts(v, original, now=now)
    edit = await users.editor.open(42)
    edit = await change(users.editor, 42, edit, "episode", "", "new", "employment")
    second = edit.view[2]
    for field, value in {"roles": "Разработчик", "started": "2022-01", "ended": "2023-12"}.items():
        edit = await change(users.editor, 42, edit, "episode", second, field, value)
    edit = await change(users.editor, 42, edit, "episode", "", "new", "project")
    project = edit.view[2]
    for field, value in {
        "type": "academic",
        "roles": "Разработчик",
        "started": "2018-01",
        "ended": "2024-12",
    }.items():
        edit = await change(users.editor, 42, edit, "episode", project, field, value)
    await users.editor.act(42, edit.token, ["save"])
    current = compare_facts(v, (await users.get(42)).profile, now=now)
    assert current.features["role_exposure"] == initial.features["role_exposure"]
    assert current.features["project_match"] == 1
    edit = await users.editor.open(42)
    edit = await change(users.editor, 42, edit, "episode", first, "started", "?")
    edit = await change(users.editor, 42, edit, "episode", second, "ongoing", "yes")
    await users.editor.act(42, edit.token, ["save"])
    current = (await users.get(42)).profile
    assert (
        compare_facts(v, current, now=now).features["role_exposure"]
        == compare_facts(v, current, now=now + timedelta(days=366)).features["role_exposure"]
    )  # a frozen snapshot is not a year of new work


async def test_resume_review_and_confirmation_buttons_use_edited_snapshot(editor_sessions):
    users = UserService(editor_sessions)
    chat = Chat(users)
    await chat.send("/start")
    await chat.send("/resume Навыки: Python\nОпыт работы\n01.2020 — 12.2021\nДолжность: Инженер")
    old_confirm = chat.data("Применить к профилю")
    await chat.click("Проверить и исправить")
    await chat.click("Опыт и проекты (1)")
    edit = await users.editor.get(42)
    _, actions = screen(edit)
    label = next(label for label, action in actions if action[:3] == ["view", "item", "episode"])
    await chat.click(label)
    await chat.field("Организация", "Исправлено в Telegram")
    await chat.click("Вернуться к подтверждению импорта")
    await chat.dispatcher.feed_update(chat.bot, callback_update(old_confirm))
    assert not (await users.get(42)).profile.professional_facts.episodes
    await chat.click("Применить к профилю")
    assert (await users.get(42)).profile.professional_facts.episodes[
        0
    ].organization.value == "Исправлено в Telegram"
    await chat.bot.session.close()


async def test_late_inference_cannot_replace_fact_only_manual_edit(editor_sessions):
    from domain.resume import ResumeExtraction

    users = UserService(editor_sessions)
    _, key = await seed_profile(users)
    entered, release = asyncio.Event(), asyncio.Event()

    async def analyze(text):
        entered.set()
        await release.wait()
        return ResumeExtraction(skills=["sql"])

    users.resumes.analyzer = SimpleNamespace(analyze=analyze)
    pending = asyncio.create_task(
        users.resumes.prepare(42, "Свободное описание профессионального пути кандидата")
    )
    await asyncio.wait_for(entered.wait(), 5)
    original = (await users.get(42)).profile
    try:
        edit = await users.editor.open(42)
        edit = await change(
            users.editor, 42, edit, "episode", key, "tasks", "Новая подтверждённая задача"
        )
        await users.editor.act(42, edit.token, ["save"])
    finally:
        release.set()
    with pytest.raises(StaleResumeWork):
        await pending
    user = await users.get(42)
    assert profile_payload(user.profile) == profile_payload(original)
    assert any(
        f.name == "Новая подтверждённая задача" for f in user.profile.professional_facts.facts
    )
    assert await users.resumes.pending(42) is None


async def test_characteristics_qualifications_and_preference_clears_survive_import(
    editor_sessions, vacancy
):
    users = UserService(editor_sessions)
    await seed_profile(users)
    edit = await users.editor.open(42)
    skill = next(f for f in edit.profile.professional_facts.facts if f.kind == "skill")
    for field, value in {"years": "4", "level": "advanced", "last_used": "2025-06-15"}.items():
        edit = await change(users.editor, 42, edit, "skill", skill.id, field, value)
    for field, value in {
        "salary": "150000 RUB на руки",
        "locations": "Москва",
        "constraints": "both",
        "qualifications": "медкнижка до 2027-12-31; нет: права B",
    }.items():
        edit = await change(users.editor, 42, edit, "basic", "", field, value)
    await users.editor.act(42, edit.token, ["save"])
    now = datetime.now(UTC)
    v = job(vacancy, now)
    wanted, _ = skill_requirement(v)
    wanted.level = type(wanted.level)(status="present", value="intermediate")
    wanted.last_used = type(wanted.last_used)(status="present", value="2025-01-01")
    compared = compare_facts(v, (await users.get(42)).profile, now=now)
    assert compared.features["skill_level_fit"] == compared.features["skill_recency_fit"] == 1
    edit = await users.editor.open(42)
    for field in ("years", "level", "last_used"):
        edit = await change(users.editor, 42, edit, "skill", skill.id, field, "?")
    for field in ("salary", "locations", "qualifications", "positions"):
        edit = await change(users.editor, 42, edit, "basic", "", field, "?")
    await users.editor.act(42, edit.token, ["save"])
    draft = await users.resumes.prepare(
        42,
        "Желаемая должность: Разработчик\nНавыки: Python — 3 года advanced\nДокументы: медкнижка",
    )
    await users.resumes.confirm(42, draft.token)
    profile = (await users.get(42)).profile
    skill = next(f for f in profile.professional_facts.facts if f.kind == "skill")
    assert all(getattr(skill, field).value is None for field in ("years", "level", "last_used"))
    assert profile.preferences.salary_min is None and not profile.preferences.salary_required
    assert not profile.preferences.locations and not profile.preferences.locations_required
    assert not profile.qualifications
    assert not profile.preferences.positions


async def test_operational_rollback_disables_editor_preserving_saved_data(editor_sessions):
    users = UserService(editor_sessions)
    original, _ = await seed_profile(users)
    disabled = UserService(editor_sessions, profile_editor_enabled=False)
    with pytest.raises(ValueError, match="отключён"):
        await disabled.editor.open(42)
    assert (await disabled.get(42)).profile == original


async def test_explicit_absence_and_unknown_are_distinct_and_not_resurrected(
    editor_sessions, vacancy
):
    users = UserService(editor_sessions)
    await seed_profile(users)
    v = job(vacancy, datetime.now(UTC))
    edit = await users.editor.open(42)
    key = next(f.id for f in edit.profile.professional_facts.facts if f.kind == "skill")
    edit = await change(users.editor, 42, edit, "skill", key, "presence", "no")
    await users.editor.act(42, edit.token, ["save"])
    await users.begin_edit(42, "locations")
    await users.accept_answer(42, "Казань")
    absent = (await users.get(42)).profile
    assert not score_vacancy(v, absent).eligible
    draft = await users.resumes.prepare(42, "Навыки: Python\nОбщий стаж: 5 лет")
    await users.resumes.confirm(42, draft.token)
    assert not score_vacancy(v, (await users.get(42)).profile).eligible
    edit = await users.editor.open(42)
    edit = await change(users.editor, 42, edit, "skill", key, "presence", "?")
    await users.editor.act(42, edit.token, ["save"])
    assert score_vacancy(v, (await users.get(42)).profile).eligible


@pytest.mark.postgres
async def test_additive_migration_preserves_populated_profiles(postgres_sessions):
    import importlib.util
    from pathlib import Path

    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    users = UserService(postgres_sessions)
    original, _ = await seed_profile(users)
    path = Path(__file__).parents[1] / "alembic/versions/0023_profile_edits.py"
    spec = importlib.util.spec_from_file_location("f15_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    engine = postgres_sessions.kw["bind"]
    async with engine.begin() as connection:
        await connection.run_sync(ProfileEditModel.__table__.drop)

        def migrate(connection):
            with Operations.context(MigrationContext.configure(connection)):
                migration.upgrade()
                migration.downgrade()
                migration.upgrade()

        await connection.run_sync(migrate)
    assert (await users.get(42)).profile == original
