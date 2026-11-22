"""Plain text, paginated Telegram screens for the shared durable editor."""

from aiogram import F, Router
from aiogram.dispatcher.event.bases import SkipHandler
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from bot.cards import _short, calendar_notes, experience_label, render_profile, split_message
from bot.keyboards import profile_keyboard
from bot.profile_form import PROMPTS, salary_prompt
from jobs.profile_editor import section_for
from jobs.profiles import SALARY_CURRENCIES
from nlp.extraction import canonical_skill
from nlp.resume import looks_like_resume
from scoring.comparison import candidate_skill_names

LABELS = {
    "episode": "Опыт и проекты",
    "skill": "Навыки",
    "role": "Прошлые роли",
    "task": "Задачи",
    "experience": "Общий стаж",
    "positions": "Должности для поиска",
    "remote": "Формат работы",
    "salary": "Зарплата",
    "locations": "Города",
    "qualifications": "Документы и допуски",
    "constraints": "Обязательные условия",
    "started": "Начало",
    "ended": "Окончание",
    "ongoing": "Продолжается сейчас",
    "roles": "Занимаемые должности",
    "tasks": "Задачи",
    "skills": "Навыки",
    "type": "Вид опыта",
    "name": "Название",
    "presence": "Изменить статус",
}
TYPES = {
    ("employment", "work"): "Коммерческая работа",
    ("project", "work"): "Коммерческий проект",
    ("project", "academic"): "Учебный проект",
    ("project", "personal"): "Личный проект",
}
PAGE = 6
SKILLS_PER_PAGE = 12


def claim_text(claim):
    if claim.status == "conflict":
        return "есть противоречие: " + ", ".join(map(str, claim.alternatives))
    if claim.status == "explicit_absent":
        return "указано отсутствие"
    if claim.value is None:
        return "неизвестно"
    if isinstance(claim.value, bool):
        return "да" if claim.value else "нет"
    return str(claim.value)


def records(profile, kind):
    facts = profile.professional_facts
    if facts is None:
        return []
    return (
        facts.episodes
        if kind == "episode"
        else [
            f
            for f in facts.facts
            if f.kind == kind and not (kind == "role" and f.context == "desired")
        ]
    )


def episode_name(episode, profile):
    indexed = {f.id: f for f in profile.professional_facts.facts}
    work = next(
        (indexed[i].name for i in (*episode.role_ids, *episode.task_ids) if i in indexed), None
    )
    label = episode.title.value or work or episode.organization.value
    label = label or TYPES.get(
        (episode.kind, episode.context), "Проект" if episode.kind == "project" else "Место работы"
    )
    return f"{label} · {str(episode.started.value)[:7]}" if episode.started.value else label


def basic_value(profile, field):
    """Full saved values; summary truncation must not hide editable information."""
    from bot.cards import REMOTE_LABELS

    pref = profile.preferences
    if field == "skills":
        return ", ".join(sorted(candidate_skill_names(profile))) or "не указаны"
    if field == "experience":
        return experience_label(profile) or "неизвестно"
    if field in {"positions", "locations"}:
        return "; ".join(getattr(pref, field)) or "не указаны"
    if field == "remote":
        return REMOTE_LABELS[pref.remote]
    if field == "salary":
        if pref.salary_min is None:
            return "не указана"
        return f"{pref.salary_min} {pref.currency} в месяц, " + (
            "до налогов" if pref.salary_gross else "на руки"
        )
    if field == "constraints":
        return f"Зарплата обязательна: {'да' if pref.salary_required else 'нет'}; " + (
            f"города обязательны: {'да' if pref.locations_required else 'нет'}"
        )
    if field == "qualifications":
        facts = profile.professional_facts
        return (
            "\n".join(
                f"{f.name}: {claim_text(f.assertion)}; срок: {claim_text(f.valid_until)}"
                for f in (facts.facts if facts else [])
                if f.kind == "qualification"
            )
            or "не указаны"
        )
    return "не указано"


def can_edit_skills_as_text(profile):
    # Long lists and explicit absences from existing profiles cannot round-trip
    # through one comma-separated Telegram message. Keep individual editing there.
    names = list(profile.skills)
    return not (
        len(names) > 50
        or len(", ".join(names).encode("utf-16-le")) > 8000
        or any(any(separator in name for separator in (",", ";", "\n")) for name in names)
        or len({canonical_skill(name) for name in names}) != len(names)
        or set(names) != set(candidate_skill_names(profile))
        or any(f.assertion.value is not True for f in records(profile, "skill"))
    )


def screen(edit):
    """Return text and commands; callback indexes are resolved only for this token."""
    profile, view = edit.profile, edit.view
    actions = []

    def button(label, *command):
        actions.append((label, list(command)))

    def back(*path):
        button("← Назад", "view", *(path or ("home",)))

    lines = []
    if view[0] == "home":
        title = "Проверка резюме" if edit.target == "resume" else "Редактирование профиля"
        lines.append(render_profile(profile, title=title))
        if profile.professional_facts and profile.professional_facts.input_scope == "partial":
            lines.append("Часть резюме не удалось разобрать. Проверьте сведения.")
        button("Навыки", "view", "list", "skill", 0)
        button("Условия поиска", "view", "preferences")
        button("Опыт работы", "view", "career")
        button(LABELS["qualifications"], "view", "documents")
    elif view[0] == "documents":
        lines.extend([LABELS["qualifications"], basic_value(profile, "qualifications")])
        button("Изменить список", "view", "input", "basic", "", "qualifications")
        back()
    elif view[0] in {"preferences", "career"}:
        career = view[0] == "career"
        lines.append("Опыт работы" if career else "Условия поиска")
        fields = (
            ("experience",)
            if career
            else ("positions", "locations", "remote", "salary", "constraints")
        )
        for field in fields:
            lines.append(LABELS[field] + ": " + _short(basic_value(profile, field), 120))
            button(LABELS[field], "view", "input", "basic", "", field)
        if career:
            lines.extend(calendar_notes(profile))
            for kind in ("episode", "role", "task"):
                button(f"{LABELS[kind]} ({len(records(profile, kind))})", "view", "list", kind, 0)
        back()
    elif view[0] == "list":
        kind, page = view[1], int(view[2])
        items = records(profile, kind)
        size = SKILLS_PER_PAGE if kind == "skill" else PAGE
        pages = max(1, (len(items) + size - 1) // size)
        page = min(max(0, page), pages - 1)
        lines.append(f"{LABELS[kind]} · {len(items)}")
        if pages > 1:
            lines.append(f"Страница {page + 1} из {pages}")
        for number, item in enumerate(items[page * size : (page + 1) * size], page * size + 1):
            name = episode_name(item, profile) if kind == "episode" else item.name
            if kind == "skill":
                button(_short(name, 40), "view", "item", kind, item.id)
            else:
                lines.append(f"{number}. {name}")
                button(f"{number}. {_short(name, 45)}", "view", "item", kind, item.id)
        if page:
            button("← Предыдущие", "view", "list", kind, page - 1)
        if page + 1 < pages:
            button("Следующие →", "view", "list", kind, page + 1)
        if kind == "episode":
            button("+ Место работы", "set", kind, "", "new", "employment")
            button("+ Проект", "set", kind, "", "new", "project")
        else:
            button("+ Добавить", "view", "input", kind, "", "new")
        if kind == "skill" and can_edit_skills_as_text(profile):
            button("Изменить списком", "view", "input", "basic", "", "skills")
        back("career" if kind != "skill" else "home")
    elif view[0] in {"item", "delete"}:
        kind, key = view[1:]
        items = records(profile, kind)
        index = next((n for n, item in enumerate(items) if item.id == key), None)
        item = items[index] if index is not None else None
        page = (index or 0) // (SKILLS_PER_PAGE if kind == "skill" else PAGE)
        if item is None:
            lines.append("Запись удалена.")
            back("list", kind, 0)
        elif view[0] == "delete":
            lines.append("Удалить эту запись из профиля?")
            if kind == "episode":
                lines.append("Роли и задачи, указанные только в этой записи, тоже будут удалены.")
            button("Да, удалить", "set", kind, key, "delete", "yes", page)
            back("item", kind, key)
        else:
            if kind == "episode":
                indexed = {f.id: f for f in profile.professional_facts.facts}
                lines.append(TYPES.get((item.kind, item.context), "Вид опыта не уточнён"))
                lines.append(episode_name(item, profile))
                for field in ("started", "ended", "ongoing"):
                    lines.append(LABELS[field] + ": " + claim_text(getattr(item, field)))
                for field, ids in (
                    ("roles", item.role_ids),
                    ("tasks", item.task_ids),
                ):
                    lines.append(
                        LABELS[field]
                        + ": "
                        + ("; ".join(indexed[i].name for i in ids) or "не указаны")
                    )
                fields = [
                    "type",
                    "roles",
                    "started",
                    "ended",
                    "ongoing",
                    "tasks",
                ]
            else:
                lines.append(item.name)
                if kind != "skill" or item.assertion.value is not True:
                    lines.append("Статус: " + claim_text(item.assertion))
                fields = ["name", "presence"]
                linked = [
                    episode_name(e, profile)
                    for e in records(profile, "episode")
                    if key in getattr(e, kind + "_ids")
                ]
                if linked:
                    lines.append("Указано в записях об опыте: " + "; ".join(linked))
            for field in fields:
                button(LABELS[field], "view", "input", kind, key, field)
            button("Удалить…", "view", "delete", kind, key)
            back("list", kind, page)
    elif view[0] == "input":
        kind, key, field = view[1:]
        lines.append(LABELS.get(field, "Новая запись"))
        if kind == "basic":
            lines.append("Сейчас: " + basic_value(profile, field))
            if field == "salary":
                currency = key or profile.preferences.currency
                lines.append(salary_prompt(profile.preferences, currency))
                for code in SALARY_CURRENCIES:
                    button(code, "view", "input", kind, code, field)
            elif field == "skills":
                lines.append("Отправьте новый список навыков через запятую целиком.")
            else:
                lines.append(PROMPTS[field])
        elif field in {"roles", "tasks"}:
            lines.append(
                "Укажите полный список через точку с запятой. Новый список заменит прежний "
                "в этой записи. Для удаления отдельного пункта отправьте список без него."
            )
        elif field in {"started", "ended"}:
            lines.append(
                "Введите дату: 2020-03 или 15.03.2020. Неизвестный месяц можно пропустить."
            )
        elif field == "presence":
            lines.append(
                "Вы подтверждаете этот навык, прошлую роль или выполнявшуюся задачу? "
                "«Нет» означает явное отсутствие; «Не знаю» оставляет неопределённость."
            )
        elif field not in {"ongoing", "type"}:
            lines.append("Введите название.")
        choices = {
            "ongoing": [("Да", "yes"), ("Нет", "no")],
            "presence": [("Да", "yes"), ("Нет", "no")],
            "type": [
                ("Коммерческая работа", "work"),
                ("Коммерческий проект", "commercial"),
                ("Учебный проект", "academic"),
                ("Личный проект", "personal"),
                ("Не знаю", "unknown"),
            ],
            "remote": [
                ("Любой", "any"),
                ("Предпочтительно удалённо", "preferred"),
                ("Только удалённо", "required"),
                ("На месте работы", "office"),
            ],
            "constraints": [
                ("Не обязательны", "none"),
                ("Зарплата", "salary"),
                ("Города", "locations"),
                ("Зарплата и города", "both"),
            ],
        }.get(field, [])
        for label, value in choices:
            button(label, "set", kind, key, field, value)
        if field not in {"name", "new", "type", "remote", "constraints"}:
            button(
                "Рассчитать по датам"
                if field == "experience"
                else "Очистить"
                if kind == "basic"
                else "Не знаю / очистить",
                "set",
                kind,
                key,
                field,
                "?",
            )
        if kind == "basic" and field == "skills":
            back("list", "skill", 0)
        elif kind == "basic":
            back(section_for(field))
        elif key:
            back("item", kind, key)
        else:
            back("list", kind, 0)
    if edit.dirty and view[0] != "input":
        lines.append("Есть несохранённые изменения.")
        button(
            "Сохранить правки" if edit.target == "profile" else "Вернуться к подтверждению импорта",
            "save",
        )
    if view[0] == "home":
        button("Отменить все правки" if edit.dirty else "Закрыть", "cancel")
    separator = "\n\n" if view[0] in {"home", "input"} else "\n"
    return separator.join(lines), actions


async def show_editor(message, edit, *, replace=False):
    text, actions = screen(edit)
    rows = []
    previous_group = None
    for index, (label, command) in enumerate(actions):
        key = InlineKeyboardButton(text=label, callback_data=f"pd5:{edit.token}:{index}")
        group, width = None, 2
        if edit.view[:2] == ["list", "skill"] and command[:2] == ["view", "item"]:
            group = "skills"
        elif edit.view[0] == "list" and command[:2] == ["view", "list"]:
            group = "pages"
        elif edit.view[0] == "input" and edit.view[-1] == "salary" and label in SALARY_CURRENCIES:
            group, width = "currencies", 3
        elif (
            edit.view[0] in {"item", "preferences", "career"}
            and command[0] == "view"
            and command[1] in {"input", "list"}
            and len(label) <= 20
        ):
            group = "fields"
        if group and group == previous_group and len(rows[-1]) < width:
            rows[-1].append(key)
        else:
            rows.append([key])
        previous_group = group
    keyboard = InlineKeyboardMarkup(inline_keyboard=rows)
    chunks = split_message(text)
    if replace and len(chunks) == 1:
        await message.edit_text(text, parse_mode=None, reply_markup=keyboard)
        return
    for index, chunk in enumerate(chunks):
        await message.answer(
            chunk, parse_mode=None, reply_markup=keyboard if index == len(chunks) - 1 else None
        )


def register_profile_editor(router: Router, service):
    async def finished(message, edit, action):
        if edit.target == "resume":
            from bot.resume_cards import render_resume, resume_keyboard

            draft = await service.resumes.pending(message.chat.id)
            user = await service.get(message.chat.id)
            if draft is None:
                await message.answer(
                    render_profile(user.profile), reply_markup=profile_keyboard(), parse_mode=None
                )
                return
            for index, chunk in enumerate(
                chunks := split_message(render_resume(draft, user.profile))
            ):
                await message.answer(
                    chunk,
                    parse_mode=None,
                    reply_markup=resume_keyboard(draft, user.profile)
                    if index == len(chunks) - 1
                    else None,
                )
        else:
            user = await service.get(message.chat.id)
            title = "Профиль сохранён" if action[0] == "save" else "Мой профиль"
            await message.edit_text(
                render_profile(user.profile, title=title),
                reply_markup=profile_keyboard(),
                parse_mode=None,
            )

    @router.callback_query(F.data.startswith("details:"))
    async def begin(query: CallbackQuery, user_id: int):
        section = query.data.split(":", 1)[1]
        if section == "skills":
            section = "skill"
        view = ["list", section, 0] if section in {"episode", "role", "task", "skill"} else ["home"]
        if section in PROMPTS and section != "skills":
            view = ["input", "basic", "", section]
        try:
            edit = await service.editor.open(user_id, view=view)
            await query.answer()
            await show_editor(query.message, edit, replace=True)
        except ValueError as exc:
            await query.answer(str(exc)[:200], show_alert=True)

    @router.callback_query(F.data.startswith(("pd:", "pd2:", "pd3:", "pd4:")))
    async def old_editor(query: CallbackQuery):
        # Previous versions indexed a different set of actions on the same saved draft.
        await query.answer("Меню обновилось. Откройте черновик через /profile.", show_alert=True)

    @router.callback_query(F.data.startswith("pd5:"))
    async def action(query: CallbackQuery, user_id: int):
        try:
            _, token, number = query.data.split(":")
            edit = await service.editor.get(user_id)
            if edit is None or edit.token != token:
                raise ValueError("Кнопка устарела. Откройте /profile или /resume.")
            _, actions = screen(edit)
            if not number.isdecimal() or not 0 <= int(number) < len(actions):
                raise ValueError("Некорректная кнопка")
            command = actions[int(number)][1]
            result = await service.editor.act(user_id, token, command)
            await query.answer()
            if result:
                await show_editor(query.message, result, replace=True)
            else:
                await finished(query.message, edit, command)
        except ValueError as exc:
            await query.answer(str(exc)[:200], show_alert=True)

    @router.message(F.text & ~F.text.startswith("/"))
    async def answer(message: Message, user_id: int):
        edit = await service.editor.get(user_id)
        if edit is None or edit.view[0] != "input" or looks_like_resume(message.text):
            raise SkipHandler
        try:
            result = await service.editor.act(user_id, edit.token, ["text"], text=message.text)
            await show_editor(message, result)
        except ValueError as exc:
            await message.answer(str(exc)[:500], parse_mode=None)
