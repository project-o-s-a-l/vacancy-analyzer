"""Plain text, paginated Telegram screens for the shared durable editor."""

from aiogram import F, Router
from aiogram.dispatcher.event.bases import SkipHandler
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from bot.cards import _short, render_profile, split_message
from bot.profile_form import PROMPTS

LABELS = {
    "episode": "Опыт и проекты",
    "skill": "Навыки",
    "role": "Прошлые роли",
    "task": "Задачи",
    "experience": "Общий стаж",
    "positions": "Желаемые должности",
    "remote": "Формат работы",
    "salary": "Зарплата",
    "locations": "Города",
    "qualifications": "Документы и допуски",
    "constraints": "Обязательные условия",
    "title": "Название записи / проекта",
    "organization": "Организация",
    "started": "Начало",
    "ended": "Окончание",
    "ongoing": "Продолжается сейчас",
    "roles": "Занимаемые должности",
    "tasks": "Задачи",
    "skills": "Навыки этого опыта",
    "type": "Вид опыта",
    "parent": "Связь с местом работы",
    "name": "Название",
    "years": "Лет применения навыка",
    "level": "Уровень навыка",
    "last_used": "Последнее использование",
    "presence": "Уточнить наличие",
}
TYPES = {
    ("employment", "work"): "Коммерческая работа",
    ("project", "work"): "Коммерческий проект",
    ("project", "academic"): "Учебный проект",
    ("project", "personal"): "Личный проект",
}
LEVELS = {"beginner": "Начальный", "intermediate": "Средний", "advanced": "Продвинутый"}
PAGE = 6


def claim_text(claim):
    if claim.status == "conflict":
        return "есть противоречие: " + ", ".join(map(str, claim.alternatives))
    if claim.status == "explicit_absent":
        return "указано отсутствие"
    if claim.value is None:
        return "неизвестно"
    if isinstance(claim.value, bool):
        return "да" if claim.value else "нет"
    return LEVELS.get(str(claim.value), str(claim.value))


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
    roles = [indexed[i].name for i in episode.role_ids if i in indexed]
    return episode.title.value or episode.organization.value or "; ".join(roles) or "Без названия"


def basic_value(profile, field):
    """Full saved values; summary truncation must not hide editable information."""
    from bot.cards import REMOTE_LABELS
    from scoring.comparison import total_experience

    pref = profile.preferences
    if field == "experience":
        return claim_text(total_experience(profile))
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


def screen(edit):
    """Return text and commands; callback indexes are resolved only for this token."""
    profile, view = edit.profile, edit.view
    actions = []

    def button(label, *command):
        actions.append((label, list(command)))

    def back(*path):
        button("← Назад", "view", *(path or ("home",)))

    title = "Проверка импорта" if edit.target == "resume" else "Редактор профиля"
    lines = [title + ". Изменения ещё не применены."]
    if view[0] == "home":
        if profile.professional_facts and profile.professional_facts.input_scope == "partial":
            lines.append("Сохранённый разбор неполный; неизвестные сведения можно заполнить здесь.")
        lines += [
            render_profile(profile),
            "Прошлая должность описывает ваш опыт; желаемая — направление поиска.",
            "Сведения сохраняются с ваших слов, без внешней проверки.",
        ]
        for kind in ("episode", "skill", "role", "task"):
            button(f"{LABELS[kind]} ({len(records(profile, kind))})", "view", "list", kind, 0)
        button("Предпочтения, общий стаж и документы", "view", "preferences")
    elif view[0] == "preferences":
        for field in (
            "experience",
            "positions",
            "remote",
            "salary",
            "locations",
            "qualifications",
            "constraints",
        ):
            lines.append(LABELS[field] + ": " + basic_value(profile, field))
            button(LABELS[field], "view", "input", "basic", "", field)
        back()
    elif view[0] == "list":
        kind, page = view[1], int(view[2])
        items = records(profile, kind)
        page = min(max(0, page), max(0, (len(items) - 1) // PAGE))
        lines.append(f"{LABELS[kind]}: {len(items)}. Страница {page + 1}.")
        for number, item in enumerate(items[page * PAGE : (page + 1) * PAGE], page * PAGE + 1):
            name = episode_name(item, profile) if kind == "episode" else item.name
            lines.append(f"{number}. {name}")
            button(f"{number}. {_short(name, 45)}", "view", "item", kind, item.id)
        if page:
            button("← Предыдущие", "view", "list", kind, page - 1)
        if (page + 1) * PAGE < len(items):
            button("Следующие →", "view", "list", kind, page + 1)
        if kind == "episode":
            button("+ Место работы", "set", kind, "", "new", "employment")
            button("+ Проект", "set", kind, "", "new", "project")
        else:
            button("+ Добавить", "view", "input", kind, "", "new")
        back()
    elif view[0] in {"item", "delete"}:
        kind, key = view[1:]
        item = next((i for i in records(profile, kind) if i.id == key), None)
        if item is None:
            lines.append("Запись удалена.")
            back("list", kind, 0)
        elif view[0] == "delete":
            lines.append(
                "Удалить запись и её связи? Задачи и роли, связанные только с этой работой, "
                "будут удалены. Навыки останутся в общем списке; вложенные проекты "
                "останутся отдельными записями. История прежних ревизий сохраняется."
            )
            button("Да, удалить", "set", kind, key, "delete", "yes")
            back("item", kind, key)
        else:
            if kind == "episode":
                indexed = {f.id: f for f in profile.professional_facts.facts}
                lines.append(TYPES.get((item.kind, item.context), "Вид опыта не уточнён"))
                for field in ("title", "organization", "started", "ended", "ongoing"):
                    lines.append(LABELS[field] + ": " + claim_text(getattr(item, field)))
                for field, ids in (
                    ("roles", item.role_ids),
                    ("tasks", item.task_ids),
                    ("skills", item.skill_ids),
                ):
                    lines.append(
                        LABELS[field]
                        + ": "
                        + ("; ".join(indexed[i].name for i in ids) or "не указаны")
                    )
                parent = next((e for e in records(profile, kind) if e.id == item.parent_id), None)
                lines.append(
                    "Связь с работой: " + (episode_name(parent, profile) if parent else "нет")
                )
                lines.append(
                    "Связь с навыком не означает его использование весь период. "
                    "Общий стаж задаётся отдельно. Пересечения периодов не складываются."
                )
                fields = [
                    "title",
                    "organization",
                    "type",
                    "roles",
                    "started",
                    "ended",
                    "ongoing",
                    "tasks",
                    "skills",
                ]
                if item.kind == "project":
                    fields.append("parent")
            else:
                lines += [item.name, "Наличие: " + claim_text(item.assertion)]
                fields = ["name", "presence"]
                if kind == "skill":
                    fields += ["years", "level", "last_used"]
                    for field in ("years", "level", "last_used"):
                        lines.append(LABELS[field] + ": " + claim_text(getattr(item, field)))
                linked = [
                    episode_name(e, profile)
                    for e in records(profile, "episode")
                    if key in getattr(e, kind + "_ids")
                ]
                lines.append("Связан с опытом: " + ("; ".join(linked) or "нет"))
            for field in fields:
                button(LABELS[field], "view", "input", kind, key, field)
            button("Удалить…", "view", "delete", kind, key)
            back("list", kind, 0)
    elif view[0] == "input":
        kind, key, field = view[1:]
        lines.append("Изменить: " + LABELS.get(field, "Название новой записи"))
        if kind == "basic":
            lines.append("Сейчас: " + basic_value(profile, field))
            lines.append(PROMPTS[field])
        elif field in {"roles", "tasks", "skills"}:
            lines.append(
                "Укажите полный список через точку с запятой. Новый список заменит прежний "
                "в этой записи. Для удаления отдельного пункта отправьте список без него."
            )
            if field == "skills":
                lines.append(
                    "Используйте названия из раздела «Навыки»; сначала добавьте новые там."
                )
        elif field == "last_used":
            lines.append(
                "Точная дата последнего применения: 15.03.2025. Если не помните — «Не знаю»."
            )
        elif field in {"started", "ended"}:
            lines.append(
                "Введите дату: 2020-03 или 15.03.2020. Неизвестный месяц можно пропустить."
            )
        elif field == "years":
            lines.append("Сколько лет вы применяли именно этот навык? От 0 до 80, например 1.5.")
        elif field == "presence":
            lines.append(
                "Вы подтверждаете этот навык, прошлую роль или выполнявшуюся задачу? "
                "«Нет» означает явное отсутствие; «Не знаю» оставляет неопределённость."
            )
        elif field not in {"level", "ongoing", "type", "parent"}:
            lines.append("Введите текст. Организация и название проекта служат контекстом.")
        choices = {
            "level": [(v, k) for k, v in LEVELS.items()],
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
        if field == "parent":
            # Parent choices are paginated through a separate view.
            button("Выбрать работу", "view", "parents", key, 0)
        for label, value in choices:
            button(label, "set", kind, key, field, value)
        if field not in {"name", "new", "type"}:
            button("Не знаю / очистить", "set", kind, key, field, "?")
        if kind == "basic":
            back("preferences")
        elif key:
            back("item", kind, key)
        else:
            back("list", kind, 0)
        lines.append("«Назад» пропускает поле и сохраняет прежнее значение.")
    elif view[0] == "parents":
        key, page = view[1], int(view[2])
        items = [e for e in records(profile, "episode") if e.kind == "employment"]
        lines.append("Выберите место работы для проекта.")
        for item in items[page * PAGE : (page + 1) * PAGE]:
            button(
                _short(episode_name(item, profile), 45), "set", "episode", key, "parent", item.id
            )
        if page:
            button("← Предыдущие", "view", "parents", key, page - 1)
        if (page + 1) * PAGE < len(items):
            button("Следующие →", "view", "parents", key, page + 1)
        back("item", "episode", key)
    button(
        "Сохранить правки" if edit.target == "profile" else "Вернуться к подтверждению импорта",
        "save",
    )
    button("Отменить все правки", "cancel")
    return "\n\n".join(lines), actions


async def show_editor(message, edit):
    text, actions = screen(edit)
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=label, callback_data=f"pd:{edit.token}:{index}")]
            for index, (label, _) in enumerate(actions)
        ]
    )
    chunks = split_message(text)
    for index, chunk in enumerate(chunks):
        await message.answer(
            chunk, parse_mode=None, reply_markup=keyboard if index == len(chunks) - 1 else None
        )


def register_profile_editor(router: Router, service):
    async def finished(message, edit, action):
        if action[0] == "save" and edit.target == "resume":
            from bot.resume_cards import render_resume, resume_keyboard

            draft = await service.resumes.pending(message.chat.id)
            user = await service.get(message.chat.id)
            for index, chunk in enumerate(
                chunks := split_message(render_resume(draft, user.profile))
            ):
                await message.answer(
                    chunk,
                    parse_mode=None,
                    reply_markup=resume_keyboard(draft) if index == len(chunks) - 1 else None,
                )
        else:
            from bot.keyboards import profile_keyboard

            user = await service.get(message.chat.id)
            await message.answer(
                ("Правки сохранены.\n\n" if action[0] == "save" else "Правки отменены.\n\n")
                + render_profile(user.profile),
                reply_markup=profile_keyboard(),
                parse_mode=None,
            )

    @router.callback_query(F.data.startswith("details:"))
    async def begin(query: CallbackQuery, user_id: int):
        section = query.data.split(":", 1)[1]
        view = ["list", section, 0] if section in {"episode", "skill", "role", "task"} else ["home"]
        if section in PROMPTS and section != "skills":
            view = ["input", "basic", "", section]
        try:
            edit = await service.editor.open(user_id, view=view)
            await query.answer()
            await show_editor(query.message, edit)
        except ValueError as exc:
            await query.answer(str(exc)[:200], show_alert=True)

    @router.callback_query(F.data.startswith("pd:"))
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
                await show_editor(query.message, result)
            else:
                await finished(query.message, edit, command)
        except ValueError as exc:
            await query.answer(str(exc)[:200], show_alert=True)

    @router.message(F.text & ~F.text.startswith("/"))
    async def answer(message: Message, user_id: int):
        edit = await service.editor.get(user_id)
        if edit is None:
            raise SkipHandler
        try:
            result = await service.editor.act(user_id, edit.token, ["text"], text=message.text)
            await show_editor(message, result)
        except ValueError as exc:
            await message.answer(str(exc)[:500], parse_mode=None)
