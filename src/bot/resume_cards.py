from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from bot.cards import render_profile
from domain import CandidateProfile
from domain.resume import ResumeDraft
from jobs.resumes import prepared_profile
from nlp.resume_document import EXPERIENCE_NOTES

RESUME_PROMPT = (
    "Прикрепите резюме в PDF, DOCX или TXT до 2 МБ. Для PDF подойдёт файл с текстом "
    "до 20 страниц. Можно также отправить /resume и вставить текст после команды.\n\n"
    "Вы увидите профиль и сможете исправить его перед сохранением. "
    "Сервис хранит извлечённый текст и обрабатывает его локально. "
    "/resume_delete удаляет текст загрузки и черновик. Подтверждённые сведения "
    "и цитаты остаются в профиле и истории."
)


def resume_keyboard(draft: ResumeDraft, current: CandidateProfile) -> InlineKeyboardMarkup:
    def button(action, title):
        return [InlineKeyboardButton(text=title, callback_data=f"resume:{action}:{draft.token}")]

    rows = [button("confirm", "Применить к профилю"), button("review", "Проверить и исправить")]
    facts = current.professional_facts
    has_experience = current.experience_years is not None or (facts and facts.episodes)
    if draft.reviewed_profile is None:
        if draft.apply_mode == "replace_all":
            rows.append(button("merge", "Дополнить профиль…"))
        elif (
            current.skills
            or current.preferences.positions
            or current.qualifications
            or has_experience
            or (facts and facts.reviewed_sections)
        ):
            rows.append(button("replace_all", "Заменить данные профиля…"))
    rows.append(button("cancel", "Отменить загрузку"))
    return InlineKeyboardMarkup(inline_keyboard=rows)


def render_resume(draft: ResumeDraft, current: CandidateProfile) -> str:
    owner = draft.professional_facts.owner_id if draft.professional_facts else 1
    profile = prepared_profile(current, draft, owner)
    summary = render_profile(profile, title="Предварительный профиль из резюме — ещё не применён")
    extracted = draft.extraction
    roles = list(
        dict.fromkeys(
            o.name for o in extracted.observations if o.kind == "role" and o.context != "desired"
        )
    )
    tasks = list(dict.fromkeys(o.name for o in extracted.observations if o.kind == "task"))
    if profile.professional_facts:
        roles = [
            f.name
            for f in profile.professional_facts.facts
            if f.kind == "role" and f.context != "desired"
        ]
        tasks = [f.name for f in profile.professional_facts.facts if f.kind == "task"]
    # The summary shows the effective profile; /resume_details retains raw source calculations.
    notes = [n for n in extracted.notes if not n.startswith(EXPERIENCE_NOTES)]
    if draft.apply_mode != "replace_all" and current.professional_facts:
        labels = {
            "career": "опыт, роли и задачи",
            "skills": "навыки",
            "experience": "общий стаж",
            "qualifications": "документы",
            "positions": "желаемые должности",
            "salary": "зарплата",
            "locations": "города",
            "remote": "формат работы",
            "constraints": "обязательные условия",
        }
        protected = [
            labels[key] for key in current.professional_facts.reviewed_sections if key in labels
        ]
        if protected:
            notes.append("Сохранены вручную изменённые разделы: " + ", ".join(protected) + ".")
    if extracted.completeness != "complete":
        notes.insert(0, "Разбор неполный — проверьте неразобранные части в /resume_details.")
    if not extracted.skills and current.skills:
        notes.append("Навыки не определены. Показаны прежние значения профиля.")
    added = sorted(set(profile.skills) - set(current.skills))
    removed = sorted(set(current.skills) - set(profile.skills))
    changes = []
    if added:
        changes.append("Добавятся навыки: " + ", ".join(added))
    if removed:
        changes.append("Будут удалены навыки: " + ", ".join(removed))
    sections = [summary]
    career = []
    if roles:
        career.append("Прошлые должности: " + ", ".join(roles[:5]))
    if tasks:
        career.append("Задачи: " + ", ".join(tasks[:3]))
    if career:
        sections.append("\n".join(career))
    if current.skills and changes:
        sections.append("\n".join(changes))
    sections.append(
        {
            "merge": "Резюме дополнит ваш профиль.",
            "replace": "Неподтверждённые списки будут заменены данными резюме.",
            "replace_all": "Полная замена: прежние навыки, опыт и документы будут заменены.",
        }[draft.apply_mode]
    )
    if notes:
        sections.append("\n".join(dict.fromkeys(notes)))
    sections.append("Сведения из исходного текста доступны по /resume_details.")
    sections.append("Новая загрузка заменит этот предпросмотр и его правки.")
    return "\n\n".join(sections)


def render_resume_details(draft: ResumeDraft) -> str:
    extracted = draft.extraction
    lines = ["Сведения для проверки (ещё не применены):"]
    if draft.reviewed_profile is not None:
        lines.append(
            "Ниже — исходные наблюдения извлечения. Вы уже исправили предложение: "
            "точный результат откройте кнопкой «Проверить и исправить» в /resume."
        )
    for item in extracted.preferences:
        pref = item.values
        if item.group == "locations":
            values = "Города: " + (", ".join(pref.locations) or "ограничение снято")
            values += " (обязательно)" if pref.locations_required else " (пожелание)"
        elif item.group == "salary":
            values = "Зарплата: " + (
                f"от {pref.salary_min} {pref.currency} в месяц "
                + ("до налога" if pref.salary_gross else "на руки")
                if pref.salary_min is not None
                else "ограничение снято"
            )
            values += " (обязательно)" if pref.salary_required else " (пожелание)"
        else:
            values = {
                "required": "Только удалённая работа",
                "preferred": "Желательна удалённая работа",
            }.get(pref.remote, "Ограничение формата работы снято")
        source = (
            draft.text[item.start : item.end] if item.start is not None else "Правка в черновике"
        )
        lines.append(f"{values}\nИсточник: {source}")
    labels = {
        "skill": "Навык",
        "role": "Роль",
        "task": "Задача/проект",
        "qualification": "Квалификация",
    }
    for index, item in enumerate(extracted.observations, 1):
        lines.append(
            f"{index}. {labels[item.kind]}: {item.name}\n"
            f"  Источник: {draft.text[item.start : item.end]}"
        )
        if item.detail_start is not None:
            lines.append("  Самооценка навыка: " + draft.text[item.detail_start : item.detail_end])
    lines.append(
        "Убрать ошибочный факт из предложения: /resume_edit remove НОМЕР. "
        "Для исправления периодов используйте кнопку «Проверить и исправить» в /resume; "
        "общий стаж можно задать через /resume_edit experience."
    )
    for period in extracted.periods:
        kind = (
            "работа"
            if period.kind == "employment"
            else "учебный проект"
            if period.context == "academic"
            else "проект"
            if period.kind == "project"
            else "вид опыта неизвестен"
        )
        lines.append(f"{kind}: {period.date_start} — {period.date_end}")
    reasons = {
        "needs_semantic_review": "нужна смысловая проверка",
        "model_unavailable": "модель недоступна",
        "timeout": "истекло время запроса",
        "total_timeout": "истёк бюджет времени",
        "invalid_response": "ответ не прошёл проверку",
        "invalid_evidence": "часть утверждений не подтверждена текстом",
    }
    for section in extracted.coverage:
        if section.status == "unresolved":
            lines.append(
                "Не разобрано ("
                + reasons.get(section.reason, "нужна проверка")
                + "):\n"
                + draft.text[section.start : section.end]
            )
    lines += extracted.notes
    return "\n\n".join(lines)
