from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from bot.cards import render_profile
from domain import CandidateProfile
from domain.resume import ResumeDraft
from jobs.resumes import proposed_profile

RESUME_PROMPT = (
    "Отправьте резюме файлом PDF, DOCX или TXT, до 2 МБ и 20 страниц PDF. "
    "PDF должен содержать текст, сканы пока не распознаются. "
    "Можно также отправить /resume и текст резюме после команды.\n\n"
    "Покажу предварительный профиль перед применением. Текст обрабатывается и хранится "
    "в базе этого сервиса, внешним сервисам анализа не отправляется. "
    "Исходный файл не сохраняется. /resume_delete — удалить тексты импортов и черновик. "
    "Подтверждённые сведения и их цитаты остаются в профиле и истории."
)


def resume_keyboard(draft: ResumeDraft) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Применить к профилю", callback_data=f"resume:confirm:{draft.token}"
                )
            ],
            [
                InlineKeyboardButton(
                    text="Отменить загрузку", callback_data=f"resume:cancel:{draft.token}"
                )
            ],
        ]
    )


def render_resume(draft: ResumeDraft, current: CandidateProfile) -> str:
    profile = proposed_profile(current, draft)
    summary = render_profile(profile).split("Чтобы изменить профиль", 1)[0].rstrip()
    summary = summary.replace("Текущий профиль", "Профиль после применения", 1)
    extracted = draft.extraction
    roles = list(
        dict.fromkeys(
            o.name for o in extracted.observations if o.kind == "role" and o.context != "desired"
        )
    )
    tasks = list(dict.fromkeys(o.name for o in extracted.observations if o.kind == "task"))
    notes = list(extracted.notes)
    if extracted.completeness != "complete":
        notes.insert(0, "Разбор неполный — проверьте неразобранные части в /resume_details.")
    if (
        extracted.experience_years != profile.experience_years
        and profile.experience_years is not None
    ):
        notes.append("Показаны прежние значения общего стажа: подтверждённая правка сохранена.")
    if not extracted.skills and current.skills:
        notes.append("Навыки не определены. Показаны прежние значения профиля.")
    added = sorted(set(profile.skills) - set(current.skills))
    removed = sorted(set(current.skills) - set(profile.skills))
    changes = "Добавятся навыки: " + (", ".join(added) or "нет")
    if removed:
        changes += "\nУйдут неподтверждённые навыки: " + ", ".join(removed)
    return (
        "Предварительный профиль из резюме — ещё не применён\n\n"
        + summary
        + "\n\nПрежние роли: "
        + ("; ".join(roles[:5]) or "не определены")
        + "\nЗадачи/проекты: "
        + ("; ".join(tasks[:3]) or "не определены")
        + f"\nПериодов работы/проектов: {len(extracted.periods)}."
        + "\n\n"
        + changes
        + "\nРежим: "
        + ("дополнение" if draft.apply_mode == "merge" else "замена неподтверждённых списков")
        + ". Подтверждённые значения сохраняются. "
        "Уровень и годы навыка без свидетельств неизвестны."
        + "\n"
        + "\n".join("• " + n for n in dict.fromkeys(notes))
        + "\n\nВсе факты и неразобранные части: /resume_details. Исправить перед применением:"
        "\n/resume_edit skills Python, SQL\n/resume_edit positions Разработчик"
        "\n/resume_edit experience 3 (или ? — неизвестно)"
        "\n/resume_edit mode merge (дополнение) или replace (замена)."
        "\nПосле применения исправления доступны через /profile; вакансии — /latest."
    )


def render_resume_details(draft: ResumeDraft) -> str:
    extracted = draft.extraction
    lines = ["Сведения для проверки (ещё не применены):"]
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
        "Для исправления периодов загрузите исправленный текст; "
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
