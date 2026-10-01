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
    "Исходный файл не сохраняется. /resume_delete — удалить сохранённый текст и черновик."
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
    notes = "\n".join("• " + value for value in draft.extraction.notes)
    retained = []
    if not draft.extraction.skills and current.skills:
        retained.append("навыки")
    if not draft.extraction.positions and current.preferences.positions:
        retained.append("должность")
    if draft.extraction.experience_years is None and current.experience_years is not None:
        retained.append("общий стаж")
    if retained:
        notes += (
            "\n• Из резюме не определены: "
            + ", ".join(retained)
            + ". Показаны прежние значения профиля."
        )
    return (
        "Предварительный профиль из резюме — ещё не применён\n\n" + summary + "\n\n"
        "Явно найденные навыки, должность и общий стаж заменят соответствующие поля. "
        "Неопределённые поля и ваши остальные предпочтения сохранятся. "
        "Уровни и годы по навыкам не определяются.\n"
        + notes
        + "\n\nПосле применения проверьте и исправьте данные через /profile, "
        "затем запросите /latest. Стаж по отдельным местам работы не суммируется."
    )
