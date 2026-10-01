from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from domain import Vacancy
from domain.feedback import REASONS, FeedbackState


def vacancy_keyboard(vacancy: Vacancy, state: FeedbackState | None = None) -> InlineKeyboardMarkup:
    state = state or FeedbackState(vacancy.id)
    identifier = vacancy.id.hex

    def button(action: str, text: str) -> InlineKeyboardButton:
        return InlineKeyboardButton(text=text, callback_data=f"fb:{action}:{identifier}")

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Открыть вакансию", url=str(vacancy.url))],
            [
                button("like", "👍 Интересно ✓" if state.interest == "like" else "👍 Интересно"),
                button(
                    "dislike",
                    "👎 Не подходит ✓" if state.interest == "dislike" else "👎 Не подходит",
                ),
            ],
            [
                button(
                    "unsave" if state.saved else "save",
                    "Убрать из сохранённых" if state.saved else "📌 Сохранить",
                ),
                button(
                    "unapply" if state.applied else "apply",
                    "Убрать отметку отклика" if state.applied else "✅ Откликнулся",
                ),
            ],
        ]
    )


def reason_keyboard(state: FeedbackState) -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(
            text=label, callback_data=f"fbr:{state.vacancy_id.hex}:{state.version}:{code}"
        )
        for code, label in REASONS.items()
    ]
    rows = [buttons[index : index + 2] for index in range(0, len(buttons), 2)]
    rows.append(
        [
            InlineKeyboardButton(
                text="Без уточнения",
                callback_data=f"fbr:{state.vacancy_id.hex}:{state.version}:skip",
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def page_keyboard(kind: str, page: int, has_next: bool) -> InlineKeyboardMarkup | None:
    buttons = []
    if page:
        buttons.append(
            InlineKeyboardButton(text="← Назад", callback_data=f"fbpage:{kind}:{page - 1}")
        )
    if has_next:
        buttons.append(
            InlineKeyboardButton(text="Далее →", callback_data=f"fbpage:{kind}:{page + 1}")
        )
    rows = [buttons] if buttons else []
    if kind == "history":
        rows.append([InlineKeyboardButton(text="Очистить историю", callback_data="history:reset")])
    return InlineKeyboardMarkup(inline_keyboard=rows) if rows else None


def history_reset_keyboard(token: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Да, сбросить историю", callback_data=f"history:confirm:{token}"
                )
            ],
            [InlineKeyboardButton(text="Отмена", callback_data=f"history:cancel:{token}")],
        ]
    )


def profile_keyboard() -> InlineKeyboardMarkup:
    fields = [
        ("skills", "Навыки"),
        ("experience", "Опыт"),
        ("positions", "Должности"),
        ("remote", "Формат работы"),
        ("salary", "Зарплата"),
        ("locations", "Города"),
        ("qualifications", "Документы и допуски"),
        ("constraints", "Обязательные условия"),
    ]
    buttons = [
        InlineKeyboardButton(text=title, callback_data="profile:" + field)
        for field, title in fields
    ]
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Загрузить резюме", callback_data="resume:upload")],
            *[buttons[index : index + 2] for index in range(0, len(buttons), 2)],
            [InlineKeyboardButton(text="История рекомендаций", callback_data="fbpage:history:0")],
        ]
    )


def form_keyboard(field: str) -> InlineKeyboardMarkup | None:
    if field == "constraints":
        return InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text=title, callback_data="answer:constraints:" + value)]
                for value, title in [
                    ("none", "Зарплата и города — предпочтения"),
                    ("salary", "Обязательна зарплата"),
                    ("locations", "Обязательны города"),
                    ("both", "Обязательны зарплата и города"),
                ]
            ]
        )
    if field == "remote":
        values = [
            ("any", "Любой"),
            ("preferred", "Предпочтительно удалённо"),
            ("required", "Только удалённо"),
            ("office", "На месте работы"),
        ]
        return InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text=title, callback_data="answer:remote:" + value)]
                for value, title in values
            ]
        )
    if field != "skills":
        return InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="Пропустить", callback_data=f"answer:{field}:skip")]
            ]
        )
    return None
