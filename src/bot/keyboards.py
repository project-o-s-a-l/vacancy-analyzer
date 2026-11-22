from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from domain import Vacancy
from domain.feedback import REASONS, FeedbackState
from domain.vacancy import SOURCE_NAMES
from jobs.profiles import SALARY_CURRENCIES


def main_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Найти вакансии", callback_data="menu:latest")],
            [
                InlineKeyboardButton(text="Мои вакансии", callback_data="menu:library"),
                InlineKeyboardButton(text="Мой профиль", callback_data="menu:profile"),
            ],
        ]
    )


def library_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Сохранённые", callback_data="fbpage:saved:0"),
                InlineKeyboardButton(text="Мои отклики", callback_data="fbpage:applied:0"),
            ],
            [InlineKeyboardButton(text="История рекомендаций", callback_data="fbpage:history:0")],
            [InlineKeyboardButton(text="← В меню", callback_data="menu:home")],
        ]
    )


def vacancy_keyboard(
    vacancy: Vacancy, state: FeedbackState | None = None, *, expanded: bool = False
) -> InlineKeyboardMarkup:
    state = state or FeedbackState(vacancy.id)
    identifier = vacancy.id.hex

    def button(action: str, text: str) -> InlineKeyboardButton:
        return InlineKeyboardButton(text=text, callback_data=f"fb:{action}:{identifier}")

    label = (
        SOURCE_NAMES.get(vacancy.source, vacancy.source)
        if vacancy._source_links
        else "Открыть вакансию"
    )
    rows = [[InlineKeyboardButton(text=label, url=str(vacancy.url))]]
    for source, url in sorted(vacancy._source_links.items()):
        rows.append([InlineKeyboardButton(text=SOURCE_NAMES.get(source, source), url=url)])
    rows.append(
        [
            button("like", "👍 Интересно ✓" if state.interest == "like" else "👍 Интересно"),
            button(
                "dislike",
                "👎 Не подходит ✓" if state.interest == "dislike" else "👎 Не подходит",
            ),
        ]
    )
    if expanded:
        rows.extend(
            [
                [
                    InlineKeyboardButton(
                        text="Как подобрана вакансия", callback_data=f"card:match:{identifier}"
                    )
                ],
                [
                    button(
                        "unapply" if state.applied else "apply",
                        "Снять отметку об отклике" if state.applied else "Отметить отклик",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text="← К вакансии", callback_data=f"card:back:{identifier}"
                    )
                ],
            ]
        )
    else:
        rows.append(
            [
                button(
                    "unsave" if state.saved else "save",
                    "Убрать из сохранённых" if state.saved else "Сохранить",
                ),
                InlineKeyboardButton(text="Ещё действия", callback_data=f"card:more:{identifier}"),
            ]
        )
    return InlineKeyboardMarkup(inline_keyboard=rows)


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


def page_keyboard(kind: str, page: int, has_next: bool) -> InlineKeyboardMarkup:
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
    rows.append([InlineKeyboardButton(text="← Мои вакансии", callback_data="menu:library")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


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


def profile_keyboard(*, editing: bool = False) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Продолжить редактирование" if editing else "Изменить профиль",
                    callback_data="details:home",
                )
            ],
            [InlineKeyboardButton(text="Загрузить резюме", callback_data="resume:upload")],
            [InlineKeyboardButton(text="← В меню", callback_data="menu:home")],
        ]
    )


def form_keyboard(field: str, revision: int = 0) -> InlineKeyboardMarkup:
    if field == "constraints":
        return InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text=title, callback_data=f"answer:{revision}:constraints:" + value
                    )
                ]
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
                [
                    InlineKeyboardButton(
                        text=title, callback_data=f"answer:{revision}:remote:" + value
                    )
                ]
                for value, title in values
            ]
        )
    if field != "skills":
        rows = []
        if field == "salary":
            rows.append(
                [
                    InlineKeyboardButton(
                        text=code, callback_data=f"answer:{revision}:salary:{code}"
                    )
                    for code in SALARY_CURRENCIES
                ]
            )
        rows.append(
            [
                InlineKeyboardButton(
                    text="Пропустить", callback_data=f"answer:{revision}:{field}:skip"
                )
            ]
        )
        return InlineKeyboardMarkup(inline_keyboard=rows)
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Загрузить резюме", callback_data="resume:upload")]
        ]
    )
