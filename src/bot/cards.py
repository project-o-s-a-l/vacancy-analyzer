from domain import CandidateProfile, Vacancy
from domain.matching import MatchResult
from domain.vacancy_analysis import CATEGORY_LABELS
from nlp.analysis_identity import current_vacancy_analysis as current_analysis
from scoring.presentation import evidence_label, score_label

EXPERIENCE_LABELS = {
    "no_experience": "Без опыта",
    "1-3": "1–3 года",
    "3-6": "3–6 лет",
    "6+": "Более 6 лет",
}
PERIOD_LABELS = {"MONTH": "мес.", "HOUR": "час", "YEAR": "год", "SHIFT": "смену"}
REMOTE_LABELS = {
    "any": "любой",
    "preferred": "предпочтительно удалённо",
    "required": "только удалённо",
    "office": "на месте работы",
}
QUALIFICATION_LABELS = {
    "declared_present": "есть со слов пользователя; без проверки",
    "declared_absent": "указано, что нет",
    "expired": "указанный срок закончился",
    "unknown": "не указано в профиле",
    "uncertain": "нужна ручная проверка",
    "not_required": "не требуется",
    "context": "упоминание в описании",
}


def split_message(text: str, limit: int = 3500) -> list[str]:
    """Preserve text while bounding Telegram UTF-16 units, including emoji."""
    if limit < 2:
        raise ValueError("Message limit must be at least two UTF-16 units")
    chunks, start, units = [], 0, 0
    for index, character in enumerate(text):
        width = 2 if ord(character) > 0xFFFF else 1
        if units + width > limit:
            chunks.append(text[start:index])
            start, units = index, 0
        units += width
    if start < len(text):
        chunks.append(text[start:])
    return chunks


def _short(value: str, limit: int) -> str:
    # Bound UTF-16 length as well as Python's codepoint count (e.g. emoji).
    value = " ".join(value.split())
    if len(value.encode("utf-16-le")) // 2 <= limit:
        return value
    return value.encode("utf-16-le")[: (limit - 1) * 2].decode("utf-16-le", errors="ignore") + "…"


def _salary(vacancy: Vacancy) -> str:
    lower, upper = vacancy.salary_from, vacancy.salary_to
    if lower is None and upper is None:
        return "Зарплата не указана"
    amount = (
        f"{lower:,}–{upper:,}"
        if lower is not None and upper is not None
        else (f"от {lower:,}" if lower is not None else f"до {upper:,}")
    )
    tax = {True: "до налогов", False: "на руки", None: "налоги не уточнены"}[vacancy.salary_gross]
    period = PERIOD_LABELS.get(vacancy.salary_period or "", "период не указан")
    return f"{amount.replace(',', ' ')} {vacancy.currency or ''} / {period}, {tax}"


def render_card(vacancy: Vacancy, match: MatchResult) -> str:
    """Plain text prevents source-controlled text from becoming Telegram markup."""
    remote = {True: "Удалённо", False: "Без удалёнки", None: "Формат не указан"}[vacancy.remote]
    no_matches = "не определено" if match.components.get("skills") is None else "нет совпадений"
    analysis = current_analysis(vacancy)
    experience = EXPERIENCE_LABELS.get(vacancy.experience or "", "не указан")
    if match.eligibility is not None:
        requirement = match.eligibility.experience
        if requirement.minimum_years is not None:
            minimum = requirement.minimum_years
            source = "по описанию" if requirement.minimum_source == "description" else "HH"
            experience = "Без опыта" if minimum == 0 else f"от {minimum:g} лет ({source})"
            if requirement.conflict:
                experience += "; сведения источников расходятся"
        elif requirement.required:
            experience = "требуется, минимум лет не указан в описании"
        else:
            experience = "не определён"
    elif analysis and analysis.extraction.experience_min_years is not None:
        minimum = analysis.extraction.experience_min_years
        experience = "Без опыта" if minimum == 0 else f"от {minimum:g} лет (по описанию)"
    elif analysis and analysis.extraction.experience_evidence is not None:
        experience = "требуется, минимум лет не указан в описании"
    if match.comparison is not None:
        scopes = [
            f"{item.name}: от {item.years_required:g} лет"
            for item in match.comparison.items
            if item.years_required is not None
        ]
        experience = (
            "; ".join(dict.fromkeys(scopes)) or "минимум лет не указан; смотрите неопределённость"
        )
        experience = _short(experience, 250)
    lines = [
        "Проверено: "
        + (
            vacancy.last_checked_at.strftime("%d.%m.%Y %H:%M UTC")
            if vacancy.last_checked_at
            else "дата проверки неизвестна"
        ),
        "Опубликовано: " + vacancy.published_at.strftime("%d.%m.%Y"),
        score_label(match),
        _short(vacancy.title, 200),
        _short(vacancy.company, 150),
        _short(_salary(vacancy), 180),
        f"{remote} · {_short(vacancy.location or 'Город не указан', 100)}",
        f"Опыт: {experience}",
        evidence_label(match),
        "Порядок и допуск определены моделью и полнотой сведений. Не вероятность найма."
        if match.ranking and match.ranking.applied
        else "Оценка по правилам, не вероятность найма.",
        "",
        "Совпало: " + _short(", ".join(match.matched_skills) or no_matches, 350),
    ]
    if analysis:
        lines.insert(3, "Категория: " + CATEGORY_LABELS[analysis.extraction.category])
    groups = (
        ("Обязательно, нет в профиле", match.missing_required),
        ("Желательно, нет в профиле", match.missing_preferred),
        ("Важность не указана, нет в профиле", match.missing_unspecified),
    )
    if any(values for _, values in groups):
        lines += [
            label + ": " + _short(", ".join(values), 200) for label, values in groups if values
        ]
    elif not match.alternatives or match.missing_skills:
        lines.append(
            "Не найдено в профиле: " + _short(", ".join(match.missing_skills) or "нет", 350)
        )
    alternatives = [
        g for g in match.alternatives if g.importance not in {"not_required", "context"}
    ]
    for group in alternatives[:2]:
        importance = {
            "required": "обязательно",
            "preferred": "желательно",
            "unspecified": "важность не указана",
        }[group.importance]
        state = "есть: " + ", ".join(group.matched_options) if group.satisfied else "нет в профиле"
        lines.append(_short(f"Один из ({importance}): {' / '.join(group.options)} — {state}", 180))
    if len(alternatives) > 2:
        lines.append(f"Других групп альтернатив: {len(alternatives) - 2}.")
    documents = [q for q in match.qualifications if q.importance not in {"not_required", "context"}]
    for document in documents[:3]:
        importance = {
            "required": "обязательно",
            "preferred": "желательно",
            "unspecified": "важность не указана",
        }[document.importance]
        lines.append(
            _short(
                f"Документ ({importance}): {document.name} — "
                f"{QUALIFICATION_LABELS[document.status]}",
                170,
            )
        )
    if len(documents) > 3:
        lines.append(f"Других документов/допусков для уточнения: {len(documents) - 3}.")
    if documents and not (match.ranking and match.ranking.applied):
        lines.append("Документы не включены в балл; сведения не проверены.")
    elif documents:
        lines.append("Документы участвуют в проверке условий; сведения не проверены.")
    if match.reasons:
        heading = "Факты сравнения:" if match.ranking and match.ranking.applied else "Причины:"
        lines += ["", heading] + ["• " + _short(reason, 180) for reason in match.reasons[:3]]
    if not match.eligible:
        lines.insert(0, "Не соответствует обязательным условиям профиля.\n")
    if match.warnings:
        lines += ["", "Уточнения:"] + [
            "• " + _short(warning, 180) for warning in match.warnings[:4]
        ]
    return "\n".join(lines)


def render_profile(profile: CandidateProfile) -> str:
    from scoring.comparison import candidate_skill_names

    prefs = profile.preferences
    experience = profile.experience_years if profile.experience_years is not None else "не указан"
    documents = []
    for document in profile.qualifications:
        label = document.name + (
            " (указано: нет)" if document.status == "absent" else " (с ваших слов)"
        )
        if document.valid_until:
            label += f" до {document.valid_until:%d.%m.%Y}"
        documents.append(label)
    return "\n".join(
        [
            "Текущий профиль",
            "Навыки: "
            + _short(", ".join(sorted(candidate_skill_names(profile))) or "не указаны", 1500),
            f"Общий опыт: {experience}",
            "Позиции: " + _short(", ".join(prefs.positions) or "не указаны", 700),
            "Формат: " + REMOTE_LABELS[prefs.remote],
            "Города: " + _short(", ".join(prefs.locations) or "любые", 400),
            "Обязательные условия: "
            + (
                ", ".join(
                    name
                    for flag, name in [
                        (prefs.salary_required, "зарплата"),
                        (prefs.locations_required, "города"),
                    ]
                    if flag
                )
                or "зарплата и города — предпочтения"
            ),
            "Документы и допуски: " + _short("; ".join(documents) or "не указаны", 500),
            "Минимум зарплаты: "
            + (
                _short(f"{prefs.salary_min} {prefs.currency}", 100)
                + ", "
                + ("до налогов" if prefs.salary_gross else "на руки")
                if prefs.salary_min is not None
                else "не задан"
            ),
            "Чтобы изменить профиль, выберите поле ниже.",
        ]
    )
