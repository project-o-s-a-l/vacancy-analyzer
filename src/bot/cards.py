import re
from datetime import UTC, datetime

from aiogram.utils.formatting import Bold, Text, as_list

from domain import CandidateProfile, Vacancy
from domain.matching import MatchResult
from domain.vacancy import SOURCE_NAMES
from domain.vacancy_analysis import CATEGORY_LABELS
from jobs.catalog_freshness import RECHECK_AFTER
from nlp.analysis_identity import current_vacancy_analysis as current_analysis
from nlp.extraction import canonical_skill
from nlp.profile_experience import work_calendar
from scoring.presentation import evidence_label, score_label

EXPERIENCE_LABELS = {
    "no_experience": "без опыта",
    "1-3": "1–3 года",
    "3-6": "3–6 лет",
    "6+": "более 6 лет",
}
PERIOD_LABELS = {"MONTH": "в месяц", "HOUR": "в час", "YEAR": "в год", "SHIFT": "за смену"}
REMOTE_LABELS = {
    "any": "любой",
    "preferred": "предпочтительно удалённо",
    "required": "только удалённо",
    "office": "на месте работы",
}
QUALIFICATION_LABELS = {
    "declared_present": "есть с ваших слов",
    "declared_absent": "вы указали, что его нет",
    "expired": "срок действия истёк",
    "unknown": "не указан в профиле",
    "uncertain": "стоит уточнить",
    "not_required": "не требуется",
    "context": "упомянут в описании",
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
    value = " ".join(value.split())
    encoded = value.encode("utf-16-le")
    if len(encoded) <= limit * 2:
        return value
    return encoded[: (limit - 1) * 2].decode("utf-16-le", errors="ignore").rstrip() + "…"


def _amount_range(lower, upper) -> str:
    if lower == upper:
        amount = f"{lower:,.0f}"
    elif lower is None:
        amount = f"до {upper:,.0f}"
    elif upper is None:
        amount = f"от {lower:,.0f}"
    else:
        amount = f"{lower:,.0f}–{upper:,.0f}"
    return amount.replace(",", " ")


def _salary(vacancy: Vacancy) -> str:
    if vacancy.salary_from is None and vacancy.salary_to is None:
        return _short(vacancy.salary_raw or "Зарплата не указана", 150)
    amount = _amount_range(vacancy.salary_from, vacancy.salary_to)
    currency = {"RUB": "₽", "RUR": "₽"}.get(vacancy.currency, vacancy.currency)
    period = PERIOD_LABELS.get(vacancy.salary_period, "· период не указан")
    tax = {True: "до налогов", False: "на руки", None: "налоги не указаны"}[vacancy.salary_gross]
    return f"{amount} {currency or '(валюта не указана)'} {period} · {tax}"


def _minimum_years(years: float) -> str:
    if years == 0:
        return "без опыта"
    unit = "года" if years % 1 or (years % 10 == 1 and years % 100 != 11) else "лет"
    return f"от {years:g} {unit}".replace(".", ",")


def _experience(vacancy: Vacancy, match: MatchResult) -> str | None:
    # The aggregate eligibility minimum can belong to a single skill. Only role or
    # total tenure belongs on the general experience line.
    if match.comparison is not None:
        roles = [
            item
            for item in match.comparison.items
            if item.kind in {"role", "experience_total"} and item.id.startswith("req:experience:")
        ]
        if any(item.state == "conflict" and item.years_required is None for item in roles):
            return "требования к стажу стоит уточнить"
        for item in roles:
            if item.importance == "required":
                return (
                    _minimum_years(item.years_required)
                    if item.years_required is not None
                    else "требуется, минимум лет не указан"
                )
    elif match.eligibility:
        requirement = match.eligibility.experience
        if requirement.conflict:
            return "требования к стажу стоит уточнить"
        if requirement.minimum_years is not None:
            return _minimum_years(requirement.minimum_years)
        if requirement.required:
            return "требуется, минимум лет не указан"
    return EXPERIENCE_LABELS.get(vacancy.experience)


def _source_spelling(name, evidence):
    pattern = re.compile(re.escape(name), re.IGNORECASE)
    for item in evidence:
        original = pattern.search(item.text)
        if original:
            return original[0]
    return name


def _match_summary(vacancy: Vacancy, match: MatchResult) -> list[str]:
    labels = {canonical_skill(name): name for name in vacancy.skills}
    for requirement in match.requirements:
        if requirement.skill not in labels:
            labels[requirement.skill] = _source_spelling(requirement.skill, requirement.evidence)

    def skill_list(values):
        names = [labels.get(name, name) for name in values[:5]]
        more = f" и ещё {len(values) - 5}" if len(values) > 5 else ""
        return _short(", ".join(names), 150) + more

    group = next(
        (g for g in match.alternatives if g.importance not in {"not_required", "context"}), None
    )
    group_matches = group.matched_options if group and group.satisfied else []
    standalone = [name for name in match.matched_skills if name not in group_matches]
    lines = []
    if standalone:
        lines.append("Навыки из вашего профиля: " + skill_list(standalone))
    if match.missing_required:
        lines.append("Нужны, но не указаны в профиле: " + skill_list(match.missing_required))

    if group:
        options = " или ".join(labels.get(name, name) for name in group.options)
        prefix = {
            "required": "Нужен один из навыков",
            "preferred": "Будет плюсом один из навыков",
            "unspecified": "Упомянут один из навыков",
        }[group.importance]
        if group.satisfied and group.matched_options:
            result = "в профиле есть " + skill_list(group.matched_options)
        elif group.satisfied:
            result = "условие выполнено"
        else:
            result = "стоит уточнить"
        lines.append(f"{prefix}: {_short(options, 100)} — {result}.")

    if match.comparison:
        for item in match.comparison.items:
            if (
                item.kind == "skill"
                and item.importance == "required"
                and item.years_required
                and item.state != "match"
            ):
                name = _short(labels.get(item.name, item.name), 80)
                lines.append(f"Требуемый опыт с {name}: {_minimum_years(item.years_required)}.")
                break

    for doc in match.qualifications:
        if doc.importance in {"not_required", "context"}:
            continue
        label = {
            "required": "Нужен документ",
            "preferred": "Желательный документ",
            "unspecified": "В вакансии упомянут документ",
        }[doc.importance]
        name = _short(_source_spelling(doc.name, doc.evidence), 100)
        lines.append(f"{label}: {name} — {QUALIFICATION_LABELS[doc.status]}.")
        break

    if match.eligibility:
        issues = [*match.eligibility.exclusions, *match.eligibility.uncertainties]
        for issue in issues:
            if issue.field not in {"salary", "remote", "location"}:
                continue
            if (
                issue.code == "salary_incomparable"
                and vacancy.salary_from is None
                and vacancy.salary_to is None
            ):
                continue  # Salary is already marked as undisclosed above.
            labels_by_code = {
                "salary_incomparable": "Уточните, как эта сумма соотносится с вашим минимумом.",
                "salary_not_guaranteed": "Уточните, предложат ли вам желаемую зарплату.",
                "remote_preferred": "Вы предпочитаете удалённую работу. Уточните формат.",
            }
            lines.append(labels_by_code.get(issue.code, _short(issue.message, 150) + "."))
            break
    if match.missing_preferred:
        lines.append("Будут плюсом: " + skill_list(match.missing_preferred))
    unclassified = match.missing_unspecified
    if not (match.missing_required or match.missing_preferred or unclassified):
        unclassified = match.missing_skills
    if unclassified:
        lines.append("Также упомянуты в вакансии: " + skill_list(unclassified))
    if not lines:
        lines.append(
            "В вакансии мало сведений для сравнения навыков."
            if match.components.get("skills") is None
            else "Совпадений по навыкам пока нет."
        )
    return lines[:4]


def format_card(vacancy: Vacancy, match: MatchResult) -> Text:
    """Native Telegram entities keep employer-supplied text literal."""
    heading = as_list(Bold(_short(vacancy.title, 180)), _short(vacancy.company, 120))
    conditions = [_short(_salary(vacancy), 180)]
    if converted := match.salary_conversion:
        amount = _amount_range(converted.lower, converted.upper)
        conditions.append(
            f"≈ {amount} {converted.currency} по курсу ЦБ на {converted.as_of:%d.%m.%Y}"
        )
    work_format = vacancy.work_format or {True: "Удалённо", False: "Без полной удалёнки"}.get(
        vacancy.remote
    )
    place = " · ".join(_short(s, 100) for s in (vacancy.location, work_format) if s)
    if place:
        conditions.append(place)
    experience = _experience(vacancy, match)
    if experience:
        conditions.append("Опыт: " + experience)

    sections = [heading]
    exclusions = match.eligibility.exclusions if match.eligibility else []
    if vacancy.archived or any(i.code in {"archived", "expired"} for i in exclusions):
        sections.append("Вакансия в архиве или срок её действия истёк.")
    elif not match.eligible:
        sections.append("Не соответствует обязательным условиям вашего профиля.")
    sections.append(as_list(*conditions))
    analysis = current_analysis(vacancy)
    if analysis and analysis.extraction.responsibilities:
        sections.append("Задачи: " + _short(analysis.extraction.responsibilities[0], 160))
    sections.append(as_list(*_match_summary(vacancy, match)))
    footer = f"Опубликована {vacancy.published_at:%d.%m.%Y}"
    if analysis and analysis.extraction.category != "other":
        footer += " · " + CATEGORY_LABELS[analysis.extraction.category]
    if not vacancy.archived:
        source = SOURCE_NAMES.get(vacancy.source, vacancy.source)
        if vacancy.last_checked_at is None:
            footer += f"\nДоступность вакансии на {source} ещё не проверена."
        elif vacancy.last_checked_at <= datetime.now(UTC) - RECHECK_AFTER:
            footer += f"\nПоследняя проверка на {source} — {vacancy.last_checked_at:%d.%m.%Y}."
    return as_list(*sections, footer, sep="\n\n")


def render_card(vacancy: Vacancy, match: MatchResult) -> str:
    return format_card(vacancy, match).render()[0]


def render_match_details(match: MatchResult) -> str:
    lines = ["Как подобрана вакансия", score_label(match), evidence_label(match)]
    if match.ranking and match.ranking.applied:
        lines.append("Модель учла совпадения и полноту сведений в момент подбора.")
    else:
        lines.append("Баллы показывают, насколько вакансия совпала с профилем на момент подбора.")
    for heading, values in (("Что совпало", match.reasons), ("Что уточнить", match.warnings)):
        if values:
            lines += ["", heading, *("• " + _short(value, 220) for value in values[:6])]
    return "\n".join(lines)


def experience_label(profile: CandidateProfile) -> str | None:
    from scoring.comparison import total_experience

    claim = total_experience(profile)
    if claim.status == "conflict":
        values = " и ".join(
            (f"{v:g}" if isinstance(v, (float, int)) else str(v)).replace(".", ",")
            for v in claim.alternatives
        )
        return f"требует уточнения ({values} г.)"
    value = claim.value
    if value is None:
        return None
    years = f"{value:g}".replace(".", ",")
    unit = "лет"
    if value % 1 or (value % 10 in {2, 3, 4} and not 12 <= value % 100 <= 14):
        unit = "года"
    elif value % 10 == 1 and value % 100 != 11:
        unit = "год"
    facts = profile.professional_facts
    by_dates = facts and any(
        f.id == "experience_total" and f.name == "calendar_union_years" for f in facts.facts
    )
    source = "по датам работы" if by_dates else "указан в профиле"
    return f"{years} {unit} ({source})"


def calendar_notes(profile):
    facts = profile.professional_facts
    if not facts or not facts.episodes:
        return []
    years, complete, _ = work_calendar(facts)
    lines = []
    if years is not None and years != profile.experience_years:
        number = f"{years:g}".replace(".", ",")
        lines.append(f"По заполненным периодам работы — {number} г.")
    if not complete:
        lines.append("Для полного расчёта уточните вид опыта и даты записей.")
    return lines


def render_profile(profile: CandidateProfile, *, title: str = "Мой профиль") -> str:
    from scoring.comparison import candidate_skill_names

    prefs = profile.preferences
    skills = sorted(candidate_skill_names(profile))
    lines = [title, ""]
    if prefs.positions:
        lines.append("Должности для поиска: " + _short(", ".join(prefs.positions), 450))
    lines.append("Навыки: " + _short(", ".join(skills) or "пока не указаны", 800))
    if experience := experience_label(profile):
        lines.append(f"Общий стаж: {experience}")
    lines.extend(calendar_notes(profile))
    if profile.professional_facts and profile.professional_facts.episodes:
        lines.append(f"Мест работы и проектов: {len(profile.professional_facts.episodes)}")
    conditions = []
    if prefs.locations:
        conditions.append("Города: " + _short(", ".join(prefs.locations), 300))
    if prefs.remote != "any":
        conditions.append("Формат: " + REMOTE_LABELS[prefs.remote])
    if prefs.salary_min is not None:
        tax = "до налогов" if prefs.salary_gross else "на руки"
        amount = f"{prefs.salary_min:,}".replace(",", " ")
        conditions.append(_short(f"От {amount} {prefs.currency} в месяц · {tax}", 150))
    required = [
        name
        for flag, name in (
            (prefs.salary_required, "зарплата"),
            (prefs.locations_required, "города"),
        )
        if flag
    ]
    if required:
        conditions.append("Обязательные условия: " + ", ".join(required))
    if conditions:
        lines += ["", *conditions]
    if profile.qualifications:
        documents = []
        for doc in profile.qualifications:
            label = "указано: нет" if doc.status == "absent" else "с ваших слов"
            expiry = f" до {doc.valid_until:%d.%m.%Y}" if doc.valid_until else ""
            documents.append(f"{doc.name}{expiry} ({label})")
        lines += ["", "Документы и допуски: " + _short(", ".join(documents), 450)]
    return "\n".join(lines)
