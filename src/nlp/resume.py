"""Conservative explicit-section extraction; no profession whitelist or invented dates."""

import re

from domain.resume import ResumeExtraction
from nlp.extraction import canonical_skill, skill_key
from nlp.text import contains_phrase

SKILLS = re.compile(
    r"^(?:ключевые |профессиональные |технические |technical |key )?"
    r"(?:навыки|компетенции|skills|competencies)(?:\s*[:—-]\s*(.*)|\s*)$",
    re.I,
)
POSITION = re.compile(
    r"^(?:желаемая должность(?: и зарплата)?|желаемая позиция|desired position|"
    r"desired job title|target role)"
    r"\s*(?:[:—-]\s*(.*))?$",
    re.I,
)
STOP = re.compile(
    r"^(?:опыт(?: работы)?|общий опыт|образование|о себе|обо мне|контакты|языки|курсы|"
    r"сертификаты|документы|допуски|портфолио|дополнительная информация|work experience|"
    r"experience|education|about me|contacts?|languages?|certificates?|licen[cs]es?|portfolio)\b",
    re.I,
)
EXPERIENCE = re.compile(
    r"^(?:общий (?:опыт|стаж)(?: работы)?|опыт работы|total (?:work )?experience)\s*[:—-]?\s*"
    r"(?:(?P<years>\d+(?:[.,]\d+)?)\s*(?:лет|год[а]?|years?|yrs?)\s*)?"
    r"(?:(?P<months>\d+)\s*(?:месяц(?:а|ев)?|months?|mos?)\s*)?$",
    re.I,
)
NO_EXPERIENCE = re.compile(
    r"^(?:общий (?:опыт|стаж)(?: работы)?|опыт работы|total (?:work )?experience)\s*[:—-]\s*"
    r"(?:без опыта|нет опыта|no experience)\s*$",
    re.I,
)
EXPERIENCE_HEADING = re.compile(
    r"^(?:общий (?:опыт|стаж)(?: работы)?|опыт работы|total (?:work )?experience)\s*[:—-]?\s*$",
    re.I,
)


def invalid_resume_skill(value: str, positions=()) -> bool:
    value = value.strip(" .;•–—-")
    if EXPERIENCE.fullmatch(value) or NO_EXPERIENCE.fullmatch(value):
        return True
    if re.fullmatch(
        r"студент(?:ка)?|учащийся|учащаяся|student|без опыта(?: работы)?|нет опыта", value, re.I
    ):
        return True
    key = skill_key(value)
    return any(
        key == skill_key(position) or len(key) >= 2 and contains_phrase(skill_key(position), key)
        for position in positions
    )


def looks_like_resume(text: str) -> bool:
    if not 20 <= len(text) <= 60_000:
        return False
    parts = [part.strip(" .;•–—-") for part in re.split(r"(?<=[.!?;])\s+|\n+", text)]
    return any(
        EXPERIENCE.fullmatch(part) or NO_EXPERIENCE.fullmatch(part) or POSITION.fullmatch(part)
        for part in parts
    ) or bool(len(parts) > 1 and any(SKILLS.fullmatch(part) for part in parts))


def _heading(line: str) -> bool:
    skills = SKILLS.fullmatch(line)
    position = POSITION.fullmatch(line)
    return bool(
        skills
        and not skills[1]
        or position
        and not position[1]
        or EXPERIENCE_HEADING.fullmatch(line)
        or STOP.fullmatch(line.rstrip(" :—-"))
    )


def _boundary(line: str) -> bool:
    return bool(
        SKILLS.fullmatch(line)
        or POSITION.fullmatch(line)
        or STOP.match(line)
        or EXPERIENCE_HEADING.fullmatch(line)
        or re.match(r"^[^:]{1,60}:", line)
    )


def _logical_lines(text: str) -> list[str]:
    """Restore explicit fields from word-per-line text without altering stored resume text."""
    raw = [value for line in text.splitlines() if (value := line.strip().lstrip("•●▪*–- "))]
    lines = []
    split_heading = False
    index = 0
    while index < len(raw):
        end = index + 1
        # Only explicit headings are joined; employment prose and dates remain untouched.
        for candidate_end in range(index + 2, min(index + 6, len(raw)) + 1):
            if _heading(" ".join(raw[index:candidate_end])):
                end = candidate_end
        split_heading |= end > index + 1
        lines.append(" ".join(raw[index:end]))
        index = end
    wrapped = split_heading and sum(len(line.split()) == 1 for line in raw) >= len(raw) * 0.75
    result = []
    index = 0
    while index < len(lines):
        line = lines[index]
        position = POSITION.fullmatch(line)
        skills = SKILLS.fullmatch(line)
        if EXPERIENCE.fullmatch(line):
            end = index + 1
            for candidate_end in range(index + 2, min(index + 8, len(lines)) + 1):
                candidate = " ".join(lines[index:candidate_end])
                years = EXPERIENCE.fullmatch(candidate)
                if (
                    years
                    and (years["years"] or years["months"])
                    or NO_EXPERIENCE.fullmatch(candidate)
                ):
                    end = candidate_end
            result.append(" ".join(lines[index:end]))
            index = end
            continue
        if wrapped and (position or skills):
            end = index + 1
            while end < len(lines) and not _boundary(lines[end]):
                end += 1
            field = position or skills
            value = " ".join([field[1] or "", *lines[index + 1 : end]])
            if position:
                result.append("Желаемая должность: " + value.strip())
                index = end
                continue
            if re.search(r"[,;|•]", value):
                # Delimited multi-word skills stay intact, e.g. REST / API on separate lines.
                result.append("Навыки: " + value.strip())
                index = end
                continue
        result.append(line)
        index += 1
    return result


def skill_details(value):
    """Only a directly attached self-report is a skill level/tenure, never job dates."""
    match = re.fullmatch(
        r"(.+?)\s*[:—–]\s*(?:(\d+(?:[.,]\d+)?)\s*(?:лет|года?|years?)\s*)?"
        r"(?:(?:уровень\s*)?(начинающий|средний|продвинутый|beginner|intermediate|advanced))?",
        value,
        re.I,
    )
    if not match or not (match[2] or match[3]):
        return value, None, None
    years = float(match[2].replace(",", ".")) if match[2] else None
    if years is not None and years > 80:
        return match[1], None, None
    level = {"начинающий": "beginner", "средний": "intermediate", "продвинутый": "advanced"}.get(
        (match[3] or "").lower(), (match[3] or "").lower() or None
    )
    return match[1], years, level


def extract_resume(text: str) -> ResumeExtraction:
    skills, positions, experience = [], [], []
    in_skills = False
    next_position = False
    notes = []
    for line in _logical_lines(text):
        line = line.strip().lstrip("•●▪*–- ")
        if not line:
            continue
        skill_heading = SKILLS.fullmatch(line)
        position = POSITION.fullmatch(line)
        years = EXPERIENCE.fullmatch(line)
        if position:
            in_skills = False
            next_position = not bool(position[1])
            positions.extend(v.strip() for v in re.split(r"[;|]", position[1] or "") if v.strip())
        elif NO_EXPERIENCE.fullmatch(line):
            next_position = False
            experience.append(0.0)
            in_skills = False
        elif years and (years["years"] or years["months"]):
            next_position = False
            value = (
                float((years["years"] or "0").replace(",", ".")) + int(years["months"] or 0) / 12
            )
            if value <= 80 and (not years["years"] or int(years["months"] or 0) <= 11):
                experience.append(round(value, 2))
            in_skills = False
        elif skill_heading:
            next_position = False
            in_skills = True
            skills.extend(re.split(r"[,;|•]", skill_heading[1] or ""))
        elif STOP.match(line) or line.endswith(":"):
            in_skills = False
            next_position = False
        elif next_position:
            positions.append(line)
            next_position = False
        elif in_skills:
            skills.extend(re.split(r"[,;|•]", line))
    names = {}
    for value in skills:
        value = value.strip().strip("•●▪*–- ")
        value, _, _ = skill_details(value)
        if invalid_resume_skill(value, positions):
            notes.append(
                "Должность, сведения об опыте и личный статус исключены из списка навыков."
            )
            continue
        if re.search(
            r"@|https?://|www\.|\+?\d[\d() -]{8,}|не (?:знаю|владею|умею)|"
            r"нет опыта|без опыта|no experience",
            value,
            re.I,
        ):
            continue
        if value and len(value) <= 100:
            names.setdefault(skill_key(value), canonical_skill(value))
        elif value:
            notes.append("Длинные фразы из раздела навыков пропущены; проверьте профиль.")
    if len(names) > 50:
        notes.append("Извлечены первые 50 навыков; проверьте список.")
    positions = list(dict.fromkeys(v for v in positions if len(v) <= 120))[:10]
    experience = list(dict.fromkeys(experience))
    if len(experience) > 1:
        notes.append("Найдено несколько разных значений общего опыта; укажите его вручную.")
    if not names:
        notes.append("Явный раздел навыков не найден или пуст; добавьте навыки через /profile.")
    if not positions:
        notes.append("Желаемая должность не определена; задайте её через /profile.")
    if len(experience) != 1:
        notes.append("Общий стаж не определён. Даты отдельных мест работы не суммируются.")
    return ResumeExtraction(
        skills=list(names.values())[:50],
        positions=positions,
        experience_years=experience[0] if len(experience) == 1 else None,
        notes=list(dict.fromkeys(notes)),
    )
