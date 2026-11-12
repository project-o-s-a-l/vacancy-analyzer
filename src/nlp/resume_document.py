"""Source-preserving resume sections, conservative rules and calendar interval union.

No inference about an occupation from an employer, or skill tenure from employment.
All offsets address the normalized, stored document (Unicode code points).
"""

import calendar
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from domain.resume import ResumeCoverage, ResumeObservation, ResumePeriod
from nlp.extraction import canonical_skill, skill_key
from nlp.resume import EXPERIENCE, NO_EXPERIENCE, POSITION, SKILLS, extract_resume, skill_details

MONTHS = (
    ("январ", "january", "jan"),
    ("феврал", "february", "feb"),
    ("март", "march", "mar"),
    ("апрел", "april", "apr"),
    ("май", "мая", "may"),
    ("июн", "june", "jun"),
    ("июл", "july", "jul"),
    ("август", "august", "aug"),
    ("сентябр", "september", "sep"),
    ("октябр", "october", "oct"),
    ("ноябр", "november", "nov"),
    ("декабр", "december", "dec"),
)
MONTH_WORD = "(?:" + "|".join(v + r"[а-яa-z]*" for m in MONTHS for v in m) + ")"
DATE = (
    rf"(?:\d{{1,2}}\.\d{{1,2}}\.\d{{4}}|\d{{4}}-\d{{2}}-\d{{2}}|"
    rf"\d{{1,2}}[./]\d{{4}}|\d{{4}}-\d{{2}}|{MONTH_WORD}\s+\d{{4}}|\d{{4}})"
)
PRESENT = r"(?:настоящее время|по настоящее время|н\.?\s*в\.?|present|current|now|сейчас)"
PERIOD = re.compile(
    rf"(?P<start>{DATE})\s*(?:[–—−-]|\bпо\b|\bto\b)\s*(?P<end>{PRESENT}|{DATE})(?!\d)", re.I
)
SECTION = re.compile(
    r"^(?:опыт(?: работы)?|employment|work experience|experience|образование|education|"
    r"учебные проекты?|academic projects?|личные проекты?|personal projects?|проекты?|projects?|"
    r"навыки|skills|квалификации|qualifications|сертификаты|certificates|о себе|about me|"
    r"контакты|contacts)\s*[:—-]?\s*$",
    re.I,
)
LABEL = re.compile(
    r"^(?P<label>должность|позиция|роль|role|position|задачи|обязанности|достижения|"
    r"tasks|responsibilities|achievements|выполнял задачи|описание проекта|"
    r"проект|учебный проект|личный проект|project|"
    r"сертификат|квалификация|допуск|certificate|qualification)\s*[:—-]\s*(?P<value>.+)$",
    re.I,
)
NEGATED = re.compile(
    r"\b(?:не (?:знаю|владею|использовал|умею)|нет опыта|без опыта|no experience|not proficient)\b",
    re.I,
)


@dataclass(frozen=True)
class Section:
    start: int
    end: int
    context_start: int


def sections(text: str, *, max_bytes=2400, overlap_bytes=240) -> list[Section]:
    """Prefer heading/episode boundaries, then lines/sentences; bounded UTF-8 context.

    Core ranges partition the entire input. Overlap never crosses a semantic boundary.
    A pathological single line is split without dropping a character.
    """
    boundaries = {0, len(text)}
    for match in re.finditer(r"[^\n]+", text):
        if SECTION.fullmatch(match[0].strip()) or PERIOD.search(match[0]):
            boundaries.add(match.start())
    result = []
    points = sorted(boundaries)
    for lo, hi in zip(points, points[1:], strict=False):
        start = lo
        while start < hi:
            end = start
            size = 0

            def weight(char):
                # JSON escaping and per-fragment labels must also fit the request.
                return len(char.encode()) + (16 if char in "\n.!?;" else 1 if char in '\\"' else 0)

            while end < hi and size + weight(text[end]) <= max_bytes:
                size += weight(text[end])
                end += 1
            if end < hi:
                preferred = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
                if preferred > start + (end - start) // 2:
                    end = preferred + 1
            context = start
            size = 0
            while context > lo and size + weight(text[context - 1]) <= overlap_bytes:
                context -= 1
                size += weight(text[context])
            result.append(Section(start, end, context))
            start = end
    return result


def parse_date(value: str) -> tuple[date | None, str]:
    value = value.strip().casefold()
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            return date.fromisoformat(value), "day"
        if re.fullmatch(r"\d{1,2}\.\d{1,2}\.\d{4}", value):
            day, month, year = map(int, value.split("."))
            return date(year, month, day), "day"
        if re.fullmatch(r"\d{4}-\d{2}", value):
            year, month = map(int, value.split("-"))
            return date(year, month, 1), "month"
        if re.fullmatch(r"\d{1,2}[./]\d{4}", value):
            month, year = map(int, re.split(r"[./]", value))
            return date(year, month, 1), "month"
        for index, variants in enumerate(MONTHS, 1):
            if any(value.startswith(v) for v in variants):
                return date(int(value.split()[-1]), index, 1), "month"
    except ValueError:
        pass
    # Year-only dates are not silently expanded to twelve months.
    return None, "unknown"


def interval(period, reference_time):
    start, start_precision = parse_date(period.date_start)
    end, end_precision = parse_date(period.date_end)
    ongoing = bool(re.fullmatch(PRESENT, period.date_end, re.I))
    today = reference_time.date()
    if not start or (not end and not ongoing) or start > today:
        return None
    if ongoing:
        end = today
    elif end_precision == "month":
        end = end.replace(day=calendar.monthrange(end.year, end.month)[1])
    if end > today or end < start:
        return None
    return start, end + timedelta(days=1), start_precision, end_precision, ongoing


def union_years(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return round(sum((end - start).days for start, end in merged) / 365.2425, 2)


def resolve_experience(result):
    result.notes = [
        n
        for n in result.notes
        if not n.startswith(
            (
                "Общий стаж и даты расходятся",
                "По подтверждённым периодам работы:",
                "Есть неполные/неверные даты",
            )
        )
    ]
    valid = []
    unresolved = False
    for period in result.periods:
        if period.kind == "project":
            continue
        value = interval(period, result.reference_time)
        if period.kind != "employment" or value is None:
            unresolved = True
        else:
            valid.append(value[:2])
    result.calculated_experience_years = union_years(valid) if valid else None
    stated = sorted(set(result.stated_experience_years))
    calculated = result.calculated_experience_years
    conflict = len(stated) > 1 or (
        bool(stated) and calculated is not None and abs(stated[0] - calculated) > 0.1
    )
    if conflict:
        result.experience_years = None
        result.notes.append("Общий стаж и даты расходятся; значения сохранены для проверки.")
    elif stated:
        result.experience_years = stated[0]
    elif valid and not unresolved and calculated <= 80:
        result.experience_years = calculated
    else:
        result.experience_years = None
    if unresolved or calculated is not None and calculated > 80:
        result.notes.append(
            "Есть неполные/неверные даты или неясный вид опыта; полный стаж неизвестен."
        )
    if calculated is not None:
        result.notes.append(
            f"По подтверждённым периодам работы: {calculated:g} г. на "
            f"{result.reference_time.date().isoformat()}; пересечения учтены один раз."
        )
    result.notes = list(dict.fromkeys(result.notes))


def extract_document(text, *, reference_time=None):
    from nlp.resume_preferences import HEADING, extract_preferences

    result = extract_resume(text)
    result.preferences, preference_notes = extract_preferences(text)
    result.notes.extend(preference_notes)
    result.reference_time = reference_time or datetime.now(UTC)
    result.stated_experience_years = []
    context = "unspecified"
    in_skills = False
    next_position = False
    task_section = False
    known_ranges = [(p.start, p.end) for p in result.preferences]
    last_period = None
    for match in re.finditer(r"[^\n]+", text):
        line = match[0].strip().lstrip("•●▪*–- ")
        start = match.start() + match[0].find(line)
        end = start + len(line)
        if not line:
            continue
        if HEADING.fullmatch(line) or any(
            start <= p.start and p.end <= end for p in result.preferences
        ):
            in_skills = next_position = task_section = False
            continue
        section = SECTION.fullmatch(line)
        if section:
            if last_period:
                last_period.end = start
                last_period = None
            context = (
                "work"
                if re.match(r"опыт|employment|work experience|experience", line, re.I)
                else "academic"
                if re.match(r"учеб|academic|образование|education", line, re.I)
                else "personal"
                if re.match(r"личн|personal", line, re.I)
                else "unspecified"
            )
            known_ranges.append((start, end))
        period = PERIOD.search(line)
        incomplete = re.match(rf"(?P<start>{DATE})\s*[–—-]\s*(?:\?|\.\.\.)?\s*$", line, re.I)
        if period:
            if last_period:
                last_period.end = start
            last_period = ResumePeriod(
                start=start,
                end=len(text),
                date_start=period["start"],
                date_end=period["end"],
                kind="employment"
                if context == "work"
                else "project"
                if context in {"academic", "personal"}
                else "unspecified",
                context=context,
            )
            result.periods.append(last_period)
            if period[0] == line:
                known_ranges.append((start, end))
        elif incomplete:
            if last_period:
                last_period.end = start
            last_period = ResumePeriod(
                start=start,
                end=len(text),
                date_start=incomplete["start"],
                date_end="",
                kind="employment" if context == "work" else "unspecified",
                context=context,
            )
            result.periods.append(last_period)
        total = (
            extract_resume(line).experience_years
            if EXPERIENCE.fullmatch(line) or NO_EXPERIENCE.fullmatch(line)
            else None
        )
        if total is not None:
            result.stated_experience_years.append(total)
            if re.match(r"опыт работы|total work experience", line, re.I):
                context = "work"
        skill_heading = SKILLS.fullmatch(line)
        position = POSITION.fullmatch(line)
        label = LABEL.fullmatch(line)
        if section or position or label or total is not None or period:
            in_skills = False
            task_section = False
        if skill_heading:
            in_skills = True
        values = []
        if skill_heading or in_skills:
            content = skill_heading[1] or "" if skill_heading else line
            values = [
                ("skill", s.strip(), "unspecified")
                for s in re.split(r"[,;|•]", content)
                if s.strip()
            ]
        elif position:
            next_position = not bool(position[1])
            values = [
                ("role", s.strip(), "desired")
                for s in re.split(r"[;|]", position[1] or "")
                if s.strip()
            ]
        elif label:
            name = label["label"].casefold()
            kind = (
                "role"
                if name in {"должность", "позиция", "роль", "role", "position"}
                else "qualification"
                if name in {"сертификат", "квалификация", "допуск", "certificate", "qualification"}
                else "task"
            )
            values = [
                (
                    kind,
                    label["value"],
                    "held" if kind == "role" and context == "work" else "unspecified",
                )
            ]
            if last_period is None and (
                kind == "role" and context == "work" or "проект" in name or name == "project"
            ):
                project = "проект" in name or name == "project"
                last_period = ResumePeriod(
                    start=start,
                    end=len(text),
                    date_start="",
                    date_end="",
                    kind="project" if project else "employment",
                    context="academic"
                    if "учеб" in name
                    else "personal"
                    if "личн" in name
                    else context,
                )
                result.periods.append(last_period)
        elif next_position and not section:
            values = [("role", line, "desired")]
            next_position = False
        elif task_section:
            values = [("task", line, "unspecified")]
        if re.fullmatch(r"(?:задачи|обязанности|tasks|responsibilities)\s*:", line, re.I):
            task_section = True
            in_skills = False
            known_ranges.append((start, end))
        accepted = 0
        for kind, name, role_context in values:
            original_name = name
            years, level = None, None
            if kind == "skill":
                name, years, level = skill_details(name)
            if NEGATED.search(line) or len(name) > 1000:
                continue
            if kind == "skill" and canonical_skill(name) not in result.skills:
                continue
            offset = text.find(name, start, end)
            if offset >= 0:
                result.observations.append(
                    ResumeObservation(
                        kind=kind,
                        name=canonical_skill(name) if kind == "skill" else name,
                        start=offset,
                        end=offset + len(name),
                        context=role_context,
                        years=years,
                        level=level,
                        detail_start=offset if years is not None or level else None,
                        detail_end=offset + len(original_name)
                        if years is not None or level
                        else None,
                    )
                )
                accepted += 1
        # A literal task is useful already, but its embedded skills may still need LLM.
        if (
            (values and accepted == len(values) and all(k != "task" for k, _, _ in values))
            or total is not None
            or (skill_heading and not values)
            or position
        ):
            known_ranges.append((start, end))
    if not result.stated_experience_years and result.experience_years is not None:
        result.stated_experience_years = [result.experience_years]
    for block in sections(text):
        unresolved = False
        for line in re.finditer(r"\S[^\n]*", text[block.start : block.end]):
            a, b = block.start + line.start(), block.start + line.end()
            if not any(lo <= a and b <= hi for lo, hi in known_ranges):
                unresolved = True
        result.coverage.append(
            ResumeCoverage(
                start=block.start,
                end=block.end,
                status="unresolved" if unresolved else "rules",
                reason="needs_semantic_review" if unresolved else None,
            )
        )
    result.completeness = (
        "partial" if any(c.status == "unresolved" for c in result.coverage) else "complete"
    )
    resolve_experience(result)
    if result.experience_years is not None:
        result.notes = [n for n in result.notes if not n.startswith("Общий стаж не определён")]
    return result


def merge_extraction(base, inferred):
    """Keep valid observations from every block; overlap only adds evidence, not tenure."""
    skills = list(dict.fromkeys([*base.skills, *inferred.skills]))
    positions = list(dict.fromkeys([*base.positions, *inferred.positions]))
    if len(skills) > 50 or len(positions) > 10:
        base.notes.append(
            "В кратком профиле показаны первые 50 навыков и 10 желаемых ролей; "
            "остальные свидетельства доступны в подробностях."
        )
    base.skills, base.positions = skills[:50], positions[:10]
    observations = {
        (o.kind, skill_key(o.name), o.start, o.end, o.context): o for o in base.observations
    }
    for observation in inferred.observations:
        key = (
            observation.kind,
            skill_key(observation.name),
            observation.start,
            observation.end,
            observation.context,
        )
        observations.setdefault(key, observation)
    base.observations = list(observations.values())
    base.notes = list(dict.fromkeys([*base.notes, *inferred.notes]))
    if inferred.experience_years is not None:
        base.stated_experience_years.append(inferred.experience_years)
    resolve_experience(base)
    return base
