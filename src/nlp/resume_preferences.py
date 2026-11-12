"""Explicit desired conditions only; residence/employer/pay history is not a preference."""

import re

from domain.candidate import Preferences
from domain.resume import ResumePreference

HEADING = re.compile(
    r"^(?P<heading>обязательн(?:ые|ое) услови(?:я|е)|пожелания|условия работы|"
    r"желаемые условия|requirements|preferences)\s*:\s*(?P<body>.*)$",
    re.I,
)
LOCATION = re.compile(
    r"(?:(?P<only>только)\s+(?:работа\s+в\s+)?|"
    r"(?:желаемые? города?|город работы|локация|location)\s*:\s*)(?P<value>.+)",
    re.I,
)
SALARY = re.compile(
    r"(?:желаемая )?(?:зарплата|salary)\s*:?\s*(?:от|не ниже|from|at least)\s*"
    r"(?P<amount>\d[\d \u00a0]*)\s*(?P<currency>RUB|RUR|руб\.?|₽|USD|EUR)\s*"
    r"(?:в месяц|/\s*месяц|per month|monthly)\s*"
    r"(?P<tax>до налога|до вычета налогов|gross|на руки|после налогов|net)",
    re.I,
)
REMOTE = re.compile(
    r"(?:(?P<required>только|only)\s+)?(?:удал[её]нная работа|remote(?: work)?)", re.I
)


def extract_preferences(text):
    proposals, conflicts, notes = {}, set(), []
    for line in re.finditer(r"[^\n]+", text):
        raw = line[0].strip().lstrip("•●▪*–- ")
        heading = HEADING.fullmatch(raw)
        body = heading["body"] if heading else raw
        required = bool(heading and re.match(r"обязательн|requirements", heading["heading"], re.I))
        base = line.start() + line[0].find(body)
        for piece in re.finditer(r"[^;]+", body):
            clause = piece[0].strip().rstrip(".")
            start = base + piece.start() + piece[0].find(clause)
            end = start + len(clause)
            remote = REMOTE.fullmatch(clause)
            salary = SALARY.fullmatch(clause)
            location = LOCATION.fullmatch(clause)
            group, values = None, None
            if remote:
                group = "remote"
                values = Preferences(
                    remote="required" if required or remote["required"] else "preferred"
                )
            elif salary:
                group = "salary"
                currency = salary["currency"].upper()
                currency = "RUB" if currency in {"RUR", "РУБ", "РУБ.", "₽"} else currency
                values = Preferences(
                    salary_min=int(re.sub(r"\s", "", salary["amount"])),
                    currency=currency,
                    salary_gross=salary["tax"].casefold()
                    in {"до налога", "до вычета налогов", "gross"},
                    salary_required=required,
                )
            elif location and (heading or not location["only"]):
                cities = [
                    x.strip() for x in re.split(r",|\s+или\s+", location["value"]) if x.strip()
                ]
                # A city value must be a short literal; never reinterpret a remote/salary clause.
                if cities and all(
                    re.fullmatch(r"[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё .-]{0,79}", x) for x in cities
                ):
                    group = "locations"
                    values = Preferences(
                        locations=cities, locations_required=required or bool(location["only"])
                    )
            if group:
                item = ResumePreference(group=group, values=values, start=start, end=end)
                old = proposals.get(group)
                if old and old.values != values:
                    conflicts.add(group)
                proposals[group] = item
            elif heading or re.match(
                r"(?:желаемая )?(?:зарплата|salary|город работы|локация)\b", clause, re.I
            ):
                notes.append(
                    "Условие не разобрано однозначно: «" + clause + "». Исправьте в черновике."
                )
    for group in sorted(conflicts):
        proposals.pop(group, None)
        notes.append(f"Противоречивые условия ({group}); задайте значение через /resume_edit.")
    return list(proposals.values()), list(dict.fromkeys(notes))
