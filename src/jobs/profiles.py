"""Shared profile transformations; no Telegram IDs are accepted from user input."""

import math
import re
from datetime import date, datetime

from domain import CandidateProfile, Skill
from nlp.extraction import canonical_skill
from nlp.qualifications import qualification_key, qualification_name
from nlp.resume import invalid_resume_skill, looks_like_resume

FIELDS = ("skills", "experience", "positions", "remote", "salary", "locations")
EDIT_FIELDS = (*FIELDS, "qualifications", "constraints")
_SKIP = {"пропустить", "нет", "-", "?", "не знаю", "очистить"}


def _list(text: str, *, maximum: int, item_length: int) -> list[str]:
    values = [" ".join(value.split()) for value in re.split(r"[,;\n]", text)]
    values = list(dict.fromkeys(value for value in values if value))
    if not values or len(values) > maximum or any(len(value) > item_length for value in values):
        raise ValueError(f"Укажите от 1 до {maximum} значений, каждое до {item_length} символов.")
    return values


def update_field(profile: CandidateProfile, field: str, text: str) -> CandidateProfile:
    text = text.strip()
    if not text or len(text) > 4000:
        raise ValueError("Отправьте текст длиной от 1 до 4000 символов.")
    data = profile.model_dump(mode="json")
    skip = text.casefold() in _SKIP
    if field == "skills":
        if skip:
            data["skills"] = {}
            return CandidateProfile.model_validate(data)
        if looks_like_resume(text):
            raise ValueError(
                "Это похоже на описание резюме. Отправьте его через /resume, "
                "чтобы разделить навыки, должность и опыт."
            )
        names = [canonical_skill(value) for value in _list(text, maximum=50, item_length=100)]
        if any(invalid_resume_skill(name) for name in names):
            raise ValueError(
                "В навыках нужны конкретные умения и технологии. "
                "Опыт укажите отдельно; «студент» не является навыком."
            )
        data["skills"] = {
            name: profile.skills.get(name, Skill()).model_dump(mode="json") for name in names
        }
    elif field == "experience":
        try:
            years = None if skip else float(text.replace(",", "."))
        except ValueError:
            raise ValueError("Укажите число лет от 0 до 80 или «пропустить».") from None
        if years is not None and (not math.isfinite(years) or not 0 <= years <= 80):
            raise ValueError("Укажите число лет от 0 до 80 или «пропустить».")
        data["experience_years"] = years
    elif field in {"positions", "locations"}:
        data["preferences"][field] = [] if skip else _list(text, maximum=10, item_length=120)
        if field == "locations" and skip:
            data["preferences"]["locations_required"] = False
    elif field == "remote":
        values = {
            "любой": "any",
            "any": "any",
            "предпочтительно удалённо": "preferred",
            "preferred": "preferred",
            "только удалённо": "required",
            "required": "required",
            "офис": "office",
            "на месте работы": "office",
            "очно": "office",
            "office": "office",
        }
        value = "any" if skip else values.get(text.casefold())
        if value is None:
            raise ValueError(
                "Укажите формат: любой, предпочтительно удалённо, "
                "только удалённо или на месте работы."
            )
        data["preferences"]["remote"] = value
    elif field == "salary":
        if skip:
            data["preferences"]["salary_min"] = None
            data["preferences"]["salary_required"] = False
        else:
            match = re.fullmatch(
                r"([\d ]+)\s+([a-zA-Z]{3})\s+(до налогов|на руки|gross|net)", text, re.IGNORECASE
            )
            if not match:
                raise ValueError("Укажите сумму, валюту и налоги: например, 150000 RUB на руки.")
            amount = int(match[1].replace(" ", ""))
            if not 0 <= amount <= 1_000_000_000:
                raise ValueError("Укажите зарплату от 0 до 1 000 000 000.")
            data["preferences"].update(
                salary_min=amount,
                currency=match[2].upper(),
                salary_gross=match[3].casefold() in {"до налогов", "gross"},
            )
    elif field == "constraints":
        modes = {
            "none": (False, False),
            "salary": (True, False),
            "locations": (False, True),
            "both": (True, True),
        }
        mode = "none" if skip else text.casefold()
        if mode not in modes:
            raise ValueError(
                "Выберите обязательные условия кнопкой: зарплата, города, оба или ни одного."
            )
        salary_required, locations_required = modes[mode]
        if salary_required and data["preferences"]["salary_min"] is None:
            raise ValueError("Сначала укажите минимум зарплаты через /profile.")
        if locations_required and not data["preferences"]["locations"]:
            raise ValueError("Сначала укажите города через /profile.")
        data["preferences"].update(
            salary_required=salary_required, locations_required=locations_required
        )
    elif field == "qualifications":
        entries = []
        if not skip:
            values = [v.strip() for v in re.split(r"[;\n]", text) if v.strip()]
            if not 1 <= len(values) <= 20:
                raise ValueError("Укажите до 20 документов через точку с запятой.")
            known = set()
            for value in values:
                status = "present"
                if value.casefold().startswith("нет:"):
                    value, status = value[4:].strip(), "absent"
                expires = None
                ending = re.search(r"\s+до\s+(\S+)\s*$", value, flags=re.I)
                quantity = ending and re.fullmatch(r"\d+(?:[.,]\d+)?[a-zа-яё%/]+", ending[1], re.I)
                if ending and not quantity:
                    try:
                        raw_date = ending[1]
                        expires = (
                            datetime.strptime(raw_date, "%d.%m.%Y").date()
                            if "." in raw_date
                            else date.fromisoformat(raw_date)
                        )
                    except ValueError:
                        raise ValueError("Срок укажите как 31.12.2027 или 2027-12-31.") from None
                    value = value[: ending.start()].strip()
                name = qualification_name(value)
                if status == "absent" and expires is not None:
                    raise ValueError("Для отсутствующего документа не указывайте срок действия.")
                if not name:
                    raise ValueError("Укажите название документа после «нет:».")
                key = qualification_key(name)
                if key in known:
                    raise ValueError("Один документ указан несколько раз. Оставьте одну запись.")
                known.add(key)
                entries.append({"name": name, "status": status, "valid_until": expires})
        data["qualifications"] = entries
    else:
        raise ValueError("Неизвестное поле профиля")
    return CandidateProfile.model_validate(data)
