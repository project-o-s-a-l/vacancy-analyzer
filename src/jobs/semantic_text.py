"""Conservative professional inputs: confirmed fields and explicit job sections only."""

import re

from processing.cleaner import clean_html
from scoring.serving_features import digest

TEXT_SPEC = {
    "version": "professional-text-v1",
    "max_characters": 6000,
    "resume_scope": "confirmed_positions_skills_experience_only",
    "job_scope": "title_skills_experience_and_explicit_professional_sections",
    "personal_fields": "never_include_name_company_address_location_contacts",
}
_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[\w-]+", re.I)
_URL = re.compile(r"(?:https?://|www\.|t\.me/|telegram\.me/)\S+", re.I)
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\s().-]*){10,15}(?!\w)")
_PERSONAL = re.compile(
    r"(?:фио|фамилия|имя\s*[:：]|отчество|дата рождения|прожива|"
    r"адрес|улиц[аеы]?\b|телефон|контакт|связаться|почт[аы]|telegram|"
    r"зарплат|оплат|оформление|дмс|отпуск|место работы)",
    re.I,
)
_START = re.compile(
    r"^(?:обязанности|требования(?: к кандидату)?|основные должностные обязанности|"
    r"задачи|ваши задачи|что (?:нужно|будешь|предстоит) делать|чем предстоит заниматься|"
    r"что мы ожидаем|мы ожидаем(?: от тебя)?|вам понадобятся|очень пригодятся|"
    r"основной фокус|duties|responsibilities|requirements|qualifications)\s*[:：]?$",
    re.I,
)
_STOP = re.compile(
    r"^(?:условия.*|что (?:мы )?предлагаем.*|мы предлагаем.*|о (?:нас|компании).*|"
    r"как откликнуться.*|контакты.*|кому подойдет.*|benefits.*|about us.*|"
    r"условия и преимущества.*|присоединившись.*|что получ.*)\s*[:：]?$",
    re.I,
)


def scrub(value, *, literals=()):
    value = _URL.sub(" ", _EMAIL.sub(" ", str(value)))
    value = _PHONE.sub(" ", value)
    for literal in literals:
        if literal and len(literal.strip()) >= 3:
            value = re.sub(re.escape(literal.strip()), " ", value, flags=re.I)
    return " ".join(value.split())


def _field(value, literals=()):
    # A contact-bearing value is not a professional skill or position.
    if _PERSONAL.search(str(value)) or _EMAIL.search(str(value)) or _URL.search(str(value)):
        return ""
    return scrub(value, literals=literals)


def profile_text(profile):
    # Do not scrape free resume text: names, contacts and precise addresses never enter here.
    positions = filter(None, (_field(v) for v in profile.preferences.positions))
    skills = sorted(filter(None, (_field(v) for v in profile.skills)))
    return "\n".join(
        [
            "Position: " + "; ".join(positions),
            "Skills: " + "; ".join(skills),
            "Experience years: "
            + (
                str(profile.experience_years) if profile.experience_years is not None else "unknown"
            ),
        ]
    )[: TEXT_SPEC["max_characters"]]


def vacancy_text(vacancy):
    literals = (vacancy.company, vacancy.address, vacancy.location)
    title = _field(vacancy.title, literals)
    skills = sorted(filter(None, (_field(v, literals) for v in vacancy.skills)))
    # Unknown headings are intentionally not interpreted by a model or copied as biography.
    lines, in_section = [], False
    body = clean_html(vacancy.description or vacancy.description_html)
    for line in body.splitlines():
        line = line.strip()
        if _START.match(line):
            in_section = True
            continue
        if _STOP.match(line):
            in_section = False
            continue
        if in_section and line and not _PERSONAL.search(line):
            lines.append(scrub(line, literals=literals))
    return "\n".join(
        [
            "Position: " + title,
            "Skills: " + "; ".join(skills),
            "Experience: " + _field(vacancy.experience or "unknown"),
            "Tasks and requirements: " + " ".join(lines),
        ]
    )[: TEXT_SPEC["max_characters"]]


def text_revision(text):
    return digest({"spec": TEXT_SPEC, "text": text})
