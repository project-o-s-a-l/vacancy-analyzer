"""Alternative spellings of occupations and specializations for lexical retrieval."""

from dataclasses import dataclass
from functools import lru_cache
from itertools import islice, product

from nlp.text import word_tokens

# Each phrase is interchangeable only as a unit. Keep different specializations
# in separate groups (e.g. product/project management and manual/automated QA).
TITLE_GROUPS = (
    ("data analyst", "аналитик данных", "дата аналитик"),
    ("systems analyst", "системный аналитик", "system analyst"),
    ("business analyst", "бизнес аналитик"),
    ("product analyst", "продуктовый аналитик", "аналитик продукта"),
    ("bi analyst", "bi аналитик", "аналитик bi"),
    ("data engineer", "инженер данных", "дата инженер"),
    ("data scientist", "исследователь данных", "дата сайентист"),
    ("ml engineer", "ml инженер", "machine learning engineer", "инженер машинного обучения"),
    ("devops engineer", "devops инженер", "девопс инженер"),
    ("mlops engineer", "mlops инженер"),
    (
        "qa engineer",
        "тестировщик",
        "qa инженер",
        "инженер по тестированию",
        "инженер тестировщик",
        "тестировщик ПО",
        "software tester",
        "qa специалист",
    ),
    (
        "qa automation engineer",
        "тестировщик автоматизатор",
        "automation qa engineer",
        "aqa",
        "инженер по автоматизации тестирования",
        "инженер по автоматизированному тестированию",
    ),
    (
        "manual qa engineer",
        "ручной тестировщик",
        "manual tester",
        "инженер по ручному тестированию",
    ),
    (
        "system administrator",
        "системный администратор",
        "systems administrator",
        "сисадмин",
        "sysadmin",
    ),
    (
        "support engineer",
        "инженер поддержки",
        "technical support engineer",
        "инженер технической поддержки",
    ),
    ("software architect", "архитектор программного обеспечения", "архитектор ПО"),
    ("solution architect", "архитектор решений", "solutions architect"),
    (
        "backend developer",
        "бэкенд разработчик",
        "backend engineer",
        "бекенд разработчик",
        "бэкендер",
        "back end developer",
        "back end engineer",
        "бэкэнд разработчик",
    ),
    (
        "frontend developer",
        "фронтенд разработчик",
        "frontend engineer",
        "фронтендер",
        "front end developer",
        "front end engineer",
        "фронтэнд разработчик",
    ),
    (
        "full stack developer",
        "фулстек разработчик",
        "fullstack developer",
        "fullstack software engineer",
        "full stack software engineer",
        "фуллстек разработчик",
        "фулстэк разработчик",
    ),
    ("backend", "бэкенд", "back end", "бекенд", "бэкэнд"),
    ("frontend", "фронтенд", "front end", "фронтэнд"),
    ("fullstack", "фулстек", "full stack", "фуллстек", "фулстэк"),
    (
        "software engineer",
        "инженер программист",
        "software developer",
        "разработчик ПО",
        "разработчик программного обеспечения",
        "developer",
        "разработчик",
        "программист",
    ),
    ("product manager", "менеджер продукта", "продуктовый менеджер", "продакт менеджер"),
    ("project manager", "руководитель проектов", "менеджер проектов", "проджект менеджер"),
    ("team lead", "руководитель команды", "тимлид", "teamlead"),
    ("tech lead", "технический лидер", "техлид"),
    ("analyst", "аналитик"),
    ("engineer", "инженер"),
    ("manager", "менеджер"),
    ("administrator", "администратор"),
    ("architect", "архитектор"),
    ("designer", "дизайнер"),
    ("recruiter", "рекрутер"),
    ("teacher", "преподаватель", "учитель"),
    ("accountant", "бухгалтер"),
    ("junior", "джуниор", "младший"),
    ("senior", "сеньор", "старший"),
    ("middle", "мидл", "миддл"),
)

# The literal query always survives. Bound the SQL payload for unusually long
# titles combining multiple independent aliases, such as role, level and stack.
MAX_TITLE_VARIANTS = 32


@dataclass(frozen=True)
class TitleVariant:
    terms: frozenset[str]
    # Every word of a substituted phrase must be present before it can earn title credit.
    required: frozenset[str] = frozenset()


@lru_cache(maxsize=1)
def _phrase_index(tokenize):
    def key(label):
        return len(word_tokens(label)), frozenset(tokenize(label))

    index = {}
    for group in TITLE_GROUPS:
        keys = [key(label) for label in group]
        options = tuple(dict.fromkeys(terms for _, terms in keys))
        for phrase in keys:
            index[phrase] = options
    # Broad QA queries include testers. A QA Manager is not a full match for a tester query.
    index[key("QA")] = index[key("QA engineer")]
    return index, max(size for size, _ in index)


def title_variants(value, tokenize):
    original = frozenset(tokenize(value))
    words = [frozenset(tokenize(word)) for word in word_tokens(value)]
    index, longest = _phrase_index(tokenize)
    parts = []
    start = 0
    while start < len(words):
        for end in range(min(len(words), start + longest), start, -1):
            phrase = frozenset().union(*words[start:end])
            alternatives = index.get((end - start, phrase))
            if alternatives:
                choices = tuple(dict.fromkeys((phrase, *alternatives)))
                break
        else:
            end = start + 1
            choices = (words[start],)
        parts.append(choices)
        start = end
    variants = dict.fromkeys([TitleVariant(original)])
    if any(len(part) > 1 for part in parts):
        for choice in islice(product(*parts), 1, MAX_TITLE_VARIANTS):
            terms = frozenset().union(*choice)
            if terms != original:
                required = frozenset().union(
                    *(words for words, part in zip(choice, parts, strict=True) if words != part[0])
                )
                variants[TitleVariant(terms, required)] = None
    return tuple(variants)
