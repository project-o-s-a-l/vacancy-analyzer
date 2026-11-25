from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import event, insert, update

from domain import CandidateProfile
from domain.retrieval import RetrievalParameters
from domain.vacancy_analysis import analysis_input_hash
from domain.vacancy_duplicates import vacancy_duplicate_key
from jobs.candidates import retrieve_candidates
from jobs.notify import NotificationService
from jobs.rank import rank_recent
from jobs.retrieval_evaluation import MemoryCandidateSource
from scoring import rank_vacancies
from scoring.retrieval import document_terms, lexical_evidence, profile_query, search_document
from storage.models import VacancyModel
from storage.payloads import payload_hash
from storage.repositories import VacancyRepository
from storage.repositories.notification_repository import NotificationRepository


def candidate(**updates):
    data = {
        "skills": {"java": {}, "spring": {}},
        "experience_years": 3,
        "preferences": {"positions": ["Java Developer"]},
    }
    return CandidateProfile.model_validate(data | updates)


def job(vacancy, **updates):
    return type(vacancy).model_validate(
        vacancy.model_dump()
        | {
            "id": uuid4(),
            "source_id": uuid4().hex,
            "title": "Java Developer",
            "skills": ["Java", "Spring"],
            "description": "Разработка сервисов Java Spring",
            "description_html": None,
            "experience": None,
            "valid_until": None,
            "published_at": datetime.now(UTC) - timedelta(days=1),
        }
        | updates
    )


async def bulk(sessions, vacancies):
    from nlp.professional_facts import vacancy_facts
    from scoring.fact_projection import fact_projection, projection_contract, projection_revision

    values = []
    for vacancy in vacancies:
        row = vacancy.model_dump()
        row.update(
            url=str(vacancy.url),
            company_url=str(vacancy.company_url) if vacancy.company_url else None,
            last_seen_at=vacancy.collected_at,
            last_checked_at=vacancy.collected_at,
            last_check_attempt_at=vacancy.collected_at,
            raw_payload_hash=payload_hash(vacancy.raw_payload),
            duplicate_key=vacancy_duplicate_key(vacancy),
        )
        facts = vacancy_facts(vacancy)
        document = fact_projection(vacancy, facts)
        row.update(
            professional_facts=facts.model_dump(mode="json"),
            facts_input_revision=facts.input_sha256,
            facts_revision=facts.revision,
            projection_facts_revision=facts.revision,
            projection_revision=projection_revision(vacancy, facts),
            projection_contract=projection_contract(),
            search_version=document["version"],
            search_revision=analysis_input_hash(vacancy),
            content_revision=analysis_input_hash(vacancy),
            search_document=document,
            search_terms=document_terms(document),
        )
        values.append(row)
    async with sessions.begin() as session:
        await session.execute(insert(VacancyModel), values)


async def test_old_relevant_card_beyond_500_newer_directions(db_sessions, vacancy):
    sessions = db_sessions
    relevant = job(vacancy, published_at=datetime.now(UTC) - timedelta(days=200))
    unrelated = [
        job(
            vacancy,
            title=f"Python Developer {n}",
            skills=["python"],
            description="Разработка Python API",
        )
        for n in range(510)
    ]
    await bulk(sessions, [*unrelated, relevant])
    async with sessions() as session:
        selected = await retrieve_candidates(
            VacancyRepository(session), candidate(), parameters=RetrievalParameters(k=1)
        )
        assert [v.id for v in selected.items] == [relevant.id]
        assert selected.evidence[relevant.id].strategy == "lexical"
        assert "java" in selected.evidence[relevant.id].title_matches
        assert not selected.items[0].raw_payload  # Only a projection is loaded.
        assert selected.evidence[relevant.id].source_payload_hash == payload_hash(
            relevant.raw_payload
        )


@pytest.mark.parametrize(
    "change",
    [
        {"archived": True},
        {"valid_until": datetime.now(UTC) - timedelta(days=1)},
        {"published_at": datetime.now(UTC) + timedelta(days=1)},
        {"experience": "6+", "description": "Требования: коммерческий опыт работы от 6 лет"},
        {"location": "Другой город", "remote": False},
        {
            "salary_from": 1,
            "salary_to": 2,
            "salary_gross": True,
            "salary_period": "MONTH",
            "currency": "RUB",
        },
    ],
)
async def test_constraints_precede_k(db_sessions, vacancy, change):
    sessions = db_sessions
    prefs = {
        "positions": ["Java Developer"],
        "locations": ["Москва"],
        "locations_required": True,
        "salary_min": 100,
        "salary_required": True,
        "salary_gross": True,
    }
    profile = candidate(preferences=prefs)
    good = job(
        vacancy,
        location="Москва",
        remote=False,
        salary_from=1000,
        salary_to=2000,
        salary_period="MONTH",
        salary_gross=True,
        currency="RUB",
        title="Java",
    )
    bad = job(good, title="Java Developer", **change)
    await bulk(sessions, [bad, good])
    async with sessions() as session:
        result = await retrieve_candidates(
            VacancyRepository(session), profile, parameters=RetrievalParameters(k=1)
        )
        assert [v.id for v in result.items] == [good.id]


async def test_document_absence_precedes_k_and_unknowns_remain(db_sessions, vacancy):
    profile = candidate(qualifications=[{"name": "Медкнижка", "status": "absent"}])
    bad = job(vacancy, description="Требования: обязательна медицинская книжка. Java Spring")
    unknown = job(
        vacancy,
        title="Java",
        description="Java Spring",
        experience=None,
        remote=None,
        salary_from=None,
        salary_to=None,
    )
    await bulk(db_sessions, [bad, unknown])
    async with db_sessions() as session:
        result = await retrieve_candidates(
            VacancyRepository(session), profile, parameters=RetrievalParameters(k=1)
        )
        assert [v.id for v in result.items] == [unknown.id]


async def test_explicit_work_format_unknown_is_handled_by_existing_policy(db_sessions, vacancy):
    profile = candidate(preferences={"remote": "required"})
    unknown = job(vacancy, remote=None)
    remote = job(vacancy, title="Java", remote=True)
    await bulk(db_sessions, [unknown, remote])
    async with db_sessions() as session:
        result = await retrieve_candidates(
            VacancyRepository(session), profile, parameters=RetrievalParameters(k=1)
        )
        assert [v.id for v in result.items] == [remote.id]


@pytest.mark.parametrize(
    "skill,other", [("java", "javascript"), ("c", "c++"), ("c++", "c#"), ("c#", "c")]
)
def test_exact_technology_tokens(skill, other):
    query = profile_query(CandidateProfile(skills={skill: {}}))
    matching = search_document({"title": skill, "skills": [skill]})
    different = search_document({"title": other, "skills": [other]})
    assert query.terms.intersection(document_terms(matching))
    assert not query.terms.intersection(document_terms(different))


@pytest.mark.parametrize(
    "skill,text",
    [
        ("psql", "PostgreSQL"),
        ("разработчик Java", "Java developer"),
        ("аргонодуговая сварка", "Сварщик аргонодуговой сварки"),
    ],
)
def test_aliases_and_arbitrary_professions(skill, text):
    query = profile_query(CandidateProfile(skills={skill: {}}))
    doc = search_document({"title": text})
    assert query.terms <= set(document_terms(doc))


TITLE_EQUIVALENTS = [
    ("Разработчик Java", "Java Developer"),
    ("Питон-разработчик", "Python Developer"),
    ("Аналитик данных", "Data Analyst"),
    ("Инженер данных", "Data Engineer"),
    ("Системный аналитик", "Systems Analyst"),
    ("Бизнес-аналитик", "Business Analyst"),
    ("Продуктовый аналитик", "Product Analyst"),
    ("Старший аналитик данных", "Senior Data Analyst"),
    ("Исследователь данных", "Data Scientist"),
    ("Инженер машинного обучения", "ML Engineer"),
    ("Инженер по тестированию", "QA Engineer"),
    ("QA-инженер", "Тестировщик ПО"),
    ("Тестировщик-автоматизатор", "QA Automation Engineer"),
    ("AQA", "Инженер по автоматизированному тестированию"),
    ("Инженер по ручному тестированию", "Manual Tester"),
    ("Системный администратор", "Sysadmin"),
    ("DevOps-инженер", "DevOps Engineer"),
    ("Фулстек-разработчик", "Junior Full Stack Developer"),
    ("Бэкенд-разработчик Python", "Python Backend Developer"),
    ("Frontend разработчик", "Frontend Engineer"),
    ("Python-разработчик", "Software Engineer (Python)"),
    ("Менеджер продукта", "Product Manager"),
    ("Руководитель проектов", "Project Manager"),
    ("Преподаватель информатики", "Учитель информатики"),
    ("Бухгалтер", "Accountant"),
    ("Аналитик", "Senior Data Analyst"),
    ("Инженер", "Network Engineer"),
    ("Менеджер", "Product Manager"),
    ("Администратор", "Database Administrator"),
    ("Архитектор", "Cloud Architect"),
    ("Дизайнер", "Graphic Designer"),
    ("Рекрутер", "IT Recruiter"),
    ("QA", "Тестировщик"),
    ("Сварщик аргонодуговой сварки", "Сварщики аргонодуговой сварки"),
]


def title_evidence(position, title):
    query = profile_query(CandidateProfile(preferences={"positions": [position]}))
    return lexical_evidence(search_document({"title": title}), query, RetrievalParameters())


@pytest.mark.parametrize("position,title", TITLE_EQUIVALENTS)
def test_equivalent_title_gets_full_title_weight(position, title):
    assert title_evidence(position, title).lexical_score == 0.60


@pytest.mark.parametrize(
    "position,title",
    [
        ("Менеджер продукта", "Project Manager"),
        ("Системный администратор", "Системный аналитик"),
        ("Инженер данных", "Data Analyst"),
        ("Инженер по тестированию", "Инженер по охране труда"),
        ("Инженер по тестированию", "QA Manager"),
        ("Тестировщик-автоматизатор", "Manual QA Engineer"),
        ("Инженер по ручному тестированию", "QA Automation Engineer"),
        ("Бэкенд-разработчик Python", "Backend Developer Java"),
        ("Фулстек-разработчик", "Middle+/Senior Fullstack QA Engineer"),
        ("Senior Data Analyst", "Junior Data Analyst"),
        ("Java разработчик", "JavaScript Developer"),
        ("C++ разработчик", "C# Software Engineer"),
        ("Преподаватель информатики", "Учитель музыки"),
    ],
)
def test_related_title_does_not_get_full_title_weight(position, title):
    assert title_evidence(position, title).lexical_score < 0.60


def test_translated_role_needs_the_whole_phrase_and_keeps_other_qualifiers():
    assert title_evidence("Python developer", "Network Engineer").lexical_score == 0
    assert title_evidence("Старший аналитик данных", "Data Analyst").lexical_score == 0.4


async def test_title_alias_hits_existing_index_and_does_not_promote_body_mentions(
    db_sessions, vacancy
):
    now = datetime.now(UTC)
    profile = CandidateProfile(preferences={"positions": ["Аналитик данных"]})
    cards = [
        job(
            vacancy,
            title="Data Analyst",
            skills=[],
            description="Визуализация",
            published_at=now - timedelta(days=100),
        ),
        job(
            vacancy,
            title="Бизнес-аналитик",
            skills=[],
            description="Проектирование",
            published_at=now - timedelta(days=2),
        ),
        job(
            vacancy,
            title="Курьер",
            skills=[],
            description="Доставка",
            published_at=now - timedelta(days=1),
        ),
        job(
            vacancy,
            title="Ассистент",
            skills=[],
            description="В команде есть Data Analyst",
            published_at=now - timedelta(days=10),
        ),
    ]
    await bulk(db_sessions, cards)
    for backend in ("sql", "compact"):
        params = RetrievalParameters(k=3, backend=backend)
        async with db_sessions() as session:
            result = await retrieve_candidates(
                VacancyRepository(session), profile, parameters=params
            )
        offline = await retrieve_candidates(
            None, profile, parameters=params, candidate_source=MemoryCandidateSource(cards)
        )
        assert [v.id for v in result.items] == [v.id for v in cards[:3]]
        assert result.evidence == offline.evidence
        assert result.evidence[cards[0].id].lexical_score == 0.60
        assert 0 < result.evidence[cards[1].id].lexical_score < 0.60


async def test_description_search_and_sql_equivalence(db_sessions, vacancy):
    cards = [
        job(
            vacancy, title="Специалист", skills=[], description="Разработка сервисов на Java Spring"
        ),
        job(vacancy, title="Python", skills=["python"], description="Python API"),
        job(
            vacancy, title="Java стажировка", description="Без опыта работы. Java", skills=["Java"]
        ),
    ]
    await bulk(db_sessions, cards)
    for k in (1, 2, 5):
        params = RetrievalParameters(k=k)
        async with db_sessions() as session:
            actual = await retrieve_candidates(
                VacancyRepository(session), candidate(), parameters=params
            )
        reference = await retrieve_candidates(
            None, candidate(), parameters=params, candidate_source=MemoryCandidateSource(cards)
        )
        assert [v.id for v in actual.items] == [v.id for v in reference.items]
        assert actual.evidence == reference.evidence


async def test_duplicates_and_exclusions_do_not_consume_k(db_sessions, vacancy):
    first = job(vacancy)
    copies = [job(first) for _ in range(70)]
    other = job(
        vacancy, title="Java junior", description="Другой работодатель Java", company="Другой"
    )
    await bulk(db_sessions, [first, *copies, other])
    async with db_sessions() as session:
        selection = await retrieve_candidates(
            VacancyRepository(session), candidate(), parameters=RetrievalParameters(k=2)
        )
        assert len(selection.items) == 2
        selection = await retrieve_candidates(
            VacancyRepository(session),
            candidate(),
            parameters=RetrievalParameters(k=1),
            excluded_keys={vacancy_duplicate_key(first)},
        )
        assert [v.id for v in selection.items] == [other.id]


async def test_index_refresh_and_unindexed_compatibility(db_sessions, vacancy):
    v = job(vacancy, title="Java", skills=["Java"], description="Java")
    async with db_sessions.begin() as session:
        await VacancyRepository(session).save(v)
        changed = job(
            v,
            source_id=v.source_id,
            title="JavaScript",
            skills=["JavaScript"],
            description="JavaScript",
        )
        await VacancyRepository(session).save(changed)
    async with db_sessions.begin() as session:
        row = await session.get(VacancyModel, v.id)
        assert "javascript" in row.search_terms and "java" not in row.search_terms
        await session.execute(
            update(VacancyModel)
            .where(VacancyModel.id == v.id)
            .values(search_version=None, search_document=None, search_terms=None)
        )
    async with db_sessions() as session:
        result = await retrieve_candidates(
            VacancyRepository(session),
            CandidateProfile(skills={"javascript": {}}),
            parameters=RetrievalParameters(k=1),
        )
        assert result.items[0].id == v.id and result.evidence[v.id].strategy == "lexical"


async def test_revision_fences_old_process_index_and_next_save_repairs(db_sessions, vacancy):
    original = job(
        vacancy,
        title="Python",
        skills=["Python"],
        description="Python",
        published_at=datetime.now(UTC) - timedelta(days=200),
    )
    await bulk(
        db_sessions,
        [
            original,
            job(vacancy, title="JavaScript", skills=["JavaScript"], description="JavaScript"),
        ],
    )
    changed = original.model_copy(
        update={"title": "Java", "skills": ["Java"], "description": "Java"}
    )
    async with db_sessions.begin() as session:
        await session.execute(
            update(VacancyModel)
            .where(VacancyModel.id == original.id)
            .values(
                title=changed.title,
                skills=changed.skills,
                description=changed.description,
                content_revision=analysis_input_hash(changed),
            )
        )
    async with db_sessions() as session:
        result = await retrieve_candidates(
            VacancyRepository(session),
            CandidateProfile(skills={"java": {}}),
            parameters=RetrievalParameters(k=1),
        )
        assert result.items[0].id == original.id
    async with db_sessions.begin() as session:
        await VacancyRepository(session).save(changed)
    async with db_sessions() as session:
        row = await session.get(VacancyModel, original.id)
        assert row.search_revision == row.content_revision == analysis_input_hash(changed)
        assert "java" in row.search_terms and "python" not in row.search_terms


async def test_ranking_does_not_read_raw_html_and_keeps_source_hash(db_sessions, vacancy):
    v = job(vacancy)
    await bulk(db_sessions, [v])
    queries = []
    engine = db_sessions.kw["bind"].sync_engine

    def capture(conn, cursor, statement, parameters, context, executemany):
        queries.append(statement)

    event.listen(engine, "before_cursor_execute", capture)
    try:
        async with db_sessions() as session:
            actual = await rank_recent(VacancyRepository(session), candidate())
        assert len(actual) == 1 and actual[0][1].retrieval
        assert not any(
            "vacancies.raw_payload," in sql or "vacancies.raw_payload \n" in sql for sql in queries
        )
        reference = rank_vacancies([v], candidate())
        assert actual[0][1].score == reference[0][1].score
    finally:
        event.remove(engine, "before_cursor_execute", capture)


async def test_fallback_and_diagnostic_mode(db_sessions, vacancy):
    v = job(vacancy, title="Senior Java Developer", experience="6+", description="Java")
    await bulk(db_sessions, [v])
    async with db_sessions() as session:
        repo = VacancyRepository(session)
        assert await rank_recent(repo, candidate(experience_years=0)) == []
        assert (await rank_recent(repo, candidate(experience_years=0), include_ineligible=True))[0][
            1
        ].eligible is False
        result = await retrieve_candidates(repo, CandidateProfile(skills={"редкое умение": {}}))
        assert result.items[0].id == v.id and result.evidence[v.id].strategy == "recent_fallback"


def test_parameters_are_versioned_and_bounded():
    for values in ({"k": 0}, {"k": 5001}, {"batch_size": 0}, {"title_weight": 0.8}):
        with pytest.raises(ValueError):
            RetrievalParameters(**values)


async def test_index_does_not_confuse_technology_tokens_in_sql(db_sessions, vacancy):
    cards = [
        job(vacancy, title=value, skills=[value], description=value)
        for value in ("Java", "JavaScript", "C", "C++", "C#")
    ]
    await bulk(db_sessions, cards)
    async with db_sessions() as session:
        for card in cards:
            profile = CandidateProfile(skills={card.title: {}})
            result = await retrieve_candidates(
                VacancyRepository(session), profile, parameters=RetrievalParameters(k=1)
            )
            assert result.items[0].id == card.id


async def test_history_precedes_k_and_other_users_are_independent(db_sessions, vacancy):
    now = datetime.now(UTC)
    cards = [
        job(
            vacancy,
            title=f"Java Developer {n}",
            published_at=now - timedelta(days=1 if n < 64 else 2),
        )
        for n in range(70)
    ]
    await bulk(db_sessions, cards)
    async with db_sessions.begin() as session:
        repo = NotificationRepository(session)
        for card in cards[:64]:
            token = await repo.claim(42, card.id)
            await repo.finish(42, card.id, token, "sent", message_id=1)
    first = await NotificationService(db_sessions, candidate(), chat_id=42, min_score=0).pending()
    second = await NotificationService(db_sessions, candidate(), chat_id=43, min_score=0).pending()
    assert len(first) == len(second) == 5
    assert {v.id for v, _ in first} <= {v.id for v in cards[64:]}
    assert any(v.id in {c.id for c in cards[:64]} for v, _ in second)


async def test_explicit_days_and_profiles_do_not_share_selection(db_sessions, vacancy):
    old = job(vacancy, published_at=datetime.now(UTC) - timedelta(days=100))
    python = job(vacancy, title="Python", skills=["Python"], description="Python")
    await bulk(db_sessions, [old, python])
    async with db_sessions() as session:
        repo = VacancyRepository(session)
        java = await rank_recent(repo, candidate(), limit=1)
        other = await rank_recent(repo, CandidateProfile(skills={"python": {}}), limit=1)
        limited = await rank_recent(repo, candidate(), limit=1, days=7)
        assert java[0][0].id == old.id and other[0][0].id == python.id
        assert limited[0][0].id == python.id
