"""contracts, rollback wire compatibility, restart and real PostgreSQL CAS invariants."""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from typing import Literal

import pytest
from pydantic import BaseModel, ValidationError
from sqlalchemy import func, select, update

from domain import CandidateProfile, Skill
from domain.professional import (
    Claim,
    Evidence,
    ExperienceEpisode,
    ProfessionalFacts,
    RoleFact,
    SkillFact,
)
from jobs.fact_backfill import backfill_batch
from jobs.users import UserService
from nlp.professional_facts import candidate_facts, profile_payload
from scoring.fact_projection import projection_contract
from storage.models import (
    FactBackfillModel,
    FactRevisionModel,
    ResumeModel,
    TelegramUserModel,
    VacancyModel,
)
from storage.repositories.fact_repository import FactRepository
from storage.repositories.user_repository import StaleProfileWrite, UserRepository
from storage.repositories.vacancy_repository import VacancyRepository


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def fact_sessions(request):
    return request.getfixturevalue(
        "postgres_sessions" if request.param == "postgres" else "sessions"
    )


def test_states_are_distinct_and_no_implicit_skill_level():
    assert Skill().level is None
    assert Claim(status="missing") != Claim(status="unknown")
    assert Claim(status="explicit_absent") != Claim(status="missing")
    conflict = Claim[str](status="conflict", alternatives=["junior", "senior"])
    assert conflict.value is None
    for kwargs in (
        dict(status="unknown", value=0),
        dict(status="present"),
        dict(status="conflict", alternatives=["senior"]),
        dict(status="present", value="senior", origin="machine_proposed"),
    ):
        with pytest.raises(ValidationError):
            Claim(**kwargs)


def test_published_schema_and_examples_match_runtime():
    import json
    from pathlib import Path

    from domain import Vacancy
    from nlp.professional_facts import profile_input_revision, vacancy_fact_input_revision

    directory = Path(__file__).resolve().parents[1] / "docs/contracts"

    def read(name):
        return json.loads((directory / name).read_text(encoding="utf-8"))

    assert read("professional-facts-v1.schema.json") == ProfessionalFacts.model_json_schema()
    candidate = ProfessionalFacts.model_validate(read("candidate-facts.example.json"))
    vacancy = ProfessionalFacts.model_validate(read("vacancy-facts.example.json"))
    assert candidate.input_sha256 == profile_input_revision(
        CandidateProfile.model_validate(read("candidate-profile.example.json"))
    )
    assert vacancy.input_sha256 == vacancy_fact_input_revision(
        Vacancy.model_validate(read("vacancy-source.example.json"))
    )


def test_roundtrip_rich_candidate_and_reference_validation():
    profile = CandidateProfile(skills={"python": {"level": "intermediate"}})
    facts = candidate_facts(profile, 17)
    skill = next(f for f in facts.facts if f.kind == "skill")
    assert skill.level.status == "unknown" and skill.level.legacy_value == "intermediate"
    role = RoleFact(id="role:engineer", name="Инженер")
    data = facts.model_dump(mode="json")
    data["facts"].append(role.model_dump(mode="json"))
    data["requirements"] = [
        {
            "id": "req:skill",
            "fact_id": skill.id,
            "importance": {"status": "conflict", "alternatives": ["required", "preferred"]},
        }
    ]
    data["episodes"] = [
        ExperienceEpisode(
            id="job:1", kind="project", role_ids=[role.id], skill_ids=[skill.id]
        ).model_dump(mode="json")
    ]
    rich = ProfessionalFacts.model_validate(data)
    assert ProfessionalFacts.model_validate_json(rich.model_dump_json()) == rich
    assert rich.overlap_policy == "unaggregated"
    assert rich.requirements[0].importance.status == "conflict"
    assert rich.requirements[0].importance.value is None
    assert not any(f.kind == "experience_total" for f in rich.facts)
    data["episodes"][0]["role_ids"] = [skill.id]
    with pytest.raises(ValidationError, match="episode reference"):
        ProfessionalFacts.model_validate(data)


def test_evidence_spans_are_exact_unicode_and_source_bound():
    facts = candidate_facts(CandidateProfile(skills={"python": {}}), 17)
    skill = next(f for f in facts.facts if f.kind == "skill")
    source = facts.sources[0]
    skill.assertion = Claim[bool](
        status="present",
        value=True,
        origin="machine_proposed",
        evidence=[
            Evidence(
                source_id=source.id,
                source_revision=source.revision,
                field="text",
                start=2,
                end=8,
                quote="Python",
            )
        ],
    )
    facts.verify_text(source.id, source.revision, {"text": "Я Python"})
    with pytest.raises(ValueError, match="source text"):
        facts.verify_text(source.id, source.revision, {"text": "Я Cobol!"})
    data = facts.model_dump(mode="json")
    data["facts"][0]["assertion"]["evidence"][0]["source_revision"] = "0" * 64
    with pytest.raises(ValidationError, match="undeclared source"):
        ProfessionalFacts.model_validate(data)


async def test_legacy_profile_roundtrip_and_rollback_wire(fact_sessions):
    sessions = fact_sessions
    legacy = dict(skills={"python": {"level": "intermediate", "years": 3}})
    async with sessions.begin() as session:
        await UserRepository(session).register(17)
        await session.execute(
            update(TelegramUserModel)
            .where(TelegramUserModel.user_id == 17)
            .values(profile=legacy, professional_facts=None, facts_input_revision=None)
        )
    async with sessions.begin() as session:
        repo = UserRepository(session)
        user = await repo.get(17)
        skill = next(f for f in user.profile.professional_facts.facts if f.kind == "skill")
        assert skill.level.status == "unknown"
        assert skill.level.origin == "legacy_unknown"
        assert skill.level.legacy_value == "intermediate"
        updated = await repo.save(user)
        assert (await repo.get(17)).profile == updated.profile
        assert (await session.get(TelegramUserModel, 17)).profile["skills"] == legacy["skills"]

    class PreviousSkill(BaseModel):
        level: Literal["beginner", "intermediate", "advanced"] = "intermediate"
        years: float | None = None

    # Old application can still read newly entered skills; no null level on its wire.
    data = profile_payload(CandidateProfile(skills={"new": {}}))
    assert "level" not in data["skills"]["new"]
    PreviousSkill.model_validate(data["skills"]["new"])
    users = UserService(sessions)
    draft = await users.resumes.prepare(17, "Skills: Python\nDesired position: Developer")
    confirmed = await users.resumes.confirm(17, draft.token)
    retained = next(f for f in confirmed.profile.professional_facts.facts if f.kind == "skill")
    assert retained.level.status == "unknown" and retained.level.legacy_value == "intermediate"
    assert retained.years.status == "unknown" and retained.years.legacy_value == 3


async def test_resume_suggestion_and_confirmation_are_separate(fact_sessions):
    users = UserService(fact_sessions)
    await users.register(17)
    draft = await users.resumes.prepare(17, "Skills: Python, SQL\nDesired position: Developer")
    proposed = [f for f in draft.professional_facts.facts if f.kind == "skill"]
    assert proposed and all(f.assertion.origin == "machine_proposed" for f in proposed)
    assert (await users.get(17)).profile.skills == {}
    confirmed = await users.resumes.confirm(17, draft.token)
    facts = confirmed.profile.professional_facts
    skills = [f for f in facts.facts if f.kind == "skill"]
    assert all(f.assertion.origin == "user_confirmed" for f in skills)
    assert all(f.level.status == "missing" for f in skills)
    positions = next(f for f in facts.facts if f.id == "preference:positions")
    assert positions.assertion.origin == "user_confirmed"
    assert facts.reference_time == draft.professional_facts.reference_time
    async with fact_sessions() as session:
        assert (await UserRepository(session).get(17)).profile == confirmed.profile
        history = await FactRepository(session).history("resume_proposal", 17)
        assert any(f.assertion.origin == "machine_proposed" for f in history[0].facts)


async def test_stale_profile_and_source_aba_cannot_overwrite(fact_sessions):
    sessions = fact_sessions
    users = UserService(sessions)
    await users.register(17)
    async with sessions() as session:
        first = await UserRepository(session).get(17)
    async with sessions.begin() as session:
        await UserRepository(session).save(replace(first, form_state="edit:experience"))
    async with sessions.begin() as session:
        with pytest.raises(StaleProfileWrite):
            await UserRepository(session).save(first)
    draft = await users.resumes.prepare(17, "Skills: Python\nDesired position: Developer")
    await users.accept_answer(17, "1")
    await users.begin_edit(17, "experience")
    await users.accept_answer(17, "пропустить")
    # Same profile values, newer revision: hash-only guard would admit stale work.
    with pytest.raises(ValueError, match="Профиль изменился"):
        await users.resumes.confirm(17, draft.token)


@pytest.mark.postgres
async def test_concurrent_profile_cas_has_one_winner(postgres_sessions):
    await UserService(postgres_sessions).register(17)
    async with postgres_sessions() as session:
        before = await UserRepository(session).get(17)
    ready = asyncio.Event()
    count = 0

    async def write(years):
        nonlocal count
        async with postgres_sessions.begin() as session:
            user = replace(
                before, profile=before.profile.model_copy(update={"experience_years": years})
            )
            count += 1
            if count == 2:
                ready.set()
            await ready.wait()
            try:
                return await UserRepository(session).save(user, confirmed_fields={"experience"})
            except StaleProfileWrite:
                return None

    results = await asyncio.gather(write(3), write(7))
    assert sum(r is not None for r in results) == 1
    async with postgres_sessions() as session:
        user = await UserRepository(session).get(17)
        assert user.revision == 1
        assert user.profile.experience_years in {3, 7}


async def test_facts_projection_cas_history_and_transaction_rollback(fact_sessions, vacancy):
    sessions = fact_sessions
    async with sessions.begin() as session:
        key = await VacancyRepository(session).save(vacancy)
    async with sessions() as session:
        before = await VacancyRepository(session).get_by_id(key)
    facts = before.professional_facts
    data = facts.model_dump(mode="json")
    evidence = next(f for f in facts.facts if f.kind == "role").assertion.evidence
    data["facts"].append(
        SkillFact(
            id="skill:novel",
            name="toolnova",
            assertion=Claim[bool](
                status="present", value=True, origin="machine_proposed", evidence=evidence
            ),
        ).model_dump(mode="json")
    )
    new = ProfessionalFacts.model_validate(data)
    with pytest.raises(RuntimeError):
        async with sessions.begin() as session:
            assert await VacancyRepository(session).set_facts(
                key, new, expected_facts_revision=facts.revision
            )
            raise RuntimeError("simulated interruption before commit")
    async with sessions.begin() as session:
        repo = VacancyRepository(session)
        assert (await repo.get_by_id(key)).professional_facts.revision == facts.revision
        assert await repo.set_facts(key, new, expected_facts_revision=facts.revision)
        assert not await repo.set_facts(key, facts, expected_facts_revision=facts.revision)
    async with sessions() as session:
        row = await session.get(VacancyModel, key)
        assert "toolnova" in row.search_terms
        assert row.projection_contract == projection_contract()
        assert len(await FactRepository(session).history("vacancy", key)) == 2
    async with sessions.begin() as session:
        repo = VacancyRepository(session)
        await repo.save(before.model_copy(update={"description": "Changed source text"}))
        assert not await repo.set_facts(key, new, expected_facts_revision=new.revision)


async def test_backfill_commits_cursor_with_data_and_resumes(fact_sessions, vacancy, monkeypatch):
    sessions = fact_sessions
    for owner in (11, 12, 13):
        await UserService(sessions).register(owner)
    async with sessions.begin() as session:
        await session.execute(
            update(TelegramUserModel).values(
                profile={"skills": {"python": {"level": "intermediate"}}},
                professional_facts=None,
                facts_input_revision=None,
            )
        )
        key = await VacancyRepository(session).save(vacancy)
        await session.execute(
            update(VacancyModel).values(professional_facts=None, facts_input_revision=None)
        )
    first = await backfill_batch(sessions, batch_size=1)
    assert first["cursor"] == {"stage": "users", "after": 11}
    original = FactRepository.record

    async def fail(*args, **kwargs):
        raise RuntimeError("process interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(FactRepository, "record", fail)
        with pytest.raises(RuntimeError):
            await backfill_batch(sessions, batch_size=1)
    async with sessions() as session:
        assert (await session.get(TelegramUserModel, 12)).professional_facts is None
        assert (await session.scalar(select(FactBackfillModel))).cursor == first["cursor"]
    assert FactRepository.record is original
    for _ in range(10):
        result = await backfill_batch(sessions, batch_size=1)
        if result["done"]:
            break
    assert result["done"]
    async with sessions() as session:
        count = await session.scalar(select(func.count()).select_from(FactRevisionModel))
        assert (await session.get(VacancyModel, key)).professional_facts
    assert (await backfill_batch(sessions, batch_size=1))["done"]
    # Restart a complete audit; deterministic legacy revision prevents duplicate history.
    await backfill_batch(sessions, batch_size=2, restart=True)
    for _ in range(10):
        if (await backfill_batch(sessions, batch_size=2))["done"]:
            break
    async with sessions() as session:
        assert await session.scalar(select(func.count()).select_from(FactRevisionModel)) == count


async def test_old_application_write_preserves_new_history_and_repairs_sidecar(fact_sessions):
    sessions = fact_sessions
    users = UserService(sessions)
    await users.register(17)
    await users.accept_answer(17, "Python")
    async with sessions.begin() as session:
        old = await UserRepository(session).get(17)
        old_revision = old.profile.professional_facts.revision
        # 0013 writes only its known JSON; sidecar/history remain in 0014.
        await session.execute(
            update(TelegramUserModel)
            .where(TelegramUserModel.user_id == 17)
            .values(profile={"skills": {"java": {"level": "intermediate"}}})
        )
    current = await users.get(17)
    assert set(current.profile.skills) == {"java"}
    assert (
        current.profile.professional_facts.input_sha256
        != old.profile.professional_facts.input_sha256
    )
    async with sessions.begin() as session:
        with pytest.raises(StaleProfileWrite):
            await UserRepository(session).save(old)
    await backfill_batch(sessions, batch_size=10, restart=True)
    async with sessions() as session:
        history = await FactRepository(session).history("candidate", 17)
        assert old_revision in {f.revision for f in history}
        assert any(any(f.name == "java" for f in h.facts) for h in history)


def test_contracts_ignore_ui_and_invalidate_normalization(tmp_path, monkeypatch):
    from domain import contracts
    from scoring.serving_features import feature_contract

    root = contracts.SOURCE_ROOT
    # Use a shadow tree, never modify source files being executed by other tests.
    import shutil

    for directory in ("nlp", "scoring", "processing", "domain"):
        shutil.copytree(root / directory, tmp_path / directory)
    monkeypatch.setattr(contracts, "SOURCE_ROOT", tmp_path)

    def refresh():
        contracts.source_contract.cache_clear()
        contracts.normalization_contract.cache_clear()
        contracts.facts_contract.cache_clear()
        feature_contract.cache_clear()
        return feature_contract()

    try:
        before = refresh()
        (tmp_path / "domain" / "unrelated_ui.py").write_text("# unrelated view\n")
        assert refresh() == before
        path = tmp_path / "nlp" / "skills.py"
        path.write_bytes(path.read_bytes() + b"\n# normalization dependency change\n")
        assert refresh() != before
    finally:
        monkeypatch.setattr(contracts, "SOURCE_ROOT", root)
        refresh()


async def test_current_facts_read_and_projection_do_not_reextract(
    fact_sessions, vacancy, monkeypatch
):
    async with fact_sessions.begin() as session:
        key = await VacancyRepository(session).save(vacancy)

    def fail(*args, **kwargs):
        raise AssertionError("current facts must not reparse source")

    monkeypatch.setattr("nlp.requirements.analyze_requirements", fail)
    async with fact_sessions.begin() as session:
        repo = VacancyRepository(session)
        before = await repo.get_by_id(key)
        assert await repo.refresh_facts(key)
        assert (
            await repo.get_by_id(key)
        ).professional_facts.revision == before.professional_facts.revision


@pytest.mark.postgres
async def test_0013_data_drafts_and_exposure_survive_real_migration(postgres_sessions, vacancy):
    from pathlib import Path
    from uuid import uuid4

    from alembic.autogenerate import compare_metadata
    from alembic.config import Config
    from alembic.migration import MigrationContext
    from sqlalchemy import MetaData, Table

    from alembic import command
    from domain.resume import ResumeDraft
    from jobs.resumes import profile_hash
    from nlp.resume import extract_resume
    from storage.models import Base, NotificationModel

    engine = postgres_sessions.kw["bind"]
    config = Config(str(Path(__file__).parents[1] / "alembic.ini"))
    legacy_profile = CandidateProfile(skills={"python": {"level": "intermediate"}})
    text = "Skills: Python, SQL\nDesired position: Developer"
    draft = ResumeDraft(
        token="a" * 32,
        text=text,
        format="text",
        extraction=extract_resume(text),
        profile_hash=profile_hash(legacy_profile),
    )
    old_draft = draft.model_dump(mode="json", exclude={"professional_facts", "profile_revision"})
    exposure = {
        "profile": profile_payload(legacy_profile),
        "ranker_context": {"version": "ranking-exposure-v1"},
    }

    def migrate(connection):
        config.attributes["connection"] = connection
        command.upgrade(config, "0013")
        metadata = MetaData()
        users = Table("telegram_users", metadata, autoload_with=connection)
        resumes = Table("user_resumes", metadata, autoload_with=connection)
        vacancies = Table("vacancies", metadata, autoload_with=connection)
        notifications = Table("telegram_notifications", metadata, autoload_with=connection)
        timestamp = datetime.now(UTC)
        connection.execute(
            users.insert().values(
                user_id=17,
                profile=profile_payload(legacy_profile),
                created_at=timestamp,
                updated_at=timestamp,
            )
        )
        connection.execute(
            resumes.insert().values(user_id=17, draft=old_draft, text="Previous resume")
        )
        values = vacancy.model_dump(mode="python", exclude={"professional_facts"})
        values.update(url=str(vacancy.url), company_url=None, last_seen_at=vacancy.collected_at)
        connection.execute(
            vacancies.insert().values(**{k: v for k, v in values.items() if k in vacancies.c})
        )
        connection.execute(
            notifications.insert().values(
                chat_id=17,
                vacancy_id=vacancy.id,
                claim_token=uuid4(),
                status="sent",
                attempted_at=timestamp,
                message_id=123,
                attempts=1,
                exposure_context=exposure,
            )
        )
        command.upgrade(config, "head")
        assert compare_metadata(MigrationContext.configure(connection), Base.metadata) == []

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(migrate)
    async with postgres_sessions() as session:
        assert (await session.get(ResumeModel, 17)).draft == old_draft
        assert (await session.get(NotificationModel, (17, vacancy.id))).exposure_context == exposure
    for _ in range(5):
        if (await backfill_batch(postgres_sessions, batch_size=1))["done"]:
            break
    async with postgres_sessions() as session:
        user = await UserRepository(session).get(17)
        skill = next(f for f in user.profile.professional_facts.facts if f.kind == "skill")
        assert skill.level.status == "unknown" and skill.level.legacy_value == "intermediate"
        assert (await session.get(ResumeModel, 17)).text == "Previous resume"
        assert (await session.get(ResumeModel, 17)).draft["professional_facts"]
        assert (await session.get(NotificationModel, (17, vacancy.id))).exposure_context == exposure
        assert (
            await VacancyRepository(session).get_by_id(vacancy.id)
        ).raw_payload == vacancy.raw_payload


async def test_memory_retrieval_category_and_compact_scan_match_full_load(vacancy):
    from domain.retrieval import RetrievalParameters
    from domain.vacancy_analysis import VacancyCategory
    from jobs.retrieval_evaluation import MemoryCandidateSource
    from scoring.retrieval import profile_query

    source = MemoryCandidateSource([vacancy])
    query = profile_query(CandidateProfile(skills={"python": {}}))
    options = dict(
        now=vacancy.published_at,
        since=None,
        category=None,
        excluded_ids=(),
        excluded_keys=(),
        positive=True,
        include_ineligible=False,
        limit=8,
    )
    rows = await source.shortlist(CandidateProfile(), query, RetrievalParameters(), **options)
    assert rows and "professional_facts" not in rows[0] and "description" not in rows[0]
    loaded = (await source.load_selected([vacancy.id]))[0]
    assert source.vacancy(loaded).professional_facts
    assert (
        await source.shortlist(
            CandidateProfile(),
            query,
            RetrievalParameters(),
            **(options | {"category": VacancyCategory.IT}),
        )
        == []
    )
