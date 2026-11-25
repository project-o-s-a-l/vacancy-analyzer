"""Unit boundaries and SQL/offline selection parity."""

import copy
import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest
from sqlalchemy import select

from domain import CandidateProfile, Vacancy
from domain.evaluation import EvaluationData
from domain.evaluation_units import UNIT_CONTRACT, validate_units
from domain.retrieval import RetrievalParameters
from jobs.candidates import retrieve_candidates
from jobs.cli import parser, run_command
from jobs.evaluation_data import load_data, prepare, synthetic_data
from jobs.evaluation_migration import migrate_units
from jobs.retrieval_evaluation import MemoryCandidateSource, compact_available
from scoring.eligibility import check_eligibility
from storage.models import VacancyModel
from storage.repositories import VacancyRepository
from storage.repositories.candidate_repository import CandidateRepository

PROTOCOL = Path(__file__).resolve().parents[1] / "config/evaluation/protocol.json"


@pytest.fixture(scope="module")
def unit_data():
    return synthetic_data(20261102)


@pytest.fixture
def unit_bundle(tmp_path, unit_data):
    source = tmp_path / "source.json"
    raw = unit_data.model_dump(mode="json")
    raw["profiles"] = [p for p in raw["profiles"] if p["profession"] == "python"]
    raw["vacancies"] = [v for v in raw["vacancies"] if "-python-" in v["id"]]
    source.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "bundle"
    prepare(output, PROTOCOL, data_path=source)
    return output


def rewrite_bundle(bundle, mutate, *, legacy=False):
    """Only temporary test fixtures, never the historical evaluation bundle."""
    raw = json.loads((bundle / "data.json").read_text(encoding="utf-8"))
    mutate(raw)
    (bundle / "data.json").write_text(json.dumps(raw), encoding="utf-8")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    manifest["sha256"]["data.json"] = sha256((bundle / "data.json").read_bytes()).hexdigest()
    if legacy:
        manifest["version"] = "working-evaluation-bundle-v1"
        manifest.pop("unit_contract")
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_generator_units_do_not_change_document_fact_schema(unit_data):
    assert unit_data.version == "working-evaluation-data-v2"
    assert {v.vacancy.salary_period for v in unit_data.vacancies} == {"MONTH"}
    salaries = [f for p in unit_data.profiles for f in p.draft_facts if f.field == "salary"]
    assert salaries and all(f.value["period"] == "month" for f in salaries)
    for split in ("train", "development", "calibration"):
        # Every required-salary profile, no special casing of the opened test IDs.
        for p in unit_data.profiles:
            if p.split != split or not p.profile.preferences.salary_required:
                continue
            v = next(
                v
                for v in unit_data.vacancies
                if v.split == split and v.vacancy.title in p.profile.preferences.positions
            )
            result = check_eligibility(v.vacancy, p.profile, now=unit_data.reference_times[split])
            assert "salary_incomparable" not in {issue.code for issue in result.exclusions}


@pytest.mark.parametrize(
    "field,value",
    [
        ("salary_period", "month"),
        ("salary_period", "MONTH "),
        ("salary_period", "FORTNIGHT"),
        ("currency", "rub"),
        ("currency", "RUR"),
        ("currency", "₽"),
        ("currency", ""),
        ("salary_gross", 1),
        ("remote", "false"),
        ("remote", 0),
        ("location", ""),
        ("location", " Москва "),
    ],
)
def test_vacancy_units_reject_noncanonical_values(field, value):
    with pytest.raises(ValueError, match=field):
        validate_units({"vacancies": [{"vacancy": {field: value}}]})


def test_new_bundle_rejects_boolean_coercion_even_with_legacy_enabled(unit_bundle):
    rewrite_bundle(
        unit_bundle, lambda raw: raw["vacancies"][0]["vacancy"].update(salary_gross="false")
    )
    for allow_legacy in (False, True):
        with pytest.raises(ValueError, match="salary_gross"):
            load_data(unit_bundle, allow_legacy=allow_legacy)


@pytest.mark.parametrize("field", ["currency", "salary_gross", "salary_period"])
def test_salary_amount_requires_explicit_units_even_if_unknown(unit_data, field):
    raw = unit_data.model_dump(mode="json")
    del raw["vacancies"][0]["vacancy"][field]
    with pytest.raises(ValueError, match="explicit unit"):
        EvaluationData.model_validate(raw)
    raw["vacancies"][0]["vacancy"][field] = None
    assert getattr(EvaluationData.model_validate(raw).vacancies[0].vacancy, field) is None


@pytest.mark.parametrize("field,value", [("currency", "rub"), ("salary_gross", "false")])
def test_profile_units_cannot_be_silently_normalized(unit_data, field, value):
    raw = unit_data.model_dump(mode="json")
    raw["profiles"][0]["profile"]["preferences"][field] = value
    with pytest.raises(ValueError, match=field):
        EvaluationData.model_validate(raw)


@pytest.mark.parametrize("field", ["currency", "salary_gross"])
def test_profile_salary_cannot_default_its_units(unit_data, field):
    raw = unit_data.model_dump(mode="json")
    prefs = raw["profiles"][0]["profile"]["preferences"]
    prefs["salary_min"] = 60000
    del prefs[field]
    with pytest.raises(ValueError, match="explicit unit"):
        EvaluationData.model_validate(raw)


def test_legacy_archive_is_explicit_and_does_not_rewrite_bytes(legacy_unit_bundle, tmp_path):
    before = {p.name: p.read_bytes() for p in legacy_unit_bundle.iterdir()}
    with pytest.raises(ValueError, match="Legacy evaluation bundle"):
        load_data(legacy_unit_bundle)
    with pytest.warns(UserWarning, match="archival inspection"):
        data, _, _ = load_data(legacy_unit_bundle, allow_legacy=True)
    assert {v.vacancy.salary_period for v in data.vacancies} == {"month"}
    assert before == {p.name: p.read_bytes() for p in legacy_unit_bundle.iterdir()}
    output = tmp_path / "reimport"
    with pytest.raises(ValueError, match="New imports require"):
        prepare(output, PROTOCOL, data_path=legacy_unit_bundle / "data.json")
    assert not output.exists()


def test_manifest_cannot_disguise_legacy_data(unit_bundle):
    rewrite_bundle(unit_bundle, lambda raw: raw.update(version="working-evaluation-data-v1"))
    with pytest.raises(ValueError, match="version mismatch"):
        load_data(unit_bundle)


@pytest.fixture
def legacy_unit_bundle(unit_bundle):
    def legacy(raw):
        raw["version"] = "working-evaluation-data-v1"
        for v in raw["vacancies"]:
            v["vacancy"]["salary_period"] = "month"

    rewrite_bundle(unit_bundle, legacy, legacy=True)
    return unit_bundle


def machine_draft(bundle):
    with pytest.warns(UserWarning):
        data, _, bundle_hash = load_data(bundle, allow_legacy=True)
    return {
        "bundle_hash": bundle_hash,
        "origin": "assistant_draft",
        "human_reviewed": False,
        "acceptance_eligible": False,
        "records": {
            f"pair:{p.id}:{v.id}": {"relevance": 1, "violates_required": False}
            for p in data.profiles
            if p.split != "test"
            for v in data.vacancies
            if v.split == p.split
        },
    }


async def test_explicit_migration_preserves_sources_labels_and_test_status(
    legacy_unit_bundle,
    tmp_path,
):
    bundle = legacy_unit_bundle
    before = {p.name: p.read_bytes() for p in bundle.iterdir()}
    draft = machine_draft(bundle)
    draft_path = tmp_path / "draft.json"
    draft_path.write_text(json.dumps(draft), encoding="utf-8")
    output = tmp_path / "migration"
    with pytest.warns(UserWarning):
        await run_command(
            parser().parse_args(
                [
                    "evaluation-migrate-units",
                    "--bundle",
                    str(bundle),
                    "--output",
                    str(output),
                    "--draft-labels",
                    str(draft_path),
                ]
            )
        )
    migrated, _, new_hash = load_data(output / "bundle")
    assert {v.vacancy.salary_period for v in migrated.vacancies} == {"MONTH"}
    receipt = json.loads((output / "migration.json").read_text(encoding="utf-8"))
    assert len(receipt["changes"]) == len(migrated.vacancies)
    assert receipt["independent_holdout"] is False and receipt["other_semantic_fields_unchanged"]
    moved = json.loads((output / "draft-labels.json").read_text(encoding="utf-8"))
    assert moved["records"] == draft["records"] and moved["bundle_hash"] == new_hash
    assert moved["parent_bundle_hash"] == draft["bundle_hash"] != new_hash
    assert moved["training_authorized"] is False and moved["acceptance_eligible"] is False
    assert not (output / "bundle/test-releases").exists()
    assert before == {p.name: p.read_bytes() for p in bundle.iterdir()}
    assert json.loads(draft_path.read_text(encoding="utf-8")) == draft
    with pytest.raises(FileExistsError):
        migrate_units(bundle, output)
    with pytest.raises(ValueError, match="outside"):
        migrate_units(bundle, bundle / "nested-copy")


@pytest.mark.parametrize("invalid", ["human", "bundle_hash", "test", "evidence", "units"])
def test_migration_rejects_unsafe_rebinding_before_writes(legacy_unit_bundle, tmp_path, invalid):
    draft = machine_draft(legacy_unit_bundle)
    if invalid == "human":
        draft["origin"] = "human"
    elif invalid == "bundle_hash":
        draft["bundle_hash"] = "wrong"
    elif invalid == "test":
        draft["records"]["pair:test-extra"] = {"relevance": 1, "violates_required": False}
    elif invalid == "evidence":
        next(iter(draft["records"].values()))["structured_evidence"] = []
    else:
        rewrite_bundle(
            legacy_unit_bundle,
            lambda raw: raw["vacancies"][0]["vacancy"].update(salary_period="hour"),
        )
    path = tmp_path / "draft.json"
    path.write_text(json.dumps(draft), encoding="utf-8")
    output = tmp_path / "rejected"
    with pytest.warns(UserWarning), pytest.raises(ValueError):
        migrate_units(legacy_unit_bundle, output, draft_labels=path)
    assert not output.exists()


async def test_cli_validate_import_and_reject_before_final_marker(unit_bundle, tmp_path, capsys):
    output = tmp_path / "imported"
    await run_command(
        parser().parse_args(
            [
                "evaluation-prepare",
                "--data",
                str(unit_bundle / "data.json"),
                "--output",
                str(output),
            ]
        )
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["unit_contract"] == UNIT_CONTRACT
    assert manifest["purpose"] == "imported_requires_review"
    assert load_data(output)[0] == load_data(unit_bundle)[0]
    capsys.readouterr()
    await run_command(parser().parse_args(["evaluation-validate", "--bundle", str(output)]))
    assert json.loads(capsys.readouterr().out)["unit_contract_validated"] is True
    rewrite_bundle(output, lambda raw: raw["vacancies"][0]["vacancy"].update(salary_period="month"))
    with pytest.raises(ValueError, match="salary_period"):
        await run_command(
            parser().parse_args(
                [
                    "evaluation-run",
                    "--bundle",
                    str(output),
                    "--output",
                    str(tmp_path / "run"),
                    "--split",
                    "test",
                    "--final-test",
                    "--release-id",
                    "unit-only",
                    "--test-role",
                    "candidate",
                ]
            )
        )
    assert not (output / "test-releases").exists()
    assert not (tmp_path / "run").exists()


# Independent expectation sets, normalized cases for both selection paths.
CASES = {
    "monthly": {},
    **{p.lower(): {"salary_period": p} for p in ("HOUR", "YEAR", "DAY", "WEEK", "SHIFT")},
    "period_unknown": {"salary_period": None},
    "usd": {"currency": "USD"},
    "currency_unknown": {"currency": None},
    "net": {"salary_gross": False},
    "tax_unknown": {"salary_gross": None},
    "below": {"salary_from": 45000, "salary_to": 55000},
    "upper_only": {"salary_from": None},
    "lower_only": {"salary_to": None},
    "lower_insufficient": {"salary_from": 45000, "salary_to": None},
    "amount_unknown": {"salary_from": None, "salary_to": None},
    "boundary_upper": {"salary_from": None, "salary_to": 60000},
    "boundary_lower": {"salary_from": 60000, "salary_to": None},
    "zero": {"salary_from": 0, "salary_to": 0},
    "other_city": {"location": "Казань"},
    "city_unknown": {"location": None},
    "city_prefix": {"location": "г. Москва"},
    "remote": {"remote": True, "location": "Казань"},
    "remote_unknown": {"remote": None},
}
SALARY_EXCLUDED = {
    "hour",
    "year",
    "day",
    "week",
    "shift",
    "period_unknown",
    "usd",
    "currency_unknown",
    "net",
    "tax_unknown",
    "below",
    "lower_insufficient",
    "amount_unknown",
    "zero",
}
PROFILES = [
    ({}, set()),
    ({"salary_min": 60000}, set()),
    ({"salary_min": 60000, "salary_required": True}, SALARY_EXCLUDED),
    ({"salary_min": 60000, "salary_required": True, "salary_gross": False}, set(CASES) - {"net"}),
    ({"salary_min": 60000, "salary_required": True, "currency": "USD"}, set(CASES) - {"usd"}),
    ({"remote": "required"}, set(CASES) - {"remote"}),
    ({"remote": "office"}, {"remote", "remote_unknown"}),
    ({"locations": ["Москва"], "locations_required": True}, {"other_city", "city_unknown"}),
    ({"locations": ["Москва"]}, set()),
    (
        {
            "salary_min": 60000,
            "salary_required": True,
            "locations": ["Москва"],
            "locations_required": True,
        },
        SALARY_EXCLUDED | {"other_city", "city_unknown"},
    ),
]


def source_card(name, changes):
    conditions = (
        dict(
            salary_from=70000,
            salary_to=95000,
            currency="RUB",
            salary_gross=True,
            salary_period="MONTH",
            location="Москва",
            remote=False,
        )
        | changes
    )
    card = Vacancy(
        source="sample",
        source_id=name,
        url=f"https://example.invalid/jobs/{name}",
        title="Python Developer",
        company=name,
        description="Разработка сервисов Python",
        skills=["Python"],
        published_at="2026-07-08T00:00:00Z",
        raw_payload={},
        **conditions,
    )
    validate_units({"vacancies": [{"id": name, "vacancy": card}]})
    return card


@pytest.mark.postgres
async def test_sql_offline_and_eligibility_parity(postgres_sessions):
    cards = [source_card(name, changes) for name, changes in CASES.items()]
    now = datetime(2026, 8, 1, tzinfo=UTC)
    memory = MemoryCandidateSource(cards)
    options = dict(
        now=now,
        since=None,
        category=None,
        include_ineligible=False,
        excluded_ids=(),
        excluded_keys=(),
    )
    async with postgres_sessions.begin() as session:
        repo = VacancyRepository(session)
        for card in cards:
            await repo.save(card)
    async with postgres_sessions() as session:
        for preferences, excluded in PROFILES:
            profile = CandidateProfile(
                skills={"Python": {}}, experience_years=3, preferences=copy.deepcopy(preferences)
            )
            expected = set(CASES) - excluded
            assert {
                v.source_id for v in cards if check_eligibility(v, profile, now=now).eligible
            } == expected
            assert {
                r["source_id"] for r in memory.rows if compact_available(r, profile, options)
            } == expected
            sql_ids = set(
                await session.scalars(
                    select(VacancyModel.source_id).where(
                        *CandidateRepository(session).predicates(profile, options)
                    )
                )
            )
            assert sql_ids == expected
            for source in (None, memory):
                result = await retrieve_candidates(
                    VacancyRepository(session),
                    profile,
                    now=now,
                    parameters=RetrievalParameters(k=64),
                    candidate_source=source,
                )
                assert {v.source_id for v in result.items} == expected
