"""Export the schema and synthetic examples. No Settings, database or model calls."""

import json
from datetime import UTC, datetime
from pathlib import Path

from domain import CandidateProfile, Vacancy
from domain.professional import (
    Claim,
    Evidence,
    ExperienceEpisode,
    RoleFact,
    SourceReference,
    TaskFact,
    digest,
)
from domain.professional import ProfessionalFacts as Facts
from nlp.professional_facts import candidate_facts, profile_payload, vacancy_facts


def export(directory):
    directory.mkdir(parents=True, exist_ok=True)
    at = datetime(2026, 11, 2, tzinfo=UTC)
    profile = CandidateProfile(skills={"python": {}}, experience_years=3)
    facts = candidate_facts(
        profile, 17, confirmed_fields={"skills", "experience"}, reference_time=at
    )
    work_source = {"work_history": [{"role": "Инженер", "task": "Обработка данных"}]}
    work_reference = SourceReference(
        id="resume-example:17", revision=digest(work_source), kind="resume"
    )
    evidence = Evidence(
        source_id=work_reference.id, source_revision=work_reference.revision, field="work_history.0"
    )
    role = RoleFact(
        id="role:engineer",
        name="Инженер",
        context="held",
        assertion=Claim[bool](
            status="present", value=True, origin="user_confirmed", evidence=[evidence]
        ),
    )
    task = TaskFact(
        id="task:etl",
        name="Обработка данных",
        assertion=Claim[bool](
            status="present", value=True, origin="user_confirmed", evidence=[evidence]
        ),
    )
    data = facts.model_dump(mode="json")
    data["sources"].append(work_reference.model_dump(mode="json"))
    data["facts"] += [role.model_dump(mode="json"), task.model_dump(mode="json")]
    skill = next(f for f in facts.facts if f.kind == "skill")
    data["episodes"] = [
        ExperienceEpisode(
            id="project:1",
            kind="project",
            role_ids=[role.id],
            skill_ids=[skill.id],
            task_ids=[task.id],
            evidence=[evidence],
        ).model_dump(mode="json")
    ]
    data.update(extracted_at=at.isoformat(), reference_time=at.isoformat())
    candidate = Facts.model_validate(data)
    vacancy = Vacancy(
        id="00000000-0000-0000-0000-000000000001",
        source="synthetic",
        source_id="f02-example",
        url="https://example.org/job",
        title="Инженер данных",
        company="Вымышленная компания",
        description="Требования:\nPython или Java\nУсловия:\nУдалённая работа",
        skills=["Python", "Java"],
        experience="1-3",
        remote=True,
        published_at=at,
        collected_at=at,
        raw_payload={"kind": "authored synthetic example"},
    )
    job_facts = vacancy_facts(vacancy).model_copy(update={"extracted_at": at})
    outputs = {
        "professional-facts-v1.schema.json": Facts.model_json_schema(),
        "candidate-source.example.json": work_source,
        "candidate-profile.example.json": profile_payload(profile),
        "vacancy-source.example.json": vacancy.model_dump(
            mode="json", exclude={"professional_facts"}
        ),
        "candidate-facts.example.json": candidate.model_dump(mode="json"),
        "vacancy-facts.example.json": job_facts.model_dump(mode="json"),
    }
    for name, payload in outputs.items():
        (directory / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    export(Path(__file__).resolve().parents[1] / "docs/contracts")
