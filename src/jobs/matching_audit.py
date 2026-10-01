"""Offline acceptance audit: fictitious resumes against a frozen real catalog."""

import asyncio
import json
import logging
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from domain import Vacancy
from domain.matching import MatchResult
from domain.resume import ResumeExtraction
from jobs.matching_dataset import build_matching_dataset, write_matching_dataset
from jobs.users import UserService
from observability import emit_event
from processing.cleaner import clean_html
from processing.resume import read_resume
from scoring import rank_vacancies
from storage.database import create_engine, session_factory
from storage.models import Base
from storage.repositories import VacancyRepository


class AuditCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9-]{1,64}$")
    profession: str = Field(pattern=r"^[a-z0-9-]{1,64}$")
    split: Literal["training", "validation"]
    text: str = Field(min_length=20, max_length=60_000)
    expected: ResumeExtraction


class AuditCases(BaseModel):
    model_config = ConfigDict(extra="forbid")

    synthetic_resumes: Literal[True]
    cases: list[AuditCase] = Field(min_length=1)

    @model_validator(mode="after")
    def distinct_cases(self):
        if len({case.id for case in self.cases}) != len(self.cases):
            raise ValueError("Идентификаторы учебных резюме должны быть уникальны.")
        return self


def load_audit_cases(path: Path) -> AuditCases:
    return AuditCases.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def catalog_snapshot(vacancies: list[Vacancy], *, now: datetime) -> dict:
    return {
        "version": "multirole-catalog-v1",
        "captured_at": now.isoformat(),
        "vacancy_source": "local_database",
        "vacancies": [
            vacancy.model_copy(
                update={
                    "description": clean_html(vacancy.description or vacancy.description_html),
                    "description_html": None,
                    "raw_payload": {},
                }
            ).model_dump(mode="json")
            for vacancy in vacancies
        ],
    }


class LocalSender:
    """A local transport for the audit DB; no Telegram connection or real messages."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_recommendation(self, chat_id: int, vacancy: Vacancy, match: MatchResult) -> int:
        self.sent.append(vacancy.source_id)
        return len(self.sent)


async def audit_cases(
    specification: AuditCases,
    snapshot: dict,
    output: Path,
    *,
    min_score: float = 50,
    top: int = 5,
    features: bool = False,
) -> dict:
    if snapshot.get("version") != "multirole-catalog-v1":
        raise ValueError("Неизвестная версия снимка каталога.")
    if not 0 <= min_score <= 100 or not 1 <= top <= 20:
        raise ValueError("Некорректные параметры выдачи.")
    now = datetime.fromisoformat(snapshot["captured_at"])
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("Снимку каталога нужен часовой пояс.")
    vacancies = [Vacancy.model_validate(value) for value in snapshot["vacancies"]]
    if len({vacancy.id for vacancy in vacancies}) != len(vacancies):
        raise ValueError("Снимок содержит повторные ID вакансий.")
    await asyncio.to_thread(output.mkdir, parents=True, exist_ok=False)
    snapshot_bytes = (json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n").encode()
    (output / "corpus.json").write_bytes(snapshot_bytes)
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    results = []
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        sessions = session_factory(engine)
        async with sessions.begin() as session:
            for vacancy in vacancies:
                await VacancyRepository(session).save(vacancy)
        users = UserService(sessions, top=1, min_score=min_score)
        for user_id, case in enumerate(specification.cases, start=1):
            directory = output / case.id
            directory.mkdir()
            (directory / "resume.txt").write_text(case.text, encoding="utf-8")
            text, kind = read_resume(case.text.encode("utf-8"), "resume.txt")
            await users.register(user_id)
            before = await users.get(user_id)
            draft = await users.resumes.prepare(user_id, text, format=kind)
            unchanged = (await users.get(user_id)).profile == before.profile
            user = await users.resumes.confirm(user_id, draft.token)
            expected = case.expected.model_dump(exclude={"notes"})
            actual = draft.extraction.model_dump(exclude={"notes"})
            checks = {key: actual[key] == expected[key] for key in expected}
            ranking = rank_vacancies(vacancies, user.profile, now=now)
            selected = [(vacancy, match) for vacancy, match in ranking if match.score >= min_score]
            operational = {"draft_preserved_profile": unchanged, "delivery_and_feedback": None}
            notification = await users.notifications(user_id, require_profile=False)
            sender = LocalSender()
            delivery = await notification.deliver(sender)
            if delivery.sent:
                _, history = await users.feedback.page(user_id, "history")
                vacancy, _ = history[0]
                await users.feedback.record(
                    user_id, vacancy.id, "save", event_key=case.id + ":save"
                )
                await users.feedback.record(
                    user_id, vacancy.id, "like", event_key=case.id + ":like"
                )
                await users.feedback.record(
                    user_id, vacancy.id, "apply", event_key=case.id + ":apply"
                )
                _, state = await users.feedback.card(user_id, vacancy.id)
                operational["delivery_and_feedback"] = (
                    state.saved and state.applied and state.interest == "like"
                )
            item = {
                "id": case.id,
                "profession": case.profession,
                "split": case.split,
                "resume_parser_version": draft.version,
                "extraction": draft.extraction.model_dump(),
                "expected_extraction": expected,
                "extraction_checks": checks,
                "confirmed_profile": user.profile.model_dump(mode="json"),
                "operational_checks": operational,
                "eligible_vacancies": len(ranking),
                "above_threshold": len(selected),
                "local_transport_messages": sender.sent,
                "top": [
                    {
                        "id": vacancy.source_id,
                        "title": vacancy.title,
                        "score": match.score,
                        "components": match.components,
                        "matched_skills": match.matched_skills,
                        "warnings": match.warnings,
                    }
                    for vacancy, match in ranking[:top]
                ],
            }
            (directory / "audit.json").write_text(
                json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            if features and ranking and (user.profile.skills or user.profile.preferences.positions):
                dataset = build_matching_dataset(vacancies, user.profile, resume_text=text, now=now)
                write_matching_dataset(dataset, directory / "matching")
            results.append(item)
            emit_event(logging.getLogger(__name__), "experiment.progress", count=len(selected))
    finally:
        await engine.dispose()
    report = {
        "resume_source": "synthetic",
        "vacancy_source": "frozen_real_catalog",
        "reference_time": now.isoformat(),
        "corpus_sha256": sha256(snapshot_bytes).hexdigest(),
        "catalog_count": len(vacancies),
        "threshold": min_score,
        "top": top,
        "cases": results,
        "live_telegram_messages_sent": False,
        "working_database_modified": False,
        "independent_quality_evaluation": False,
        "limitations": [
            "Учебные резюме вымышлены; реальные пользовательские данные не использовались.",
            "Оценки соответствия алгоритма не являются эталонной разметкой.",
            "Отсутствие подходящих вакансий может означать пробелы каталога или алгоритма.",
            "Ранжирование заморожено по времени снимка; служба доставки использует текущее время.",
            "Доставка и feedback проверяются только в изолированной SQLite с местным транспортом.",
        ],
    }
    (output / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
