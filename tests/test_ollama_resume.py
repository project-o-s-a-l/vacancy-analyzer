import asyncio
import json

import httpx
import pytest

from jobs.resume_analysis import resume_analyzer
from jobs.settings import Settings
from nlp.ollama_resume import OllamaResumeAnalyzer, ResumeAnalysisError
from nlp.ollama_slot import model_slot

SUMMARY = "backend python developer. опыт работы 0 лет. студент"
RESULT = {
    "skills": [{"name": "Python", "evidence": "d1"}],
    "positions": [{"name": "backend python developer", "evidence": "d1"}],
    "experience_evidence": "d2",
}


def transport(result, requests, *, done_reason="stop"):
    def handle(request):
        requests.append(request)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3:4b", "digest": "test"}]})
        return httpx.Response(
            200,
            json={
                "done": True,
                "done_reason": done_reason,
                "message": {"content": json.dumps(result)},
            },
        )

    return httpx.MockTransport(handle)


async def test_summary_separates_skill_role_experience_and_status_with_one_request():
    requests = []
    async with httpx.AsyncClient(
        base_url="http://localhost", transport=transport(RESULT, requests)
    ) as client:
        parsed = await OllamaResumeAnalyzer(client, model="qwen3:4b").analyze(SUMMARY)
    assert parsed.skills == ["python"]
    assert parsed.positions == ["backend python developer"]
    assert parsed.experience_years == 0
    assert parsed.notes == []
    assert [request.url.path for request in requests] == ["/api/tags", "/api/chat"]
    body = json.loads(requests[-1].content)
    assert body["keep_alive"] == 0 and body["stream"] is False and body["think"] is False
    assert body["options"] == {
        "temperature": 0,
        "num_ctx": 8192,
        "num_predict": 1024,
        "num_thread": 2,
    }
    assert json.loads(body["messages"][1]["content"])["fragments"]["d2"] == "опыт работы 0 лет."


async def test_invented_skills_and_copied_metadata_are_not_saved_as_skills():
    payload = {
        **RESULT,
        "skills": RESULT["skills"]
        + [
            {"name": "Docker", "evidence": "d1"},
            {"name": "backend python developer", "evidence": "d1"},
            {"name": "студент", "evidence": "d3"},
            {"name": "опыт работы 0 лет", "evidence": "d2"},
        ],
    }
    async with httpx.AsyncClient(
        base_url="http://localhost", transport=transport(payload, [])
    ) as client:
        parsed = await OllamaResumeAnalyzer(client, model="qwen3:4b").analyze(SUMMARY)
    assert parsed.skills == ["python"]


async def test_profession_does_not_invent_skills_and_student_does_not_imply_zero():
    result = {
        "skills": [{"name": "Excel", "evidence": "d1"}, {"name": "Бухгалтер", "evidence": "d1"}],
        "positions": [{"name": "Бухгалтер", "evidence": "d1"}],
        "experience_evidence": "d2",
    }
    async with httpx.AsyncClient(
        base_url="http://localhost", transport=transport(result, [])
    ) as client:
        parsed = await OllamaResumeAnalyzer(client, model="qwen3:4b").analyze(
            "Бухгалтер. Студент, ищу первую работу."
        )
    assert parsed.skills == [] and parsed.positions == ["Бухгалтер"]
    assert parsed.experience_years is None


async def test_unknown_professional_skill_can_be_extracted_without_technology_whitelist():
    result = {
        "skills": [{"name": "Перевозка опасных грузов", "evidence": "d2"}],
        "positions": [{"name": "Водитель", "evidence": "d1"}],
        "experience_evidence": None,
    }
    async with httpx.AsyncClient(
        base_url="http://localhost", transport=transport(result, [])
    ) as client:
        parsed = await OllamaResumeAnalyzer(client, model="qwen3:4b").analyze(
            "Водитель. Перевозка опасных грузов."
        )
    assert parsed.skills == ["перевозка опасных грузов"] and parsed.experience_years is None


@pytest.mark.parametrize("mutation", ["reference", "schema", "truncated"])
async def test_invalid_or_incomplete_response_is_rejected_without_raw_data(mutation):
    result = {**RESULT}
    if mutation == "reference":
        result["experience_evidence"] = "d9999"
    elif mutation == "schema":
        result["private_data"] = "private diagnostics"
    requests = []
    async with httpx.AsyncClient(
        base_url="http://localhost",
        transport=transport(
            result, requests, done_reason="length" if mutation == "truncated" else "stop"
        ),
    ) as client:
        with pytest.raises(ResumeAnalysisError) as error:
            await OllamaResumeAnalyzer(client, model="qwen3:4b").analyze(SUMMARY)
    assert "private diagnostics" not in str(error.value) and SUMMARY not in str(error.value)
    assert len(requests) == 2


async def test_oversized_resume_does_not_call_model_or_truncate_source():
    requests = []
    async with httpx.AsyncClient(
        base_url="http://localhost", transport=transport(RESULT, requests)
    ) as client:
        with pytest.raises(ResumeAnalysisError, match="не помещается"):
            await OllamaResumeAnalyzer(client, model="qwen3:4b").analyze("x" * 10000)
    assert requests == []


async def test_disabled_resume_analyzer_creates_no_client():
    async with resume_analyzer(Settings(_env_file=None)) as analyzer:
        assert analyzer is None


async def test_resume_and_vacancy_share_one_model_slot():
    entered = asyncio.Event()
    release = asyncio.Event()
    second_entered = asyncio.Event()

    async def first():
        async with model_slot():
            entered.set()
            await release.wait()

    async def second():
        async with model_slot():
            second_entered.set()

    one = asyncio.create_task(first())
    await entered.wait()
    two = asyncio.create_task(second())
    await asyncio.sleep(0)
    assert not second_entered.is_set()
    release.set()
    await asyncio.gather(one, two)
    assert second_entered.is_set()
