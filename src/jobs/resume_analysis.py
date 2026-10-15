"""Optional local resume analyzer for Telegram and future application hosts."""

from contextlib import asynccontextmanager

import httpx

from jobs.model_telemetry import ObservedOllamaClient
from jobs.settings import Settings
from nlp.ollama_resume import OllamaResumeAnalyzer


@asynccontextmanager
async def resume_analyzer(settings: Settings):
    if not settings.resume_llm_enabled:
        yield None
        return
    async with ObservedOllamaClient(
        model=settings.ollama_model,
        job_kind="resume_analysis",
        base_url=settings.ollama_base_url,
        timeout=httpx.Timeout(settings.ollama_timeout, connect=5),
        trust_env=False,
        follow_redirects=False,
    ) as client:
        yield OllamaResumeAnalyzer(
            client,
            model=settings.ollama_model,
            total_seconds=settings.resume_total_timeout,
            request_seconds=settings.ollama_timeout,
        )
