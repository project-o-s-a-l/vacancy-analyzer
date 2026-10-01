"""Recipe identity includes prompt, schemas, deterministic rules and implementation."""

import hashlib
import json
from pathlib import Path


def pipeline_hash(kind: str, model: str) -> str:
    from domain.resume import RESUME_VERSION, ResumeExtraction
    from domain.vacancy_analysis import ANALYSIS_VERSION, VacancyExtraction
    from nlp.ollama import SYSTEM_PROMPT
    from nlp.ollama_resume import RESUME_SYSTEM_PROMPT

    root = Path(__file__).parent
    files = [
        "analysis_identity.py",
        "ollama.py",
        "ollama_resume.py",
        "ollama_input.py",
        "ollama_evidence.py",
        "experience.py",
        "resume.py",
        "requirements.py",
        "qualifications.py",
    ]
    value = {
        "kind": kind,
        "model": model,
        "schema": ANALYSIS_VERSION if kind == "vacancy_analysis" else RESUME_VERSION,
        "prompt": SYSTEM_PROMPT if kind == "vacancy_analysis" else RESUME_SYSTEM_PROMPT,
        "extraction": (
            VacancyExtraction if kind == "vacancy_analysis" else ResumeExtraction
        ).model_json_schema(),
        "implementation": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files
        },
        "cleaner": hashlib.sha256((root.parent / "processing/cleaner.py").read_bytes()).hexdigest(),
        "input_domain": {
            name: hashlib.sha256((root.parent / name).read_bytes()).hexdigest()
            for name in ("domain/vacancy_analysis.py", "domain/resume.py", "processing/resume.py")
        },
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def current_vacancy_analysis(vacancy):
    """UI and ranking can fall back to rules without contacting a model service."""
    from domain.vacancy_analysis import current_analysis

    analysis = current_analysis(vacancy)
    if analysis and analysis.pipeline_hash == pipeline_hash("vacancy_analysis", analysis.model):
        return analysis
    return None
