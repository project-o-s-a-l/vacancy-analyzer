"""Recipe identity includes prompt, schemas, deterministic rules and implementation."""

from domain.contracts import normalization_contract, source_contract
from domain.professional import digest


def pipeline_hash(kind: str, model: str) -> str:
    from domain.resume import RESUME_VERSION, ResumeExtraction
    from domain.vacancy_analysis import ANALYSIS_VERSION, VacancyExtraction
    from nlp.ollama import SYSTEM_PROMPT
    from nlp.ollama_resume import RESUME_SYSTEM_PROMPT

    if kind not in {"vacancy_analysis", "resume_analysis"}:
        raise ValueError("Unknown extraction pipeline")
    shared = ("nlp/ollama_input.py", "nlp/ollama_evidence.py", "nlp/experience.py")
    specific = (
        ("nlp/ollama.py", "nlp/requirements.py", "nlp/qualifications.py", "nlp/vacancy_document.py")
        if kind == "vacancy_analysis"
        else (
            "nlp/ollama_resume.py",
            "nlp/resume.py",
            "nlp/resume_document.py",
            "nlp/resume_facts.py",
            "processing/resume.py",
        )
    )
    value = {
        "kind": kind,
        "model": model,
        "schema": ANALYSIS_VERSION if kind == "vacancy_analysis" else RESUME_VERSION,
        "prompt": SYSTEM_PROMPT if kind == "vacancy_analysis" else RESUME_SYSTEM_PROMPT,
        "extraction": (
            VacancyExtraction if kind == "vacancy_analysis" else ResumeExtraction
        ).model_json_schema(),
        "normalization": normalization_contract(),
        "input_contract": "vacancy-professional-sections-v2"
        if kind == "vacancy_analysis"
        else "normalized-resume-full-v1",
        "implementation": source_contract(shared + specific),
    }
    return digest(value)


def current_vacancy_analysis(vacancy):
    """UI and ranking can fall back to rules without contacting a model service."""
    from domain.vacancy_analysis import current_analysis

    analysis = current_analysis(vacancy)
    if analysis and analysis.pipeline_hash == pipeline_hash("vacancy_analysis", analysis.model):
        return analysis
    return None
