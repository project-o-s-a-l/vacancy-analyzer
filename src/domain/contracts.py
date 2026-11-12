"""Explicit dependency boundaries, schemas and semantics, independent of UI/project inventory."""

import hashlib
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

from domain.professional import FACT_POLICY, FACT_SCHEMA, PROFILE_SCHEMA, digest

SOURCE_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=32)
def source_contract(paths: tuple[str, ...]):
    """Changes take effect after application restart, alongside the loaded Python code."""
    return {path: hashlib.sha256((SOURCE_ROOT / path).read_bytes()).hexdigest() for path in paths}


@lru_cache(maxsize=1)
def normalization_contract():
    return digest(
        dict(
            version="professional-normalization-v1",
            semantics="NFKC-casefold-yo-lemma-skill-aliases-title-aliases-clean-html",
            dependencies=source_contract(
                (
                    "nlp/text.py",
                    "nlp/skills.py",
                    "nlp/extraction.py",
                    "nlp/titles.py",
                    "processing/cleaner.py",
                )
            ),
            morphology={name: version(name) for name in ("pymorphy3", "pymorphy3-dicts-ru")},
        )
    )


@lru_cache(maxsize=1)
def facts_contract():
    return digest(
        dict(
            schema=FACT_SCHEMA,
            profile=PROFILE_SCHEMA,
            policy=FACT_POLICY,
            normalization=normalization_contract(),
            dependencies=source_contract(
                (
                    "nlp/professional_facts.py",
                    "nlp/requirements.py",
                    "nlp/alternatives.py",
                    "nlp/vacancy_requirements.py",
                    "nlp/vacancy_document.py",
                    "nlp/vacancy_facts.py",
                    "nlp/resume.py",
                    "nlp/resume_preferences.py",
                    "nlp/resume_document.py",
                    "nlp/resume_facts.py",
                    "nlp/qualifications.py",
                    "nlp/experience_requirement.py",
                    "nlp/experience.py",
                    "nlp/ollama.py",
                    "nlp/ollama_input.py",
                    "nlp/ollama_evidence.py",
                    "domain/professional.py",
                )
            ),
        )
    )


def feature_dependencies():
    return dict(
        normalization=normalization_contract(),
        facts=facts_contract(),
        rules=source_contract(
            (
                "scoring/serving_features.py",
                "scoring/baseline.py",
                "scoring/skill_score.py",
                "scoring/eligibility.py",
                "scoring/comparison.py",
                "domain/comparison.py",
                "scoring/linear_ranker.py",
                "scoring/retrieval.py",
                "scoring/fact_projection.py",
                "nlp/qualifications.py",
                "nlp/experience_requirement.py",
                "nlp/experience.py",
            )
        ),
    )
