"""Small fixed text features, identical for offline fitting and live prediction."""

import hashlib
import json
from collections import Counter
from functools import lru_cache
from math import sqrt
from pathlib import Path

from domain.matching_dataset import PairFeatures
from domain.retrieval import RetrievalParameters
from nlp.text import word_tokens
from processing.cleaner import clean_html
from scoring.linear_ranker import FEATURE_NAMES
from scoring.text_features import FIXED_TEXT_FEATURES_VERSION

FEATURE_SPEC = {
    "version": FIXED_TEXT_FEATURES_VERSION,
    "query_scope": "confirmed_profile_fields",
    "word": "term_frequency_cosine",
    "char": "3_to_5_gram_cosine",
    "max_job_characters": 20000,
    "max_query_characters": 8000,
    "fit_scope": "none",
    "eligibility": "full_text_before_features",
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def profile_revision(profile):
    return digest(profile.model_dump(mode="json"))


@lru_cache(maxsize=1)
def feature_contract():
    root = Path(__file__).parents[1]
    # Conservative compatibility: changes to normalization, admissibility,
    # component calculation or schemas require evaluation of a new artifact.
    files = [
        p
        for module in ("domain", "scoring", "nlp", "processing")
        for p in (root / module).glob("*.py")
    ]
    return digest(
        {
            "spec": FEATURE_SPEC,
            "features": FEATURE_NAMES,
            "retrieval": RetrievalParameters().model_dump(),
            "sources": {
                p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(files)
            },
        }
    )


def _cosine(first, second):
    if not first or not second:
        return None
    denominator = sqrt(sum(v * v for v in first.values()) * sum(v * v for v in second.values()))
    return min(
        1.0, max(0.0, sum(value * second.get(key, 0) for key, value in first.items()) / denominator)
    )


def _characters(text):
    return Counter(
        text[i : i + size] for size in (3, 4, 5) for i in range(max(0, len(text) - size + 1))
    )


def pair_features(vacancy, profile, match):
    query = "\n".join([*profile.preferences.positions, *profile.skills])[
        : FEATURE_SPEC["max_query_characters"]
    ]
    text = "\n".join(
        [
            vacancy.title,
            *vacancy.skills,
            clean_html(vacancy.description or vacancy.description_html),
        ]
    )[: FEATURE_SPEC["max_job_characters"]]
    query_tokens, document_tokens = word_tokens(query), word_tokens(text)
    return PairFeatures(
        baseline_score=match.score,
        baseline_coverage=match.evidence_coverage,
        baseline_components=match.components,
        word_cosine=_cosine(Counter(query_tokens), Counter(document_tokens)),
        char_cosine=_cosine(
            _characters(" ".join(query_tokens)), _characters(" ".join(document_tokens))
        ),
        matched_skills=len(match.matched_skills),
        missing_required=len(match.missing_required),
        missing_preferred=len(match.missing_preferred),
        missing_unspecified=len(match.missing_unspecified),
    )


def shown_context(vacancy, profile, match, rank):
    features = pair_features(vacancy, profile, match)
    return {
        "version": "ranking-exposure-v1",
        "feature_contract": feature_contract(),
        "feature_spec": FEATURE_SPEC,
        "profile_revision": profile_revision(profile),
        "features": features.model_dump(mode="json"),
        "display_rank": rank,
        "decision": match.ranking.model_dump(mode="json") if match.ranking else None,
    }
