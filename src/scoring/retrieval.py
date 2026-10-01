"""Small corpus-independent lexical features; no fitted or pretrained weights."""

from dataclasses import dataclass
from datetime import UTC

from domain.retrieval import RETRIEVAL_VERSION, RetrievalEvidence
from nlp.extraction import canonical_skill, extract_skills
from nlp.titles import title_tokens
from processing.cleaner import clean_html
from storage.payloads import payload_hash

INDEX_FIELDS = ("title", "skills", "description", "description_html")


def tokens(value):
    # Apply existing skill aliases as well as the profession-independent lemmas.
    return {
        word
        for token in title_tokens(canonical_skill(value))
        for word in title_tokens(canonical_skill(token))
    }


def search_document(values):
    body = clean_html(values.get("description") or values.get("description_html"))
    skills = extract_skills(body, values.get("skills") or [])
    return {
        "version": RETRIEVAL_VERSION,
        "title": sorted(tokens(values.get("title") or "")),
        "skills": sorted({token for skill in skills for token in tokens(skill)}),
        "body": sorted(tokens(body)),
    }


def document_terms(document):
    return sorted(set(document["title"]) | set(document["skills"]) | set(document["body"]))


@dataclass(frozen=True)
class RetrievalQuery:
    titles: tuple[frozenset[str], ...]
    skills: frozenset[str]
    terms: frozenset[str]


def profile_query(profile):
    titles = tuple(frozenset(tokens(position)) for position in profile.preferences.positions)
    skills = frozenset(word for skill in profile.skills for word in tokens(skill))
    return RetrievalQuery(titles, skills, frozenset().union(skills, *titles))


def lexical_evidence(document, query, parameters):
    title, skills, body = (set(document[name]) for name in ("title", "skills", "body"))
    wanted_titles = set().union(*query.titles)
    title_matches = wanted_titles & title
    skill_matches = set(query.skills) & (skills | title)
    body_matches = set(query.terms) & body
    title_score = max(
        (len(wanted & title) / len(wanted) for wanted in query.titles if wanted), default=0
    )
    skill_score = len(skill_matches) / len(query.skills) if query.skills else 0
    body_score = len(body_matches) / len(query.terms) if query.terms else 0
    score = round(
        parameters.title_weight * title_score
        + parameters.skill_weight * skill_score
        + parameters.body_weight * body_score,
        8,
    )
    return RetrievalEvidence(
        parameters_hash=payload_hash(parameters.model_dump()),
        strategy="lexical" if score else "recent_fallback",
        lexical_score=score,
        title_matches=sorted(title_matches),
        skill_matches=sorted(skill_matches),
        body_matches=sorted(body_matches),
    )


def retrieval_eligibility(vacancy, profile, now):
    from nlp.qualifications import compare_qualifications
    from scoring.eligibility import check_eligibility

    text = vacancy.description
    if vacancy.description_html and (
        text is None or clean_html(vacancy.description_html) == clean_html(text)
    ):
        text = vacancy.description_html
    qualifications, _ = compare_qualifications(
        text, profile.qualifications, today=now.astimezone(UTC).date(), explicit=vacancy.skills
    )
    return check_eligibility(vacancy, profile, now=now, qualifications=qualifications)
