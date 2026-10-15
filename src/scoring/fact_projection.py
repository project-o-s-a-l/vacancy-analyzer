"""Lexical projection consumes persisted facts without invoking text extraction again."""

from domain.professional import digest
from nlp.professional_facts import vacancy_fact_input_revision
from scoring.retrieval import tokens

PROJECTION_VERSION = "fact-lexical-v2"


def projection_contract():
    from domain.contracts import facts_contract, normalization_contract

    return digest(
        dict(
            version=PROJECTION_VERSION,
            normalization=normalization_contract(),
            facts=facts_contract(),
        )
    )


def projection_revision(vacancy, facts):
    return digest(
        dict(
            version=PROJECTION_VERSION,
            contract=projection_contract(),
            source=vacancy_fact_input_revision(vacancy),
            facts=facts.revision,
        )
    )


def fact_projection(vacancy, facts):
    # Preserve lexical-v1 wire shape/weights. Independent projection identity binds facts.
    from domain.retrieval import RETRIEVAL_VERSION
    from nlp.vacancy_document import professional_text

    names = [f.name for f in facts.facts if f.kind == "skill" and f.assertion.status == "present"]
    work = [
        f.name
        for f in facts.facts
        if f.kind in {"role", "task"} and f.assertion.status == "present"
    ]
    return dict(
        version=RETRIEVAL_VERSION,
        title=sorted(tokens(vacancy.title)),
        skills=sorted({term for name in names for term in tokens(name)}),
        body=sorted(
            tokens(
                "\n".join(
                    [professional_text(vacancy.description or vacancy.description_html), *work]
                )
            )
        ),
    )
