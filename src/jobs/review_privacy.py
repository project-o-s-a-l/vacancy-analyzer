"""Fail-closed obvious-contact guard plus explicit review of real free text.

Pattern scanning is not anonymization and cannot detect names or indirect identifiers.
"""

import re

from jobs.evaluation_data import digest

CONTACT = re.compile(
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?:\+\d[\d ()-]{8,}\d)|"
    r"(?:t\.me/|telegram\.me/|wa\.me/|mailto:)|(?:телефон|phone)\s*[:=]",
    re.IGNORECASE,
)


def check_review_privacy(cards, *, real_data, review=None):
    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for key, child in value.items():
                yield key
                yield from strings(child)
        elif isinstance(value, list):
            for child in value:
                yield from strings(child)

    if any(CONTACT.search(value) for value in strings(cards)):
        raise ValueError("Contact detected; prepare a separately reviewed source bundle")
    if real_data and (
        not isinstance(review, dict)
        or review.get("tasks_hash") != digest(cards)
        or review.get("attestation") != "I reviewed all free text for personal data and contacts"
        or not review.get("reviewer")
        or not review.get("permission_ref")
    ):
        raise ValueError("Real free text requires explicit privacy review of this exact tasks hash")
