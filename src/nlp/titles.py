"""Optional title aliases supplement generic matching; unknown professions remain valid."""

from nlp.text import word_tokens

# This small compatibility dictionary is an enhancement, not a supported-professions list.
_ALIASES = {
    "разработчик": "developer",
    "разработчица": "developer",
    "программист": "developer",
    "programmer": "developer",
}


def title_tokens(value: str) -> set[str]:
    return {_ALIASES.get(token, token) for token in word_tokens(value)}
