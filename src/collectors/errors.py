"""Errors shared by source adapters and collection workers."""


class CollectionStopped(Exception):
    """A block, captcha, or unexpected search page pauses collection for this publisher."""

    CODES = frozenset(
        {
            "collection_stopped",
            "collection_cancelled",
            "hh_http_403",
            "hh_http_429",
            "hh_captcha",
            "hh_access_restricted",
            "hh_unexpected_search",
            "hh_parse_limit",
        }
    )

    def __init__(self, message="Сбор остановлен", *, code="collection_stopped", status_code=None):
        super().__init__(message)
        self.code = code if type(code) is str and code in self.CODES else "collection_stopped"
        self.status_code = (
            status_code if type(status_code) is int and status_code in {403, 429} else None
        )
        self.stage = None
        self.source_vacancy_id = None


class VacancyParseError(ValueError):
    """Unexpected source data that cannot be mapped to a vacancy."""
