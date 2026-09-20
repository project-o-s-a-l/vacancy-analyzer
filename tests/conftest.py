from datetime import UTC, datetime

import pytest

from domain import CandidateProfile, Vacancy


@pytest.fixture
def vacancy():
    return Vacancy(
        source="example",
        source_id="job-1",
        url="https://example.org/jobs/1",
        title="Python Developer",
        company="Example",
        skills=["Python"],
        published_at=datetime(2026, 9, 1, tzinfo=UTC),
        raw_payload={},
    )


@pytest.fixture
def profile():
    return CandidateProfile(skills={"python": {}}, experience_years=3)
