import pytest
from sqlalchemy.exc import SQLAlchemyError

from jobs.users import UserService
from storage.models import LearningAccess, LearningArtifact, SelectionObservation


@pytest.mark.postgres
@pytest.mark.parametrize("model", [SelectionObservation, LearningAccess, LearningArtifact])
async def test_readiness_rejects_incomplete_f13_migration(postgres_sessions, model):
    engine = postgres_sessions.kw["bind"]
    async with engine.begin() as connection:
        await connection.run_sync(model.__table__.drop)
    with pytest.raises(SQLAlchemyError):
        await UserService(postgres_sessions).check_ready()
    async with engine.begin() as connection:
        await connection.run_sync(model.__table__.create)
    await UserService(postgres_sessions).check_ready()
