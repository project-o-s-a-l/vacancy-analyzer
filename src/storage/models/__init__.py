from sqlalchemy import event

from storage.models.collection import CollectionBudget, CollectionDirection, CollectionObservation
from storage.models.exposure import ExposureModel
from storage.models.feedback import FeedbackEventModel, FeedbackModel
from storage.models.inference import ExtractionCache, InferenceGate
from storage.models.job import BackgroundJob
from storage.models.notification import NotificationModel
from storage.models.professional import FactBackfillModel, FactRevisionModel
from storage.models.resume import ResumeModel, ResumeSourceModel
from storage.models.search_plan import SearchPlanModel
from storage.models.selection import SavedSelectionModel, ServingRevisionModel
from storage.models.user import TelegramUserModel
from storage.models.vacancy import Base, VacancyModel, VacancySnapshot
from storage.retrieval_sql import install_metadata
from storage.serving_revisions import install_metadata as install_serving

event.listen(Base.metadata, "before_create", install_metadata)
event.listen(Base.metadata, "after_create", install_serving)

__all__ = [
    "ExposureModel",
    "ExtractionCache",
    "InferenceGate",
    "CollectionBudget",
    "CollectionDirection",
    "CollectionObservation",
    "Base",
    "FactBackfillModel",
    "FactRevisionModel",
    "BackgroundJob",
    "FeedbackModel",
    "FeedbackEventModel",
    "NotificationModel",
    "ResumeModel",
    "ResumeSourceModel",
    "SearchPlanModel",
    "SavedSelectionModel",
    "ServingRevisionModel",
    "TelegramUserModel",
    "VacancyModel",
    "VacancySnapshot",
]
