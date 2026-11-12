"""Durable optimistic editor for active profiles and exact resume proposals."""

from dataclasses import dataclass, replace
from uuid import uuid4

from domain import CandidateProfile
from domain.professional import digest
from jobs.profile_details import change
from jobs.resumes import ResumeService, prepared_profile, profile_hash
from storage.models import ProfileEditModel, ResumeModel
from storage.repositories.user_repository import UserRepository


@dataclass(frozen=True)
class EditSession:
    token: str
    base_revision: int
    profile: CandidateProfile
    view: list
    target: str
    resume_token: str | None
    dirty: bool
    base_profile_hash: str


def snapshot_hash(profile):
    from nlp.professional_facts import profile_input_revision

    return digest([profile_input_revision(profile), profile.professional_facts.revision])


def read_edit(row):
    return EditSession(
        row.token,
        row.base_revision,
        CandidateProfile.model_validate(row.payload["profile"]),
        row.payload["view"],
        row.payload["target"],
        row.payload.get("resume_token"),
        row.payload.get("dirty", False),
        row.payload["base_profile_hash"],
    )


class ProfileEditor:
    def __init__(self, sessions, *, enabled=True):
        self.sessions = sessions
        self.enabled = enabled

    def _enabled(self):
        if not self.enabled:
            raise ValueError("Редактор временно отключён. Сохранённый профиль доступен в /profile.")

    async def get(self, owner):
        async with self.sessions() as session:
            row = await session.get(ProfileEditModel, owner)
            return read_edit(row) if row else None

    async def open(self, owner, *, view=None, resume_token=None):
        self._enabled()
        async with self.sessions.begin() as session:
            user = await UserRepository(session).get(owner, lock=True)
            if user is None:
                raise ValueError("Начните с /start.")
            row = await session.get(ProfileEditModel, owner)
            if row:
                edit = read_edit(row)
                await self._current(session, user, edit)
                if resume_token is not None and edit.resume_token != resume_token:
                    raise ValueError(
                        "Сначала сохраните или отмените текущую правку через /profile."
                    )
                return edit
            profile = user.profile
            if resume_token:
                draft = ResumeService._draft(await session.get(ResumeModel, owner), resume_token)
                self._resume_current(user, draft)
                profile = prepared_profile(profile, draft, owner)
            row = ProfileEditModel(
                user_id=owner,
                token=uuid4().hex,
                base_revision=user.revision,
                payload=dict(
                    profile=profile.model_dump(mode="json"),
                    view=view or ["home"],
                    target="resume" if resume_token else "profile",
                    resume_token=resume_token,
                    dirty=False,
                    base_profile_hash=snapshot_hash(user.profile),
                ),
            )
            session.add(row)
            return read_edit(row)

    @staticmethod
    def _resume_current(user, draft):
        from domain.contracts import facts_contract
        from domain.resume import RESUME_VERSION

        if draft.profile_revision != user.revision or draft.profile_hash != profile_hash(
            user.profile
        ):
            raise ValueError("Профиль изменился; загрузите резюме заново.")
        if draft.version not in {RESUME_VERSION, RESUME_VERSION + "-ollama"} or (
            draft.professional_facts
            and draft.professional_facts.extractor_contract != facts_contract()
        ):
            raise ValueError("Обработчик обновлён; загрузите резюме заново.")

    async def _current(self, session, user, edit):
        if (
            user is None
            or user.revision != edit.base_revision
            or snapshot_hash(user.profile) != edit.base_profile_hash
        ):
            raise ValueError(
                "Профиль изменился. /cancel отменит эту правку; затем откройте /profile."
            )
        if edit.target == "resume":
            draft = ResumeService._draft(
                await session.get(ResumeModel, user.user_id), edit.resume_token
            )
            self._resume_current(user, draft)

    async def cancel(self, owner):
        async with self.sessions.begin() as session:
            await UserRepository(session).get(owner, lock=True)
            row = await session.get(ProfileEditModel, owner)
            if row:
                await session.delete(row)
            return row is not None

    async def act(self, owner, token, action, *, text=None):
        """Token is consumed once per step, including navigation and confirmation."""
        self._enabled()
        async with self.sessions.begin() as session:
            repository = UserRepository(session)
            user = await repository.get(owner, lock=True)
            row = await session.get(ProfileEditModel, owner)
            if row is None or row.token != token:
                raise ValueError("Кнопка устарела. Откройте /profile или /resume.")
            edit = read_edit(row)
            await self._current(session, user, edit)
            payload = dict(row.payload)
            op, *args = action
            if op == "cancel":
                await session.delete(row)
                return None
            if op == "save":
                if edit.target == "profile":
                    await repository.save(replace(user, profile=edit.profile, form_state=None))
                else:
                    resume = await session.get(ResumeModel, owner)
                    draft = ResumeService._draft(resume, edit.resume_token)
                    draft.reviewed_profile = edit.profile
                    draft.token = uuid4().hex
                    resume.draft = draft.model_dump(mode="json")
                await session.delete(row)
                return None
            if op == "view":
                payload["view"] = args
            elif op == "set" or op == "text":
                if op == "text":
                    if edit.view[0] != "input":
                        raise ValueError("Выберите поле кнопкой.")
                    kind, key, field = edit.view[1:]
                    value = text
                else:
                    kind, key, field, value = args
                profile, new_key = change(edit.profile, owner, kind, key, field, value)
                payload.update(profile=profile.model_dump(mode="json"), dirty=True)
                if field == "delete":
                    payload["view"] = ["list", kind, 0]
                elif kind == "basic":
                    payload["view"] = ["preferences"]
                else:
                    payload["view"] = ["item", kind, new_key or key]
            else:
                raise ValueError("Неизвестное действие")
            row.payload, row.token = payload, uuid4().hex
            return read_edit(row)
