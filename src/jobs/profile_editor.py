"""Durable optimistic editor for active profiles and exact resume proposals."""

from dataclasses import dataclass, replace
from uuid import uuid4

from domain import CandidateProfile
from domain.professional import digest
from jobs.profile_details import change
from jobs.resumes import ResumeService, prepared_profile, profile_hash, release_resume_editor
from nlp.profile_experience import sync_profile_experience
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


def snapshot_hash(user):
    return digest([user.input_revision, user.facts_revision])


def read_edit(row):
    view = row.payload["view"]
    if view[0] == "parents":
        view = ["item", "episode", view[1]]
    elif view[0] == "input" and (
        (view[1] == "episode" and view[3] in {"organization", "title", "parent", "skills"})
        or (view[1] == "skill" and view[3] in {"years", "level", "last_used"})
    ):
        view = ["item", view[1], view[2]]
    return EditSession(
        row.token,
        row.base_revision,
        sync_profile_experience(CandidateProfile.model_validate(row.payload["profile"])),
        view,
        row.payload["target"],
        row.payload.get("resume_token"),
        row.payload.get("dirty", False),
        row.payload["base_profile_hash"],
    )


def section_for(field):
    if field == "skills":
        return "home"
    if field == "qualifications":
        return "documents"
    if field == "experience":
        return "career"
    return "preferences"


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

    async def open(self, owner, *, view=None, resume_token=None, discard=None):
        self._enabled()
        async with self.sessions.begin() as session:
            user = await UserRepository(session).get(owner, lock=True)
            if user is None:
                raise ValueError("Начните с /start.")
            row = await session.get(ProfileEditModel, owner)
            if resume_token and (row is None or row.payload.get("resume_token") != resume_token):
                await release_resume_editor(session, row, resume_token, discard)
                row = None
            if row and not row.payload.get("dirty"):
                opening_profile = view is not None and resume_token is None
                base_changed = row.base_revision != user.revision or (
                    row.payload["base_profile_hash"] != snapshot_hash(user)
                )
                if base_changed or (opening_profile and row.payload["target"] == "resume"):
                    await session.delete(row)
                    await session.flush()
                    row = None
            if row:
                edit = read_edit(row)
                await self._current(session, user, edit)
                if (
                    view
                    and view != edit.view
                    and (not edit.dirty or view != ["home"])
                    and not resume_token
                    and edit.target == "profile"
                ):
                    row.token = uuid4().hex
                    row.payload = {**row.payload, "view": view}
                    return read_edit(row)
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
                    base_profile_hash=snapshot_hash(user),
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
            or snapshot_hash(user) != edit.base_profile_hash
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

    async def save_before_resume(self, owner, resume_token, edit_key):
        """Keep pending edits and issue a fresh preview before the user applies the import."""
        self._enabled()
        async with self.sessions.begin() as session:
            repository = UserRepository(session)
            user = await repository.get(owner, lock=True)
            resume = await session.get(ResumeModel, owner)
            draft = ResumeService._draft(resume, resume_token)
            row = await session.get(ProfileEditModel, owner)
            if row is None or user is None:
                raise ValueError("Правки уже закрыты. Откройте текущее резюме.")
            edit = read_edit(row)
            self._resume_current(user, draft)
            await self._current(session, user, edit)
            await release_resume_editor(session, row, resume_token, edit_key)
            if edit.target == "resume":
                draft.reviewed_profile = edit.profile
            else:
                if draft.reviewed_profile is not None:
                    raise ValueError("В этом резюме уже сохранена отдельная версия профиля.")
                user = await repository.save(replace(user, profile=edit.profile, form_state=None))
                draft.apply_mode = "merge"
                draft.profile_revision = user.revision
                draft.profile_hash = profile_hash(user.profile)
            draft.token = uuid4().hex
            resume.draft = draft.model_dump(mode="json")
            return draft

    async def act(self, owner, token, action, *, text=None):
        """Token is consumed once per step, including navigation and confirmation."""
        self._enabled()
        async with self.sessions.begin() as session:
            repository = UserRepository(session)
            user = await repository.get(owner, lock=True)
            row = await session.get(ProfileEditModel, owner)
            if row is None or row.token != token:
                raise ValueError("Кнопка устарела. Откройте /profile или /resume.")
            op, *args = action
            if op == "cancel":
                await session.delete(row)
                return None
            edit = read_edit(row)
            await self._current(session, user, edit)
            payload = dict(row.payload)
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
                back_page = []
                if op == "text":
                    if edit.view[0] != "input":
                        raise ValueError("Выберите поле кнопкой.")
                    kind, key, field = edit.view[1:]
                    value = text
                else:
                    kind, key, field, value, *back_page = args
                profile, new_key = change(edit.profile, owner, kind, key, field, value)
                payload.update(profile=profile.model_dump(mode="json"), dirty=True)
                if field == "delete":
                    payload["view"] = ["list", kind, back_page[0] if back_page else 0]
                elif kind == "basic":
                    payload["view"] = [section_for(field)]
                else:
                    payload["view"] = ["item", kind, new_key or key]
            else:
                raise ValueError("Неизвестное действие")
            row.payload, row.token = payload, uuid4().hex
            return read_edit(row)
