from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from aiogram.enums import ChatType
from aiogram.methods import SendMessage
from aiogram.types import CallbackQuery, Chat, Message, Update, User
from test_bot import TEST_TOKEN, RecordingSession, command_update

from bot.delivery import TelegramSender
from bot.handlers import build_dispatcher
from domain import CandidateProfile, Vacancy
from jobs.profiles import update_field
from jobs.users import UserService
from storage.repositories import VacancyRepository
from storage.repositories.user_repository import RegisteredUser, UserRepository


def text_update(text, user_id=42):
    update = command_update(text, user_id=user_id, chat_id=user_id)
    if not text.startswith("/"):
        update = update.model_copy(
            update={"message": update.message.model_copy(update={"entities": []})}
        )
    return update


def callback_update(data, *, user_id=42, chat_id=None):
    return Update(
        update_id=2,
        callback_query=CallbackQuery(
            id="test-callback",
            chat_instance="private-test",
            from_user=User(id=user_id, is_bot=False, first_name="Candidate"),
            data=data,
            message=Message(
                message_id=1,
                date=datetime.now(UTC),
                chat=Chat(id=chat_id or user_id, type=ChatType.PRIVATE),
                from_user=User(id=123456, is_bot=True, first_name="Test"),
            ),
        ),
    )


async def store_profile(sessions, user_id, profile):
    async with sessions.begin() as session:
        repository = UserRepository(session)
        await repository.register(user_id)
        await repository.save(RegisteredUser(user_id, profile, None))


async def test_registration_has_independent_empty_profiles_and_is_idempotent(sessions):
    users = UserService(sessions)
    first = await users.register(42)
    assert first.profile.skills == {}
    assert first.form_state == "setup:skills"
    await users.accept_answer(42, "Python, Postgres")
    again = await users.register(42)
    assert set(again.profile.skills) == {"python", "postgresql"}
    assert again.form_state == "setup:experience"
    assert (await users.register(99)).profile.skills == {}
    assert (await users.get(42)).profile.skills != (await users.get(99)).profile.skills


async def test_form_state_and_answers_survive_service_restart(sessions):
    users = UserService(sessions)
    await users.register(42)
    await users.accept_answer(42, "Python")
    restarted = UserService(sessions)
    assert (await restarted.get(42)).form_state == "setup:experience"
    await restarted.accept_answer(42, "2,5")
    assert (await users.get(42)).profile.experience_years == 2.5


async def test_two_users_can_fill_profiles_interleaved(sessions):
    users = UserService(sessions)
    await users.register(42)
    await users.register(99)
    first = [
        "Python, Docker",
        "3",
        "Python developer",
        "required",
        "150000 RUB до налогов",
        "пропустить",
    ]
    second = [
        "Java, Kafka",
        "7",
        "Java developer",
        "office",
        "250000 RUB на руки",
        "Санкт-Петербург",
    ]
    for first_answer, second_answer in zip(first, second, strict=True):
        await users.accept_answer(42, first_answer)
        await users.accept_answer(99, second_answer)
    a, b = await users.get(42), await users.get(99)
    assert a.form_state is b.form_state is None
    assert a.profile.preferences.remote == "required"
    assert b.profile.preferences.remote == "office"
    assert a.profile.preferences.salary_gross is True
    assert b.profile.preferences.salary_gross is False
    assert a.profile.preferences.locations == []
    assert b.profile.preferences.locations == ["Санкт-Петербург"]
    assert set(a.profile.skills) == {"python", "docker"}
    assert set(b.profile.skills) == {"java", "kafka"}


@pytest.mark.parametrize(
    "field,text",
    [
        ("skills", "пропустить"),
        ("skills", "x" * 101),
        ("experience", "nan"),
        ("experience", "-1"),
        ("experience", "200"),
        ("experience", "не знаю"),
        ("remote", "иногда"),
        ("salary", "150000"),
        ("salary", "150000 RUB"),
        ("salary", "1000000001 RUB net"),
        ("positions", ",,"),
    ],
)
def test_invalid_profile_answers_rejected(field, text):
    with pytest.raises(ValueError):
        update_field(CandidateProfile(), field, text)


async def test_invalid_answer_keeps_previous_data_and_step(sessions):
    users = UserService(sessions)
    await users.register(42)
    await users.accept_answer(42, "Python")
    with pytest.raises(ValueError):
        await users.accept_answer(42, "nan")
    user = await users.get(42)
    assert user.form_state == "setup:experience"
    assert user.profile.experience_years is None
    assert set(user.profile.skills) == {"python"}


async def test_edit_and_cancel_do_not_reset_other_fields(sessions, profile):
    await store_profile(sessions, 42, profile)
    users = UserService(sessions)
    await users.begin_edit(42, "locations")
    updated = await users.accept_answer(42, "Казань")
    assert updated.form_state is None
    assert updated.profile.preferences.locations == ["Казань"]
    assert updated.profile.skills == profile.skills
    await users.begin_edit(42, "skills")
    cancelled = await users.cancel(42)
    assert cancelled.form_state is None
    assert cancelled.profile == updated.profile


async def test_stale_button_does_not_change_current_form_field(sessions):
    users = UserService(sessions)
    await users.register(42)
    await users.accept_answer(42, "Python")
    with pytest.raises(ValueError, match="предыдущему шагу"):
        await users.accept_answer(42, "required", expected_field="remote")
    assert (await users.get(42)).form_state == "setup:experience"


async def test_public_start_accepts_different_users_without_shared_profile(sessions):
    users = UserService(sessions)
    recording = RecordingSession()
    bot = Bot(TEST_TOKEN, session=recording)
    dispatcher = build_dispatcher(users)
    for user_id in [42, 99]:
        await dispatcher.feed_update(bot, text_update("/start", user_id))
    assert {method.chat_id for method in recording.requests if isinstance(method, SendMessage)} == {
        42,
        99,
    }
    assert (await users.get(42)).profile.skills == {}
    assert (await users.get(99)).profile.skills == {}


async def test_profile_setup_with_real_dispatcher_and_buttons(sessions):
    users = UserService(sessions)
    recording = RecordingSession()
    bot = Bot(TEST_TOKEN, session=recording)
    dispatcher = build_dispatcher(users)
    for text in ["/start", "Python, SQL", "3", "Python developer"]:
        await dispatcher.feed_update(bot, text_update(text))
    await dispatcher.feed_update(bot, callback_update("answer:remote:required"))
    for text in ["150 000 RUB до налогов", "пропустить"]:
        await dispatcher.feed_update(bot, text_update(text))
    user = await users.get(42)
    assert user.form_state is None
    assert user.profile.preferences.salary_min == 150000
    assert user.profile.preferences.remote == "required"
    assert "Профиль сохранён" in recording.requests[-1].text


async def test_callback_cannot_edit_another_users_profile(sessions, profile):
    users = UserService(sessions)
    await store_profile(sessions, 42, profile)
    await users.register(99)
    recording = RecordingSession()
    bot = Bot(TEST_TOKEN, session=recording)
    dispatcher = build_dispatcher(users)
    await dispatcher.feed_update(bot, callback_update("profile:skills", user_id=99, chat_id=42))
    assert recording.requests == []
    assert (await users.get(42)).profile == profile
    await dispatcher.feed_update(bot, callback_update("profile:skills:42", user_id=99))
    assert (await users.get(42)).profile == profile
    assert (await users.get(99)).form_state == "setup:skills"


async def test_users_see_only_their_own_profiles(sessions, profile):
    users = UserService(sessions)
    await store_profile(sessions, 42, profile)
    java = CandidateProfile.model_validate({"skills": {"java": {}}})
    await store_profile(sessions, 99, java)
    recording = RecordingSession()
    bot = Bot(TEST_TOKEN, session=recording)
    dispatcher = build_dispatcher(users)
    await dispatcher.feed_update(bot, text_update("/profile", 42))
    await dispatcher.feed_update(bot, text_update("/profile", 99))
    messages = [method for method in recording.requests if isinstance(method, SendMessage)]
    assert "python" in messages[0].text and "java" not in messages[0].text
    assert "java" in messages[1].text and "python" not in messages[1].text


async def test_recommendations_and_delivery_history_are_independent(sessions, profile, vacancy):
    vacancy.published_at = datetime.now(UTC)
    async with sessions.begin() as session:
        await VacancyRepository(session).save(vacancy)
    for user_id in [42, 99]:
        await store_profile(sessions, user_id, profile)
    users = UserService(sessions, top=1, min_score=0)
    a, b = await users.notifications(42), await users.notifications(99)
    sender = SimpleNamespace(send_message=AsyncMock(return_value=SimpleNamespace(message_id=1)))
    assert (await a.deliver(TelegramSender(sender))).sent == 1
    assert await a.pending() == []
    assert len(await b.pending()) == 1
    assert await b.delivery_counts() == {}
    assert (await b.deliver(TelegramSender(sender))).sent == 1
    assert {call.args[0] for call in sender.send_message.await_args_list} == {42, 99}


async def test_different_profiles_get_different_first_recommendations(
    sessions, sample_data, profile
):
    async with sessions.begin() as session:
        for payload in sample_data["vacancies"]:
            payload["published_at"] = datetime.now(UTC).isoformat()
            await VacancyRepository(session).save(Vacancy.model_validate(payload))
    await store_profile(sessions, 42, profile)
    java = CandidateProfile.model_validate(
        {
            "skills": {"java": {}, "kafka": {}, "kubernetes": {}},
            "experience_years": 7,
            "preferences": {"positions": ["java developer"], "remote": "office"},
        }
    )
    await store_profile(sessions, 99, java)
    users = UserService(sessions, top=1, min_score=0)
    assert (await (await users.notifications(42)).pending())[0][0].source_id == "demo-1"
    assert (await (await users.notifications(99)).pending())[0][0].source_id == "demo-3"


async def test_unregistered_or_empty_profiles_cannot_receive_recommendations(sessions):
    users = UserService(sessions)
    with pytest.raises(ValueError, match="/start"):
        await users.notifications(42)
    await users.register(42)
    with pytest.raises(ValueError, match="навыки"):
        await users.notifications(42)
