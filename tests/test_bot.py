from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from aiogram.client.session.base import BaseSession
from aiogram.enums import ChatType, MessageEntityType
from aiogram.methods import GetMe, SendMessage
from aiogram.types import Chat, Message, MessageEntity, Update, User
from asyncpg import ConnectionDoesNotExistError

from bot.cards import render_card, render_profile
from bot.handlers import build_dispatcher
from jobs.cli import main
from jobs.notify import DeliveryReport
from jobs.settings import Settings
from jobs.users import UserService
from scoring.baseline import score_vacancy

TEST_TOKEN = "123456:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"


class RecordingSession(BaseSession):
    """Exercise the real dispatcher and Bot methods without network access."""

    def __init__(self):
        super().__init__()
        self.requests = []

    async def make_request(self, bot, method, timeout=None):  # noqa: ASYNC109 — aiogram API
        self.requests.append(method)
        if isinstance(method, GetMe):
            return User(id=123456, is_bot=True, first_name="Test", username="test_bot")
        if isinstance(method, SendMessage):
            return Message(
                message_id=len(self.requests),
                date=datetime.now(UTC),
                chat=Chat(id=method.chat_id, type=ChatType.PRIVATE),
                text=method.text,
            )
        return True

    async def close(self):
        pass

    async def stream_content(self, url, **kwargs):
        yield b""


def command_update(command, *, user_id=42, chat_id=42, chat_type=ChatType.PRIVATE):
    return Update(
        update_id=1,
        message=Message(
            message_id=1,
            date=datetime.now(UTC),
            chat=Chat(id=chat_id, type=chat_type),
            from_user=User(id=user_id, is_bot=False, first_name="Candidate"),
            text=command,
            entities=[
                MessageEntity(type=MessageEntityType.BOT_COMMAND, offset=0, length=len(command))
            ],
        ),
    )


@pytest.fixture
async def service(sessions, profile):
    from storage.repositories.user_repository import RegisteredUser, UserRepository

    service = UserService(sessions)
    async with sessions.begin() as session:
        repository = UserRepository(session)
        await repository.register(42)
        await repository.save(RegisteredUser(42, profile, None))
    return service


@pytest.mark.parametrize(
    "command,expected",
    [
        ("/start", "Помогаю выбирать"),
        ("/profile", "Текущий профиль"),
        ("/status", "Отправлено вам"),
        ("/latest", "Новых рекомендаций"),
    ],
)
async def test_registered_user_commands_pass_real_dispatcher(service, command, expected):
    session = RecordingSession()
    bot = Bot(TEST_TOKEN, session=session)
    try:
        await build_dispatcher(service).feed_update(bot, command_update(command))
        messages = [method for method in session.requests if isinstance(method, SendMessage)]
        assert messages
        assert any(expected in message.text for message in messages)
        assert messages[0].chat_id == 42
        assert messages[0].parse_mode is None
    finally:
        await bot.session.close()


@pytest.mark.parametrize(
    "update_kwargs",
    [
        {"user_id": 99, "chat_id": 42},
        {"chat_id": -100, "chat_type": ChatType.GROUP},
    ],
)
async def test_mismatched_users_and_groups_are_silent(service, update_kwargs):
    session = RecordingSession()
    bot = Bot(TEST_TOKEN, session=session)
    await build_dispatcher(service).feed_update(bot, command_update("/profile", **update_kwargs))
    assert session.requests == []


async def test_latest_reports_ambiguous_delivery(service):
    service.notifications = AsyncMock(
        return_value=SimpleNamespace(deliver=AsyncMock(return_value=DeliveryReport(uncertain=1)))
    )
    session = RecordingSession()
    bot = Bot(TEST_TOKEN, session=session)
    await build_dispatcher(service).feed_update(bot, command_update("/latest"))
    assert "не подтверждена" in session.requests[-1].text


async def test_latest_shows_specific_reason_for_empty_selection(service):
    service.notifications = AsyncMock(
        return_value=SimpleNamespace(
            deliver=AsyncMock(
                return_value=DeliveryReport(
                    empty_message="Новых рекомендаций нет: лучшая оценка 44%, порог 50%."
                )
            )
        )
    )
    session = RecordingSession()
    bot = Bot(TEST_TOKEN, session=session)
    await build_dispatcher(service).feed_update(bot, command_update("/latest"))
    assert "лучшая оценка 44%, порог 50%" in session.requests[-1].text


def test_card_plain_text_and_bounded_utf16_length(vacancy, profile, now):
    vacancy.title = "<b>" + "😀" * 10000
    vacancy.company = "<script>" + "X" * 10000
    vacancy.location = "Y" * 10000
    match = score_vacancy(vacancy, profile, now=now)
    match.matched_skills = ["😀" * 10000]
    match.missing_skills = ["Z" * 10000]
    match.reasons = ["😀" * 10000] * 100
    match.warnings = ["😀" * 10000] * 100
    text = render_card(vacancy, match)
    assert len(text.encode("utf-16-le")) // 2 < 4096
    assert "<b>" in text
    assert "<script>" in text


def test_card_shows_salary_semantics(vacancy, profile, now):
    text = render_card(vacancy, score_vacancy(vacancy, profile, now=now))
    assert "RUB / мес., до налогов" in text
    assert "180 000–250 000" in text


def test_profile_length_is_bounded(profile):
    profile.preferences.positions = ["😀" * 10000]
    profile.preferences.locations = ["😀" * 10000]
    assert len(render_profile(profile).encode("utf-16-le")) // 2 < 4096


@pytest.mark.parametrize("command", ["demo", "telegram-preview"])
def test_synthetic_commands_are_not_part_of_runtime(command, monkeypatch):
    monkeypatch.setattr("sys.argv", ["vacancy-analyzer", command])
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 2


def test_missing_settings_fail_with_actionable_message():
    settings = Settings(_env_file=None)
    assert "telegram_owner_id" not in Settings.model_fields
    with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN"):
        settings.require_telegram_token()


def test_invalid_token_not_printed(monkeypatch, capsys):
    from jobs import cli

    secret = "invalid-secret-token"
    monkeypatch.setattr("sys.argv", ["vacancy-analyzer", "notify", "--user-id", "42"])
    monkeypatch.setattr(cli, "load_profile", lambda path: SimpleNamespace())
    monkeypatch.setattr(
        cli,
        "Settings",
        lambda: Settings(_env_file=None, telegram_bot_token=secret),
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    output = capsys.readouterr()
    assert "TELEGRAM_BOT_TOKEN" in output.err
    assert secret not in output.err


@pytest.mark.parametrize("driver_error", [False, True])
async def test_database_failure_is_reported_without_diagnostics(service, driver_error):
    from sqlalchemy.exc import SQLAlchemyError

    error_class = ConnectionDoesNotExistError if driver_error else SQLAlchemyError
    service.notifications = AsyncMock(side_effect=error_class("private connection diagnostics"))
    session = RecordingSession()
    bot = Bot(TEST_TOKEN, session=session)
    await build_dispatcher(service).feed_update(bot, command_update("/status"))
    assert "Сервис временно недоступен" in session.requests[-1].text
    assert "private connection diagnostics" not in session.requests[-1].text


@pytest.mark.parametrize(
    "error",
    [
        ConnectionDoesNotExistError("private connection diagnostics"),
        ConnectionResetError("private connection diagnostics"),
    ],
)
def test_cli_database_connection_failure_is_concise(monkeypatch, capsys, error):
    from jobs import cli

    monkeypatch.setattr("sys.argv", ["vacancy-analyzer", "bot"])
    monkeypatch.setattr(cli, "run_telegram", AsyncMock(side_effect=error))
    with pytest.raises(SystemExit) as stopped:
        main()
    assert stopped.value.code == 1
    output = capsys.readouterr()
    assert "Ошибка БД" in output.err
    assert "Traceback" not in output.err
    assert "private connection diagnostics" not in output.err


async def test_existing_webhook_is_preserved(monkeypatch):
    from bot import runtime

    bot = SimpleNamespace(
        session=SimpleNamespace(close=AsyncMock()),
        get_webhook_info=AsyncMock(return_value=SimpleNamespace(url="https://example.com/webhook")),
        set_my_commands=AsyncMock(),
    )
    engine = SimpleNamespace(dispose=AsyncMock())
    service = SimpleNamespace(check_ready=AsyncMock())
    monkeypatch.setattr(runtime, "create_bot", lambda settings: bot)
    monkeypatch.setattr(runtime, "create_engine", lambda url: engine)
    monkeypatch.setattr(runtime, "session_factory", lambda engine: None)
    monkeypatch.setattr(runtime, "UserService", lambda *args, **kwargs: service)
    with pytest.raises(ValueError, match="webhook"):
        await runtime.run_bot(Settings(_env_file=None))
    bot.set_my_commands.assert_not_awaited()
    bot.session.close.assert_awaited_once()
    engine.dispose.assert_awaited_once()


async def test_main_menu_contains_six_core_actions(monkeypatch):
    from bot import runtime

    bot = SimpleNamespace(
        session=SimpleNamespace(close=AsyncMock()),
        get_webhook_info=AsyncMock(return_value=SimpleNamespace(url="")),
        set_my_commands=AsyncMock(),
    )
    engine = SimpleNamespace(dispose=AsyncMock())
    service = SimpleNamespace(check_ready=AsyncMock())
    monkeypatch.setattr(runtime, "create_bot", lambda _: bot)
    monkeypatch.setattr(runtime, "create_engine", lambda _: engine)
    monkeypatch.setattr(runtime, "session_factory", lambda _: None)
    monkeypatch.setattr(runtime, "UserService", lambda *args, **kwargs: service)
    polling = AsyncMock()
    monkeypatch.setattr(
        runtime, "build_dispatcher", lambda _: SimpleNamespace(start_polling=polling)
    )
    await runtime.run_bot(Settings(_env_file=None))
    commands = bot.set_my_commands.call_args.args[0]
    assert {item.command for item in commands} == {
        "resume",
        "latest",
        "profile",
        "saved",
        "applied",
        "skillgap",
    }
    polling.assert_awaited_once()


@pytest.mark.parametrize("command", ["/market", "/stats"])
async def test_removed_market_commands_return_core_help(service, command):
    transport = RecordingSession()
    bot = Bot(TEST_TOKEN, session=transport)
    await build_dispatcher(service).feed_update(bot, command_update(command))
    text = transport.requests[-1].text
    assert "/latest" in text and "/market" not in text and "/stats" not in text


async def test_telegram_id_uses_token_without_owner_or_database(monkeypatch, capsys):
    from bot import runtime
    from jobs import cli

    updates = [
        command_update("/start"),
        command_update("/start", user_id=99, chat_id=-100, chat_type=ChatType.GROUP),
    ]
    bot = SimpleNamespace(
        session=SimpleNamespace(close=AsyncMock()),
        get_updates=AsyncMock(return_value=updates),
        get_me=AsyncMock(return_value=SimpleNamespace(username="test_bot", first_name="Test")),
        get_webhook_info=AsyncMock(return_value=SimpleNamespace(url="")),
    )
    monkeypatch.setattr(runtime, "create_bot", lambda settings: bot)
    monkeypatch.setattr(cli, "Settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(cli, "create_engine", lambda url: pytest.fail("ID lookup opened database"))
    await cli.run_telegram(cli.parser().parse_args(["telegram-id"]))
    text = capsys.readouterr().out
    assert "42: Candidate" in text and "99:" not in text
    assert "/start" in text
    assert "@test_bot" in text
    assert "offset" not in bot.get_updates.await_args.kwargs
    assert bot.get_updates.await_args.kwargs["timeout"] == 30
    assert bot.get_updates.await_args.kwargs["request_timeout"] > 30
    bot.session.close.assert_awaited_once()


async def test_owner_lookup_reports_empty_queue_and_correct_bot(capsys):
    from bot.onboarding import show_private_chat_ids

    bot = SimpleNamespace(
        get_me=AsyncMock(
            return_value=SimpleNamespace(username="correct_bot", first_name="Correct")
        ),
        get_webhook_info=AsyncMock(return_value=SimpleNamespace(url="")),
        get_updates=AsyncMock(return_value=[]),
    )
    await show_private_chat_ids(bot, wait_seconds=0)
    text = capsys.readouterr().out
    assert "https://t.me/correct_bot" in text
    assert "другие запущенные экземпляры" in text
    assert bot.get_updates.await_args.kwargs["timeout"] == 0


async def test_owner_lookup_preserves_webhook_and_does_not_poll():
    from bot.onboarding import show_private_chat_ids

    bot = SimpleNamespace(
        get_me=AsyncMock(return_value=SimpleNamespace(username="test_bot", first_name="Test")),
        get_webhook_info=AsyncMock(return_value=SimpleNamespace(url="https://example.com/private")),
        get_updates=AsyncMock(),
    )
    with pytest.raises(ValueError, match="webhook"):
        await show_private_chat_ids(bot)
    bot.get_updates.assert_not_awaited()
