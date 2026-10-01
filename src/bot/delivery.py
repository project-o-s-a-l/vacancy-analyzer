"""Telegram formatting and error classification for the delivery service."""

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError, TelegramNetworkError, TelegramServerError
from aiogram.types import LinkPreviewOptions

from bot.cards import render_card
from bot.keyboards import vacancy_keyboard
from domain import Vacancy
from domain.delivery import DeliveryFailure
from domain.matching import MatchResult


class TelegramSender:
    def __init__(self, bot: Bot):
        self.bot = bot

    async def send_recommendation(self, chat_id: int, vacancy: Vacancy, match: MatchResult) -> int:
        try:
            message = await self.bot.send_message(
                chat_id,
                render_card(vacancy, match),
                parse_mode=None,
                reply_markup=vacancy_keyboard(vacancy),
                link_preview_options=LinkPreviewOptions(is_disabled=True),
            )
        except (TelegramNetworkError, TelegramServerError) as exc:
            raise DeliveryFailure(uncertain=True, kind=type(exc).__name__) from exc
        except TelegramAPIError as exc:
            raise DeliveryFailure(uncertain=False, kind=type(exc).__name__) from exc
        return message.message_id
