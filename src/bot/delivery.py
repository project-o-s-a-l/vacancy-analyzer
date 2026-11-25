"""Telegram formatting and error classification for the delivery service."""

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError, TelegramNetworkError, TelegramServerError
from aiogram.types import LinkPreviewOptions

from bot.cards import format_card
from bot.keyboards import vacancy_keyboard
from domain import Vacancy
from domain.delivery import DeliveryFailure
from domain.matching import MatchResult
from jobs.catalog_freshness import RECHECK_AFTER


class TelegramSender:
    def __init__(self, bot: Bot, check_interval=RECHECK_AFTER):
        self.check_interval = check_interval
        self.bot = bot

    async def send_recommendation(self, chat_id: int, vacancy: Vacancy, match: MatchResult) -> int:
        try:
            message = await self.bot.send_message(
                chat_id,
                **format_card(vacancy, match, check_interval=self.check_interval).as_kwargs(),
                reply_markup=vacancy_keyboard(vacancy),
                link_preview_options=LinkPreviewOptions(is_disabled=True),
            )
        except (TelegramNetworkError, TelegramServerError) as exc:
            raise DeliveryFailure(uncertain=True, kind=type(exc).__name__) from exc
        except TelegramAPIError as exc:
            raise DeliveryFailure(uncertain=False, kind=type(exc).__name__) from exc
        return message.message_id
