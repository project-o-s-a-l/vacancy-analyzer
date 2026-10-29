from aiogram import Dispatcher, Router
from aiogram.types import Message

from bot.common import PrivateChatsMiddleware
from bot.concurrent import OrderedDispatcher
from bot.help import HELP
from bot.profiles import register_profiles
from bot.recommendations import register_recommendations
from bot.resumes import register_resumes
from jobs.users import UserService


def build_dispatcher(service: UserService) -> Dispatcher:
    router = Router(name="user_commands")
    router.message.outer_middleware(PrivateChatsMiddleware())
    router.callback_query.outer_middleware(PrivateChatsMiddleware())
    register_profiles(router, service)
    register_resumes(router, service)
    register_recommendations(router, service)

    @router.message()
    async def fallback(message: Message) -> None:
        await message.answer(HELP, parse_mode=None)

    dispatcher = OrderedDispatcher()
    dispatcher.include_router(router)
    return dispatcher
