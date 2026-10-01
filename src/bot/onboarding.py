"""Read the bot identity and wait for a private message without consuming its update."""

from aiogram import Bot


async def show_private_chat_ids(bot: Bot, *, wait_seconds: int = 30) -> None:
    identity = await bot.get_me()
    username = identity.username
    print(
        f"Бот из TELEGRAM_BOT_TOKEN: @{username} ({identity.first_name})."
        if username
        else f"Бот из TELEGRAM_BOT_TOKEN: {identity.first_name}.",
        flush=True,
    )
    if username:
        print(f"Откройте https://t.me/{username}", flush=True)
    webhook = await bot.get_webhook_info()
    if webhook.url:
        raise ValueError(
            "Для этого бота установлен webhook: сообщения получает другой обработчик. "
            "Отключите его перед получением ID. Webhook автоматически не изменён."
        )
    if wait_seconds:
        print(
            f"Отправьте новое /start этому боту. Ожидаю сообщение до {wait_seconds} секунд…",
            flush=True,
        )
    # Omitting offset keeps pending updates available to the main bot after onboarding.
    updates = await bot.get_updates(
        timeout=wait_seconds,
        limit=100,
        allowed_updates=["message"],
        request_timeout=wait_seconds + 10,
    )
    users = {}
    for update in updates:
        message = update.message
        if (
            message
            and message.chat.type == "private"
            and message.from_user
            and not message.from_user.is_bot
            and message.chat.id == message.from_user.id
        ):
            user = message.from_user
            label = " ".join((user.full_name + " " + (user.username or "")).split())[:120]
            users[user.id] = label
    if users:
        for user_id, label in sorted(users.items()):
            print(f"{user_id}: {label}")
        print("Это ID личных чатов. Для работы с ботом пользователи регистрируются через /start.")
    else:
        print(
            "Новых личных сообщений не найдено. Проверьте @имя бота выше, "
            "остановите другие запущенные экземпляры бота, затем повторите команду "
            "и отправьте новое /start во время ожидания."
        )
