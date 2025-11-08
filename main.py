from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode

from bot.ai_client import AIClient
from bot.config import load_config
from bot.db import Database
from bot.handlers import setup_routers


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    settings = load_config()
    if not settings.bot_token:
        raise RuntimeError("BOT_TOKEN не задан в .env")

    bot = Bot(token=settings.bot_token, parse_mode=ParseMode.HTML)
    dp = Dispatcher()

    database = Database(settings.database_path)
    ai_client = AIClient(settings.lm_studio_url, settings.lm_studio_model)

    bot["settings"] = settings
    bot["db"] = database
    bot["ai_client"] = ai_client

    for router in setup_routers():
        dp.include_router(router)

    await bot.delete_webhook(drop_pending_updates=True)
    logging.info("Bot is up and polling...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logging.info("Bot stopped")
