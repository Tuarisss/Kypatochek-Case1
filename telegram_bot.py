"""Telegram bot entry point."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .chat_service import ChatService
from .config import Config, load_config
from .conversation import ConversationManager
from .db import BotDatabase, BotUser
from .document_store import DocumentStore
from .lm_client import LMStudioClient
from .whisper_client import WhisperCli

LOGGER = logging.getLogger(__name__)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Я помощник по охране труда. Для доступа укажите ваше ФИО и должность, затем задавайте вопросы текстом или голосом."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Отправьте текст или голосовое сообщение. Команды: /help, /reset, /docs, /reload_docs, /stats (админ)."
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    service: ChatService = context.application.bot_data["chat_service"]
    service.conversation.reset(update.effective_chat.id)
    await update.message.reply_text("История диалога очищена.")


async def list_docs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    store: DocumentStore = context.application.bot_data["document_store"]
    await update.message.reply_text(store.describe())


def _ensure_admin(update: Update, config: Config) -> bool:
    return bool(config.admin_ids) and update.effective_user and (
        update.effective_user.id in config.admin_ids
    )


async def reload_docs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    config: Config = context.application.bot_data["config"]
    if not _ensure_admin(update, config):
        await update.message.reply_text("Только администратор может обновлять базу документов.")
        return
    store: DocumentStore = context.application.bot_data["document_store"]
    store.reload()
    await update.message.reply_text("Нормативная база перечитана.")


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    config: Config = context.application.bot_data["config"]
    if not _ensure_admin(update, config):
        await update.message.reply_text("Команда доступна только администраторам.")
        return
    store: DocumentStore = context.application.bot_data["document_store"]
    db = _get_database(context)
    stats = db.get_stats()
    message = _format_stats_message(stats, store.document_count())
    await update.message.reply_text(message)


def _format_context_footer(ctxs):
    if not ctxs:
        return ""
    parts = [f"[{idx}] {chunk.path.name}" for idx, chunk in enumerate(ctxs, start=1)]
    return "\n\nИсточники: " + ", ".join(parts)


def _get_database(context: ContextTypes.DEFAULT_TYPE) -> BotDatabase:
    return context.application.bot_data["database"]


def _refresh_user(context: ContextTypes.DEFAULT_TYPE, telegram_id: int) -> BotUser:
    db = _get_database(context)
    return db.get_or_create_user(telegram_id)


def _get_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> BotUser:
    telegram_user = update.effective_user
    if telegram_user is None:
        raise RuntimeError("Не удалось определить пользователя Telegram")
    db = _get_database(context)
    return db.get_or_create_user(telegram_user.id, telegram_user.username)


def _process_registration_step(user: BotUser, text: str, context: ContextTypes.DEFAULT_TYPE) -> str:
    clean_text = (text or "").strip()
    if not clean_text:
        return "Пожалуйста, отправьте текстовое сообщение с требуемой информацией."
    db = _get_database(context)
    if user.state == "pending_fio":
        db.update_user_profile(user.id, fio=clean_text)
        db.update_user_state(user.id, "pending_profession")
        _refresh_user(context, user.telegram_id)
        return "Спасибо! Теперь укажите вашу должность или профессию."
    if user.state == "pending_profession":
        db.update_user_profile(user.id, profession=clean_text)
        db.update_user_state(user.id, "active")
        _refresh_user(context, user.telegram_id)
        return "Регистрация завершена. Можете задавать вопросы по охране труда."
    return "Регистрация обрабатывается. Попробуйте ещё раз."


def _format_stats_message(stats: dict, doc_count: int) -> str:
    lines = [
        f"Пользователи: {stats['total_users']} (активных {stats['active_users']}, ожидают {stats['pending_users']})",
        f"Сообщений сохранено: {stats['total_interactions']}",
        f"Загруженных документов: {doc_count}",
    ]
    if stats["top_docs"]:
        lines.append("\nТоп документов:")
        for item in stats["top_docs"]:
            name = Path(item["doc_path"]).name
            lines.append(f"- {name}: {item['count']} обращений")
    if stats["recent_doc_events"]:
        lines.append("\nПоследние запросы к документам:")
        for event in stats["recent_doc_events"]:
            who = event["fio"]
            lines.append(
                f"- {event['created_at']}: {who} → {Path(event['doc_path']).name}"
            )
    if stats["user_summaries"]:
        lines.append("\nАктивность пользователей:")
        for summary in stats["user_summaries"]:
            lines.append(
                f"- {summary['fio']} ({summary['profession']}) — {summary['duration']} в системе, последний визит {summary['last_active'] or '—'}"
            )
    return "\n".join(lines)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    service: ChatService = context.application.bot_data["chat_service"]
    user_text = update.message.text or ""
    bot_user = _get_user(update, context)
    if not bot_user.is_active:
        response = _process_registration_step(bot_user, user_text, context)
        await update.message.reply_text(response)
        return
    try:
        reply, ctxs = await service.answer_text(update.effective_chat.id, bot_user, user_text)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Text handler failed")
        await update.message.reply_text(f"Ошибка: {exc}")
        return
    footer = _format_context_footer(ctxs)
    await update.message.reply_text(reply + footer)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    service: ChatService = context.application.bot_data["chat_service"]
    bot_user = _get_user(update, context)
    if not bot_user.is_active:
        await update.message.reply_text(
            "Сначала завершите регистрацию: ответьте текстом с ФИО и должностью."
        )
        return
    voice = update.message.voice or update.message.audio
    if not voice:
        await update.message.reply_text("Не удалось получить голосовое сообщение.")
        return
    file = await voice.get_file()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp_file:
        ogg_path = Path(tmp_file.name)
    await file.download_to_drive(ogg_path)
    try:
        transcription = await service.transcribe_voice(ogg_path)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Voice transcription failed")
        await update.message.reply_text(f"Не удалось распознать голос: {exc}")
        return
    await update.message.reply_text(f"Распознанный текст: {transcription.text}")
    try:
        reply, ctxs = await service.answer_text(
            update.effective_chat.id, bot_user, transcription.text
        )
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("LLM failed after voice")
        await update.message.reply_text(f"Ошибка при обращении к модели: {exc}")
        return
    footer = _format_context_footer(ctxs)
    await update.message.reply_text(reply + footer)


async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Не понимаю этот формат сообщения. Используйте текст или голос.")


def build_application(config: Config):
    document_store = DocumentStore(config.knowledge_root)
    conversation = ConversationManager(config.max_history_messages)
    database = BotDatabase(config.database_path)
    whisper = WhisperCli(
        config.whisper_binary,
        config.whisper_model_path,
        language=config.whisper_language,
        threads=config.whisper_threads,
        ld_library_path=config.whisper_ld_library_path,
    )
    lm_client = LMStudioClient(
        config.lm_api_url,
        config.lm_model,
        temperature=config.lm_temperature,
        max_tokens=config.lm_max_tokens,
    )
    service = ChatService(
        config, lm_client, database, document_store, conversation, whisper
    )

    application = ApplicationBuilder().token(config.telegram_token).build()
    application.bot_data["chat_service"] = service
    application.bot_data["config"] = config
    application.bot_data["document_store"] = document_store
    application.bot_data["database"] = database

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(CommandHandler("docs", list_docs))
    application.add_handler(CommandHandler("reload_docs", reload_docs))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    application.add_handler(MessageHandler(filters.ALL, unknown))
    return application


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    config = load_config()
    LOGGER.info("Starting bot with model %s", config.lm_model)
    app = build_application(config)
    app.run_polling(stop_signals=None)


if __name__ == "__main__":
    main()
