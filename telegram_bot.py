"""Telegram bot entry point."""
from __future__ import annotations

import logging
import tempfile
import time
from contextlib import suppress
from pathlib import Path

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, Update
from telegram.error import TelegramError
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .chat_service import ChatService
from .config import Config, load_config
from .conversation import ConversationManager
from .db import BotDatabase, BotUser
from .document_store import DocumentStore, SUPPORTED_EXTENSIONS
from .lm_client import LMStudioClient
from .whisper_client import WhisperCli

BTN_HELP = "ℹ️ Помощь"
BTN_DOCS = "📚 Документы"
BTN_RESET = "🧹 Сброс истории"
BTN_RELOAD = "🔄 Обновить базу"
BTN_STATS = "📊 Статистика"
BTN_QUIZ = "📝 Тест"
CONSENT_NOTICE = (
    "Привет! Я помощник по охране труда."
    "затем задавайте вопросы текстом, голосом или отправляйте фото.\n\n"
    "Отправляя свои данные, вы подтверждаете согласие на обработку персональных данных "
    "в соответствии с законодательством РФ."
)
CONSENT_INSTRUCTION = (
    "Для работы бота требуется согласие на обработку персональных данных "
    "(Федеральный закон № 152-ФЗ). Отправьте сообщение «Согласен» или «Согласна», "
    "если принимаете условия."
)
CONSENT_KEYWORDS = {"согласен", "согласна", "принимаю", "да"}
AGREE_CALLBACK = "consent_agree"
DECLINE_CALLBACK = "consent_decline"
QUIZ_ANSWER_PREFIX = "quiz_answer_"
QUIZ_FINISH = "quiz_finish"

LOGGER = logging.getLogger(__name__)


def _is_admin_user(update: Update, config: Config) -> bool:
    return (
        bool(config.admin_ids)
        and update.effective_user is not None
        and update.effective_user.id in config.admin_ids
    )


def _build_keyboard(is_admin: bool) -> ReplyKeyboardMarkup:
    rows = [
        [BTN_HELP, BTN_DOCS],
        [BTN_RESET, BTN_QUIZ],
    ]
    if is_admin:
        rows.append([BTN_RELOAD, BTN_STATS])
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)


def _consent_inline_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("✅ Согласен", callback_data=AGREE_CALLBACK),
                InlineKeyboardButton("❌ Не согласен", callback_data=DECLINE_CALLBACK),
            ]
        ]
    )


def _build_quiz_keyboard(options: list[str]) -> InlineKeyboardMarkup:
    rows = [
        [
            InlineKeyboardButton(
                f"{idx + 1}) {option}", callback_data=f"{QUIZ_ANSWER_PREFIX}{idx}"
            )
        ]
        for idx, option in enumerate(options)
    ]
    rows.append([InlineKeyboardButton("⛔ Завершить тест", callback_data=QUIZ_FINISH)])
    return InlineKeyboardMarkup(rows)


async def handle_consent_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    query = update.callback_query
    await query.answer()
    user = _get_user(update, context)
    config: Config = context.application.bot_data["config"]
    is_admin = _is_admin_user(update, config)
    if user.state != "pending_consent":
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text(
            "Согласие уже подтверждено.",
            reply_markup=_build_keyboard(is_admin),
        )
        return
    if query.data == AGREE_CALLBACK:
        db = _get_database(context)
        db.mark_user_consent(user.id)
        db.update_user_state(user.id, "pending_fio")
        _refresh_user(context, user.telegram_id)
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text(
            "Спасибо! Укажите ваше полное ФИО.",
            reply_markup=_build_keyboard(is_admin),
        )
    elif query.data == DECLINE_CALLBACK:
        await query.message.reply_text(
            "Без согласия на обработку персональных данных бот недоступен. Возвращайтесь, когда будете готовы согласиться.",
            reply_markup=_consent_inline_keyboard(),
        )


async def handle_quiz_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    query = update.callback_query
    await query.answer()
    bot_user = _get_user(update, context)
    config: Config = context.application.bot_data["config"]
    is_admin = _is_admin_user(update, config)
    db = _get_database(context)
    data = query.data or ""
    if data == QUIZ_FINISH:
        session = db.get_quiz_session(bot_user.id)
        await query.edit_message_reply_markup(reply_markup=None)
        if not session:
            await query.message.reply_text(
                "Активный тест не найден.",
                reply_markup=_build_keyboard(is_admin),
            )
            return
        total = session.questions_answered
        correct = session.correct_answers
        summary = (
            f"Тест завершён. Правильных ответов: {correct} из {total}."
            if total
            else "Тест завершён. Вы не ответили ни на один вопрос."
        )
        db.clear_quiz_session(bot_user.id)
        await query.message.reply_text(
            summary,
            reply_markup=_build_keyboard(is_admin),
        )
        return
    if not data.startswith(QUIZ_ANSWER_PREFIX):
        return
    try:
        chosen_index = int(data[len(QUIZ_ANSWER_PREFIX) :])
    except ValueError:
        await query.message.reply_text(
            "Не удалось распознать ответ.",
            reply_markup=_build_keyboard(is_admin),
        )
        return
    await query.edit_message_reply_markup(reply_markup=None)
    await _handle_quiz_answer_selection(
        update.effective_chat.id,
        context,
        bot_user,
        chosen_index,
        query.message.reply_text,
        is_admin,
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    config: Config = context.application.bot_data["config"]
    bot_user = _get_user(update, context)
    is_admin = _is_admin_user(update, config)
    if bot_user.state == "pending_consent":
        await update.message.reply_text(
            CONSENT_NOTICE + "\n\n" + CONSENT_INSTRUCTION,
            reply_markup=_consent_inline_keyboard(),
        )
        return
    await update.message.reply_text(CONSENT_NOTICE, reply_markup=_build_keyboard(is_admin))


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    config: Config = context.application.bot_data["config"]
    bot_user = _get_user(update, context)
    is_admin = _is_admin_user(update, config)
    if bot_user.state == "pending_consent":
        await update.message.reply_text(
            CONSENT_NOTICE + "\n\n" + CONSENT_INSTRUCTION,
            reply_markup=_consent_inline_keyboard(),
        )
        return
    await update.message.reply_text(
        "Отправьте текст, голос или фото. Используйте кнопки ниже для быстрого доступа к функциям.",
        reply_markup=_build_keyboard(is_admin),
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    service: ChatService = context.application.bot_data["chat_service"]
    bot_user = _get_user(update, context)
    service.conversation.reset(update.effective_chat.id)
    db = _get_database(context)
    db.clear_quiz_session(bot_user.id)
    config: Config = context.application.bot_data["config"]
    is_admin = _is_admin_user(update, config)
    await update.message.reply_text(
        "История диалога очищена.",
        reply_markup=_build_keyboard(is_admin),
    )


async def list_docs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    store: DocumentStore = context.application.bot_data["document_store"]
    config: Config = context.application.bot_data["config"]
    is_admin = _is_admin_user(update, config)
    files = store.list_files()
    if not files:
        await update.message.reply_text(
            "Документы: отсутствуют. Добавьте файлы в папку knowledge_base.",
            reply_markup=_build_keyboard(is_admin),
        )
        return
    context.user_data["doc_options"] = [str(path) for path in files]
    lines = ["Документы:"]
    for idx, path in enumerate(files, start=1):
        lines.append(f"{idx}) {path.name}")
    lines.append("")
    lines.append("Отправьте номер документа, чтобы получить PDF.")
    await update.message.reply_text(
        "\n".join(lines),
        reply_markup=_build_keyboard(is_admin),
    )


def _ensure_admin(update: Update, config: Config) -> bool:
    return _is_admin_user(update, config)


async def reload_docs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    config: Config = context.application.bot_data["config"]
    if not _ensure_admin(update, config):
        await update.message.reply_text(
            "Только администратор может обновлять базу документов.",
            reply_markup=_build_keyboard(False),
        )
        return
    store: DocumentStore = context.application.bot_data["document_store"]
    store.reload()
    await update.message.reply_text(
        "Нормативная база перечитана.",
        reply_markup=_build_keyboard(True),
    )


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    config: Config = context.application.bot_data["config"]
    if not _ensure_admin(update, config):
        await update.message.reply_text(
            "Команда доступна только администраторам.",
            reply_markup=_build_keyboard(False),
        )
        return
    store: DocumentStore = context.application.bot_data["document_store"]
    db = _get_database(context)
    stats = db.get_stats()
    message = _format_stats_message(stats, store.document_count())
    await update.message.reply_text(
        message,
        reply_markup=_build_keyboard(True),
    )


async def start_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    bot_user = _get_user(update, context)
    config: Config = context.application.bot_data["config"]
    is_admin = _is_admin_user(update, config)
    if not bot_user.is_active:
        await update.message.reply_text(
            "Сначала завершите регистрацию: подтвердите согласие и укажите ФИО/должность.",
            reply_markup=_build_keyboard(is_admin),
        )
        return
    db = _get_database(context)
    db.clear_quiz_session(bot_user.id)
    waiting_message = await update.message.reply_text(
        "Генерация тестового вопроса...",
        reply_markup=_build_keyboard(is_admin),
    )
    try:
        await _send_quiz_question(update.effective_chat.id, update, context, bot_user)
    finally:
        with suppress(TelegramError):
            await waiting_message.delete()


async def handle_document_upload(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    document = update.message.document
    if not document:
        return
    config: Config = context.application.bot_data["config"]
    if not _ensure_admin(update, config):
        await update.message.reply_text(
            "Загрузка файлов доступна только администраторам.",
            reply_markup=_build_keyboard(False),
        )
        return
    file_name = document.file_name or f"document_{int(time.time())}.pdf"
    extension = Path(file_name).suffix.lower()
    allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    if extension not in SUPPORTED_EXTENSIONS:
        await update.message.reply_text(
            f"Неподдерживаемый формат ({extension}). Допустимо: {allowed}"
        )
        return
    safe_name = Path(file_name).name
    target_path = config.knowledge_root / safe_name
    if target_path.exists():
        target_path = (
            config.knowledge_root
            / f"{target_path.stem}_{int(time.time())}{target_path.suffix}"
        )
    telegram_file = await document.get_file()
    await telegram_file.download_to_drive(target_path)
    store: DocumentStore = context.application.bot_data["document_store"]
    store.reload()
    await update.message.reply_text(
        f"Файл {target_path.name} загружен и добавлен в нормативную базу.",
        reply_markup=_build_keyboard(True),
    )


def _format_context_footer(ctxs):
    filtered = [
        chunk for chunk in ctxs if getattr(chunk, "score", 1.0) >= 0.3
    ]
    if not filtered:
        return ""
    parts = [f"[{idx}] {chunk.path.name}" for idx, chunk in enumerate(filtered, start=1)]
    return "\n\nИсточники: " + ", ".join(parts)


async def _try_handle_doc_request(
    update: Update, context: ContextTypes.DEFAULT_TYPE, user_text: str
) -> bool:
    doc_options = context.user_data.get("doc_options")
    if not doc_options:
        return False
    normalized = (user_text or "").strip()
    if not normalized.isdigit():
        return False
    idx = int(normalized) - 1
    config: Config = context.application.bot_data["config"]
    is_admin = _is_admin_user(update, config)
    if idx < 0 or idx >= len(doc_options):
        await update.message.reply_text(
            "Укажите номер документа из списка.",
            reply_markup=_build_keyboard(is_admin),
        )
        return True
    file_path = Path(doc_options[idx])
    if not file_path.exists():
        await update.message.reply_text(
            "Файл не найден. Попробуйте обновить список документов.",
            reply_markup=_build_keyboard(is_admin),
        )
        return True
    with file_path.open("rb") as fh:
        await update.message.reply_document(
            document=fh,
            filename=file_path.name,
            caption=f"Документ: {file_path.name}",
        )
    return True


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
    if user.state == "pending_consent":
        if clean_text.lower() not in CONSENT_KEYWORDS:
            return CONSENT_INSTRUCTION
        db.mark_user_consent(user.id)
        db.update_user_state(user.id, "pending_fio")
        _refresh_user(context, user.telegram_id)
        return "Спасибо! Укажите ваше полное ФИО."
    if user.state == "pending_fio":
        db.update_user_profile(user.id, fio=clean_text)
        db.update_user_state(user.id, "pending_profession")
        _refresh_user(context, user.telegram_id)
        return "Спасибо! Теперь укажите вашу должность или профессию."
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


def _format_quiz_question_text(question_text: str, options: list[str]) -> str:
    lines = ["📝 Тест по охране труда", "", question_text.strip()]
    for idx, option in enumerate(options, start=1):
        lines.append(f"{idx}) {option}")
    lines.append("")
    lines.append("Выберите вариант кнопкой ниже или введите цифру 1-4.")
    return "\n".join(lines)


async def _announce_quiz_generation(
    context: ContextTypes.DEFAULT_TYPE, chat_id: int, *, new: bool
) -> None:
    text = "Генерация тестового вопроса..." if new else "Генерация нового тестового вопроса..."
    await context.bot.send_message(chat_id=chat_id, text=text)


async def _handle_quiz_answer_selection(
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    bot_user: BotUser,
    chosen_index: int,
    reply_func,
    is_admin: bool,
) -> None:
    db = _get_database(context)
    session = db.get_quiz_session(bot_user.id)
    if not session:
        await reply_func(
            "Активный тест не найден. Нажмите «📝 Тест», чтобы начать.",
            reply_markup=_build_keyboard(is_admin),
        )
        return
    if chosen_index not in range(len(session.options)):
        await reply_func(
            "Выберите вариант ответа от 1 до 4.",
            reply_markup=_build_keyboard(is_admin),
        )
        return
    correct_answer = session.options[session.correct_index]
    explanation = session.explanation or "Изучите соответствующие нормативные требования."
    if chosen_index == session.correct_index:
        feedback = "✅ Верно! Отличная работа."
        correct_delta = 1
    else:
        feedback = (
            "❌ Неверно.\n"
            f"Правильный ответ: {session.correct_index + 1}) {correct_answer}\n"
            f"Пояснение: {explanation}"
        )
        correct_delta = 0
    total_answers = session.questions_answered + 1
    total_correct = session.correct_answers + correct_delta
    feedback += f"\nСтатистика: {total_correct} из {total_answers} ответов верны."
    await reply_func(
        feedback,
        reply_markup=_build_keyboard(is_admin),
    )
    db.update_quiz_stats(
        bot_user.id, answered_delta=1, correct_delta=correct_delta
    )
    await _announce_quiz_generation(context, chat_id, new=False)
    await _send_quiz_question(chat_id, None, context, bot_user)


async def _send_quiz_question(
    chat_id: int,
    update: Update | None,
    context: ContextTypes.DEFAULT_TYPE,
    bot_user: BotUser,
) -> None:
    service: ChatService = context.application.bot_data["chat_service"]
    config: Config = context.application.bot_data["config"]
    is_admin = bot_user.telegram_id in config.admin_ids
    db = _get_database(context)
    existing = db.get_quiz_session(bot_user.id)
    try:
        question = await service.generate_quiz_question(chat_id, bot_user)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Failed to generate quiz question")
        message = f"Не удалось сгенерировать вопрос: {exc}"
        if update and update.message:
            await update.message.reply_text(
                message, reply_markup=_build_keyboard(is_admin)
            )
        else:
            await context.bot.send_message(
                chat_id=chat_id, text=message, reply_markup=_build_keyboard(is_admin)
            )
        return

    db.set_quiz_session(
        bot_user.id,
        question.question,
        question.options,
        question.correct_index,
        question.explanation,
        [str(path) for path in question.sources],
        questions_answered=existing.questions_answered if existing else 0,
        correct_answers=existing.correct_answers if existing else 0,
    )
    text = _format_quiz_question_text(question.question, question.options)
    keyboard = _build_quiz_keyboard(question.options)
    if update and update.message:
        await update.message.reply_text(text, reply_markup=keyboard)
    else:
        await context.bot.send_message(chat_id=chat_id, text=text, reply_markup=keyboard)


async def _handle_keyboard_shortcut(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    user_text: str,
) -> bool:
    if not user_text:
        return False
    config: Config = context.application.bot_data["config"]
    is_admin = _is_admin_user(update, config)
    if user_text == BTN_HELP:
        await help_command(update, context)
        return True
    if user_text == BTN_DOCS:
        await list_docs(update, context)
        return True
    if user_text == BTN_RESET:
        await reset(update, context)
        return True
    if user_text == BTN_QUIZ:
        await start_quiz(update, context)
        return True
    if user_text == BTN_RELOAD:
        if not is_admin:
            await update.message.reply_text(
                "Эта кнопка доступна только администратору.",
                reply_markup=_build_keyboard(is_admin),
            )
            return True
        await reload_docs(update, context)
        return True
    if user_text == BTN_STATS:
        if not is_admin:
            await update.message.reply_text(
                "Эта кнопка доступна только администратору.",
                reply_markup=_build_keyboard(is_admin),
            )
            return True
        await stats_command(update, context)
        return True
    return False


async def _try_handle_quiz_answer(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    bot_user: BotUser,
    user_text: str,
) -> bool:
    normalized = (user_text or "").strip()
    if normalized not in {"1", "2", "3", "4"}:
        return False
    chosen_index = int(normalized) - 1
    config: Config = context.application.bot_data["config"]
    is_admin = _is_admin_user(update, config)
    await _handle_quiz_answer_selection(
        update.effective_chat.id,
        context,
        bot_user,
        chosen_index,
        update.message.reply_text,
        is_admin,
    )
    return True


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    service: ChatService = context.application.bot_data["chat_service"]
    user_text = update.message.text or ""
    config: Config = context.application.bot_data["config"]
    bot_user = _get_user(update, context)
    if await _try_handle_doc_request(update, context, user_text):
        return
    if await _handle_keyboard_shortcut(update, context, user_text):
        return
    if not bot_user.is_active:
        response = _process_registration_step(bot_user, user_text, context)
        is_admin = _is_admin_user(update, config)
        await update.message.reply_text(
            response,
            reply_markup=_build_keyboard(is_admin),
        )
        return
    if await _try_handle_quiz_answer(update, context, bot_user, user_text):
        return
    processing_message = await update.message.reply_text("Ваш запрос обрабатывается...")
    try:
        reply, ctxs = await service.answer_text(update.effective_chat.id, bot_user, user_text)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Text handler failed")
        await update.message.reply_text(
            f"Ошибка: {exc}",
            reply_markup=_build_keyboard(_is_admin_user(update, config)),
        )
        with suppress(TelegramError):
            await processing_message.delete()
        return
    footer = _format_context_footer(ctxs)
    await update.message.reply_text(
        reply + footer,
        reply_markup=_build_keyboard(_is_admin_user(update, config)),
    )
    with suppress(TelegramError):
        await processing_message.delete()


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    service: ChatService = context.application.bot_data["chat_service"]
    bot_user = _get_user(update, context)
    config: Config = context.application.bot_data["config"]
    is_admin = _is_admin_user(update, config)
    if not bot_user.is_active:
        await update.message.reply_text(
            "Сначала завершите регистрацию: подтвердите согласие на обработку данных и отправьте ФИО и должность.",
            reply_markup=_build_keyboard(is_admin),
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
        await update.message.reply_text(
            f"Не удалось распознать голос: {exc}",
            reply_markup=_build_keyboard(is_admin),
        )
        return
    await update.message.reply_text(
        f"Распознанный текст: {transcription.text}",
        reply_markup=_build_keyboard(is_admin),
    )
    processing_message = await update.message.reply_text("Ваш запрос обрабатывается...")
    try:
        reply, ctxs = await service.answer_text(
            update.effective_chat.id, bot_user, transcription.text
        )
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("LLM failed after voice")
        await update.message.reply_text(
            f"Ошибка при обращении к модели: {exc}",
            reply_markup=_build_keyboard(is_admin),
        )
        with suppress(TelegramError):
            await processing_message.delete()
        return
    footer = _format_context_footer(ctxs)
    await update.message.reply_text(
        reply + footer,
        reply_markup=_build_keyboard(is_admin),
    )
    with suppress(TelegramError):
        await processing_message.delete()


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    service: ChatService = context.application.bot_data["chat_service"]
    bot_user = _get_user(update, context)
    config: Config = context.application.bot_data["config"]
    is_admin = _is_admin_user(update, config)
    if not bot_user.is_active:
        await update.message.reply_text(
            "Сначала завершите регистрацию: подтвердите согласие на обработку данных и отправьте ФИО и должность.",
            reply_markup=_build_keyboard(is_admin),
        )
        return
    photos = update.message.photo
    if not photos:
        await update.message.reply_text("Не удалось получить фотографию.")
        return
    best_photo = photos[-1]
    telegram_file = await best_photo.get_file()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        image_path = Path(tmp_file.name)
    await telegram_file.download_to_drive(image_path)
    caption = update.message.caption or ""
    processing_message = await update.message.reply_text("Ваш запрос обрабатывается...")
    try:
        reply, ctxs = await service.answer_image(
            update.effective_chat.id, bot_user, image_path, caption
        )
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Image handler failed")
        await update.message.reply_text(
            f"Ошибка при обработке изображения: {exc}",
            reply_markup=_build_keyboard(is_admin),
        )
        with suppress(TelegramError):
            await processing_message.delete()
        return
    finally:
        with suppress(FileNotFoundError):
            image_path.unlink()
    footer = _format_context_footer(ctxs)
    await update.message.reply_text(
        reply + footer,
        reply_markup=_build_keyboard(is_admin),
    )
    with suppress(TelegramError):
        await processing_message.delete()


async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    config: Config = context.application.bot_data["config"]
    is_admin = _is_admin_user(update, config)
    await update.message.reply_text(
        "Не понимаю этот формат сообщения. Используйте текст, голос или фото.",
        reply_markup=_build_keyboard(is_admin),
    )


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

    application = (
        ApplicationBuilder()
        .token(config.telegram_token)
        .concurrent_updates(True)
        .build()
    )
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
    application.add_handler(CallbackQueryHandler(handle_consent_callback, pattern="^consent_"))
    application.add_handler(
        CallbackQueryHandler(handle_quiz_callback, pattern="^(quiz_answer_|quiz_finish)")
    )
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document_upload))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
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
