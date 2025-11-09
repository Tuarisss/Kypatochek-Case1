"""High level orchestration for Telegram handlers."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .audio_utils import convert_ogg_to_wav
from .config import Config
from .conversation import ConversationManager
from .db import BotDatabase, BotUser
from .document_store import DocumentStore, DocumentChunk
from .image_utils import image_file_to_data_url
from .lm_client import LMStudioClient
from .whisper_client import WhisperCli, WhisperResult

LOGGER = logging.getLogger(__name__)


@dataclass
class QuizQuestion:
    question: str
    options: list[str]
    correct_index: int
    explanation: str
    sources: list[Path]


class ChatService:
    def __init__(
        self,
        config: Config,
        lm_client: LMStudioClient,
        database: BotDatabase,
        document_store: DocumentStore,
        conversation: ConversationManager,
        whisper_client: WhisperCli,
    ) -> None:
        self.config = config
        self.lm_client = lm_client
        self.database = database
        self.document_store = document_store
        self.conversation = conversation
        self.whisper_client = whisper_client

    async def answer_text(
        self, chat_id: int, user: BotUser, text: str
    ) -> tuple[str, list[DocumentChunk]]:
        contexts = self.document_store.search(text)
        messages = self.conversation.build_messages(
            chat_id,
            text,
            self.config.system_prompt,
            contexts,
        )
        reply = await asyncio.to_thread(self.lm_client.chat, messages)
        self.conversation.update(chat_id, text, reply)
        self.database.log_interaction(user.id, text, reply)
        self.database.update_last_active(user.id)
        for chunk in contexts:
            self.database.log_document_usage(user.id, chunk.path)
        return reply, contexts

    async def answer_image(
        self, chat_id: int, user: BotUser, image_path: Path, caption: str | None
    ) -> tuple[str, list[DocumentChunk]]:
        query_text = (caption or "").strip() or "Проанализируй это изображение в контексте охраны труда."
        contexts = self.document_store.search(query_text)
        messages = self.conversation.build_messages(
            chat_id,
            query_text,
            self.config.system_prompt,
            contexts,
        )
        image_data_url = await asyncio.to_thread(image_file_to_data_url, image_path)
        messages[-1]["content"] = [
            {"type": "text", "text": query_text},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]
        reply = await asyncio.to_thread(self.lm_client.chat, messages)
        prompt_label = f"[Фото] {query_text}"
        self.conversation.update(chat_id, prompt_label, reply)
        self.database.log_interaction(user.id, prompt_label, reply)
        self.database.update_last_active(user.id)
        for chunk in contexts:
            self.database.log_document_usage(user.id, chunk.path)
        return reply, contexts

    async def transcribe_voice(self, ogg_path: Path) -> WhisperResult:
        wav_path = await asyncio.to_thread(
            convert_ogg_to_wav,
            ogg_path,
            ffmpeg_binary=self.config.ffmpeg_binary,
        )
        try:
            result = await asyncio.to_thread(self.whisper_client.transcribe, wav_path)
        finally:
            for path in (wav_path, ogg_path):
                try:
                    path.unlink(missing_ok=True)
                except FileNotFoundError:
                    pass
        if not result.text:
            raise RuntimeError("Whisper returned empty transcript")
        LOGGER.info("Voice transcription detected language %s", result.language)
        return result

    async def generate_quiz_question(self, chat_id: int, user: BotUser) -> QuizQuestion:
        contexts = self.document_store.sample_chunks(2)
        context_text = ""
        source_paths: list[Path] = []
        if contexts:
            snippets = []
            for idx, chunk in enumerate(contexts, start=1):
                snippets.append(f"[{idx}] {chunk.text.strip()}")
                source_paths.append(chunk.path)
            context_text = "\n\n".join(snippets)
        else:
            context_text = (
                "Общие требования: инструктажи по охране труда, СИЗ, ответственность работодателя, "
                "порядок обучения и проверки знаний."
            )
        prompt = (
            "Сгенерируй один контрольный вопрос по охране труда на основе контекста. "
            "Вопрос должен иметь ровно четыре варианта ответа (options), только один из них верный. "
            "Верни строго JSON без пояснений:\n"
            '{"question":"...","options":["...","...","...","..."],"correct_index":0,"explanation":"..."}\n'
            "correct_index использует индексацию с нуля. explanation коротко поясняет правильный ответ.\n\n"
            f"Контекст:\n{context_text}"
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "Ты преподаватель по охране труда. Создавай тестовые вопросы, используя только переданный контекст."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        raw = await asyncio.to_thread(self.lm_client.chat, messages)
        data = self._parse_quiz_json(raw)
        options = [opt.strip() for opt in data.get("options", []) if opt.strip()]
        if len(options) != 4:
            raise RuntimeError("Модель вернула неверное количество вариантов ответа.")
        correct_index = int(data.get("correct_index", -1))
        if correct_index not in range(4):
            raise RuntimeError("Модель вернула некорректный индекс правильного ответа.")
        explanation = data.get("explanation", "").strip()
        question = data.get("question", "").strip()
        if not question:
            raise RuntimeError("Модель вернула пустой вопрос.")

        self.database.log_interaction(
            user.id,
            "[Квиз] генерация",
            f"{question} | {options}",
        )
        self.database.update_last_active(user.id)
        for path in source_paths:
            self.database.log_document_usage(user.id, path)

        return QuizQuestion(
            question=question,
            options=options,
            correct_index=correct_index,
            explanation=explanation,
            sources=source_paths,
        )

    @staticmethod
    def _parse_quiz_json(raw_text: str) -> dict:
        text = raw_text.strip()
        candidates = [text]
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                part = part.strip()
                if part.lower().startswith("json"):
                    candidates.append(part[4:].strip())
                else:
                    candidates.append(part)
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        raise RuntimeError("Не удалось разобрать JSON с тестовым вопросом.")
