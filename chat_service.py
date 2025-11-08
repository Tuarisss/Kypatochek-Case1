"""High level orchestration for Telegram handlers."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from .audio_utils import convert_ogg_to_wav
from .config import Config
from .conversation import ConversationManager
from .db import BotDatabase, BotUser
from .document_store import DocumentStore, DocumentChunk
from .lm_client import LMStudioClient
from .whisper_client import WhisperCli, WhisperResult

LOGGER = logging.getLogger(__name__)


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
