"""Conversation history helpers."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, List

from .document_store import DocumentChunk


class ConversationManager:
    def __init__(self, max_messages: int = 10) -> None:
        self.max_messages = max_messages
        self._history: Dict[int, Deque[dict]] = defaultdict(deque)

    def build_messages(
        self,
        chat_id: int,
        user_text: str,
        system_prompt: str,
        context_chunks: List[DocumentChunk] | None = None,
    ) -> List[dict]:
        history = list(self._history[chat_id])
        context_text = ""
        if context_chunks:
            snippets = []
            for idx, chunk in enumerate(context_chunks, start=1):
                snippets.append(
                    f"[{idx}] Источник: {chunk.path.name}\n{chunk.text.strip()}"
                )
            context_text = "\n\nДоступные выдержки из нормативной базы:\n" + "\n---\n".join(snippets)
        system_content = system_prompt + context_text
        messages = [{"role": "system", "content": system_content}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_text})
        return messages

    def update(self, chat_id: int, user_text: str, assistant_text: str) -> None:
        history = self._history[chat_id]
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": assistant_text})
        excess = len(history) - (self.max_messages * 2)
        if excess > 0:
            for _ in range(excess):
                history.popleft()

    def reset(self, chat_id: int) -> None:
        self._history.pop(chat_id, None)
