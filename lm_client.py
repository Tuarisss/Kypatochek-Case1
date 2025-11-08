"""Client for LM Studio compatible local models."""
from __future__ import annotations

import logging
from typing import List

import requests

LOGGER = logging.getLogger(__name__)


class LMStudioClient:
    def __init__(
        self,
        api_url: str,
        model_name: str,
        *,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        timeout: int = 120,
    ) -> None:
        self.api_url = api_url
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._session = requests.Session()

    def chat(self, messages: List[dict]) -> str:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        LOGGER.debug("Sending prompt with %s messages", len(messages))
        response = self._session.post(
            self.api_url,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:  # pylint: disable=broad-except
            LOGGER.error("Unexpected LM Studio response: %s", data)
            raise RuntimeError("LM Studio response is missing message content") from exc
        return content
