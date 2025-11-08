from __future__ import annotations

import json
from dataclasses import dataclass

import requests


@dataclass(slots=True)
class AIClient:
    url: str
    model: str

    def ask(self, question: str, user_id: int) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Ты преподаватель по охране труда."},
                {
                    "role": "user",
                    "content": f"[user_id={user_id}] {question}",
                },
            ],
            "temperature": 0.3,
            "max_tokens": 512,
        }
        response = requests.post(self.url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Unexpected LM Studio response: {json.dumps(data)}") from exc
