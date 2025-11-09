"""Configuration helpers for the Telegram safety assistant bot."""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List, Optional


DEFAULT_SYSTEM_PROMPT = (
    "Ты – виртуальный консультант по охране труда. Отвечай кратко (до 2–3 предложений), "
    "формально, ссылаясь на нормативные документы РФ, если они известны. Если нет данных, "
    "честно сообщай, что требуется уточнить, и предложи, какие документы нужны. Помогай "
    "с обучением, тестами и вопросами по охране труда. Строго отказывайся отвечать на темы, "
    "не связанные с охраной труда или промышленной безопасностью, и никогда не предоставляй "
    "примеров программного кода, не обсуждай личные или бытовые вопросы."
)


def _get_env(name: str, default: Optional[str] = None, *, required: bool = False) -> str:
    """Fetch environment variable or raise if required."""

    value = os.environ.get(name, default)
    if required and not value:
        raise RuntimeError(f"Environment variable {name} is required")
    return value


@dataclass
class Config:
    """Runtime configuration for the bot."""

    telegram_token: str = field(default_factory=lambda: _get_env("TELEGRAM_BOT_TOKEN", required=True))
    lm_api_url: str = field(default_factory=lambda: _get_env(
        "LM_STUDIO_API", "http://localhost:1234/v1/chat/completions"
    ))
    lm_model: str = field(default_factory=lambda: _get_env("LM_STUDIO_MODEL", "qwen/qwen3-vl-8b"))
    system_prompt: str = field(default_factory=lambda: _get_env("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT))
    downloads_dir: Path = field(
        default_factory=lambda: Path(_get_env("BOT_RUNTIME_DIR", ".runtime")).resolve()
    )
    database_path: Path = field(
        default_factory=lambda: Path(
            _get_env("BOT_DB_PATH", ".runtime/bot_state.sqlite3")
        ).resolve()
    )
    knowledge_root: Path = field(
        default_factory=lambda: Path(_get_env("KNOWLEDGE_BASE_DIR", "knowledge_base")).resolve()
    )
    max_history_messages: int = int(_get_env("MAX_HISTORY_MESSAGES", "10"))
    lm_temperature: float = float(_get_env("LM_TEMPERATURE", "0.3"))
    lm_max_tokens: int = int(_get_env("LM_MAX_TOKENS", "1024"))

    ffmpeg_binary: str = field(default_factory=lambda: _get_env("FFMPEG_BIN", "ffmpeg"))
    whisper_binary: Path = field(
        default_factory=lambda: Path(_get_env("WHISPER_BIN", "whisper.cpp/build/bin/whisper-cli")).resolve()
    )
    whisper_model_path: Path = field(
        default_factory=lambda: Path(_get_env("WHISPER_MODEL", "whisper.cpp/models/ggml-small.bin")).resolve()
    )
    whisper_threads: int = int(_get_env("WHISPER_THREADS", "4"))
    whisper_language: str = field(default_factory=lambda: _get_env("WHISPER_LANGUAGE", "ru"))
    whisper_ld_library_path: Optional[str] = field(
        default_factory=lambda: _get_env("WHISPER_LD_LIBRARY_PATH", None)
    )

    admin_ids: List[int] = field(
        default_factory=lambda: [
            int(part.strip())
            for part in _get_env("TELEGRAM_ADMIN_IDS", "").split(",")
            if part.strip()
        ]
    )

    def ensure_directories(self) -> None:
        """Create directories that must exist at runtime."""

        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        self.knowledge_root.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)


_config_cache: Optional[Config] = None


def load_config() -> Config:
    """Load configuration once."""

    global _config_cache
    if _config_cache is None:
        _config_cache = Config()
        _config_cache.ensure_directories()
    return _config_cache
