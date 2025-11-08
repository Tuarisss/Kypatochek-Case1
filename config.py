from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(slots=True)
class Settings:
    bot_token: str
    lm_studio_url: str
    lm_studio_model: str
    whisper_bin: Path
    whisper_model: Path
    admin_ids: tuple[int, ...]
    enable_tts: bool
    reindex_url: str
    database_path: Path
    voice_temp_dir: Path
    log_dir: Path
    docs_dir: Path


def load_config() -> Settings:
    load_dotenv()

    admin_ids_raw = os.getenv("ADMIN_IDS", "")
    admin_ids = tuple(
        int(_id.strip())
        for _id in admin_ids_raw.split(",")
        if _id.strip().isdigit()
    )

    voice_dir = Path(os.getenv("VOICE_TEMP_DIR", "data/voices")).resolve()
    log_dir = Path(os.getenv("LOG_DIR", "data/logs")).resolve()
    docs_dir = Path(os.getenv("DOCS_DIR", "data/docs")).resolve()
    db_path = Path(os.getenv("DATABASE_PATH", "data/teacher_bot.db")).resolve()

    voice_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    return Settings(
        bot_token=os.environ.get("BOT_TOKEN", ""),
        lm_studio_url=os.getenv(
            "LM_STUDIO_URL", "http://127.0.0.1:1234/v1/chat/completions"
        ),
        lm_studio_model=os.getenv("LM_STUDIO_MODEL", "mistral"),
        whisper_bin=Path(os.getenv("WHISPER_BIN", "./whisper.cpp/main")).resolve(),
        whisper_model=Path(
            os.getenv("WHISPER_MODEL", "./whisper.cpp/models/ggml-base-q5_1.bin")
        ).resolve(),
        admin_ids=admin_ids,
        enable_tts=os.getenv("ENABLE_TTS", "false").lower() == "true",
        reindex_url=os.getenv("REINDEX_URL", "http://127.0.0.1:8000/api/reindex"),
        database_path=db_path,
        voice_temp_dir=voice_dir,
        log_dir=log_dir,
        docs_dir=docs_dir,
    )
