"""Wrapper around whisper.cpp CLI."""
from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os

LOGGER = logging.getLogger(__name__)


@dataclass
class WhisperResult:
    text: str
    language: Optional[str]
    raw_json: dict


class WhisperCli:
    """Simplified whisper.cpp CLI client."""

    def __init__(
        self,
        binary_path: Path,
        model_path: Path,
        *,
        language: str = "ru",
        threads: int = 4,
        ld_library_path: Optional[str] = None,
    ) -> None:
        self.binary_path = binary_path
        self.model_path = model_path
        self.language = language
        self.threads = threads
        self.ld_library_path = ld_library_path

        if not self.binary_path.exists():
            raise FileNotFoundError(f"whisper-cli not found at {self.binary_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Whisper model not found at {self.model_path}")

    @property
    def _env(self) -> dict:
        env = dict(**os.environ)
        if self.ld_library_path:
            env["LD_LIBRARY_PATH"] = self.ld_library_path
        return env

    def transcribe(self, audio_path: Path) -> WhisperResult:
        """Transcribe audio via whisper.cpp cli."""

        audio_path = audio_path.resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio for transcription not found: {audio_path}")

        with tempfile.TemporaryDirectory(prefix="ai_omg_whisper_") as tmp_dir:
            tmp_prefix = Path(tmp_dir) / "result"
            cmd = [
                str(self.binary_path),
                "-m",
                str(self.model_path),
                "-f",
                str(audio_path),
                "-l",
                self.language,
                "-t",
                str(self.threads),
                "-oj",
                "-of",
                str(tmp_prefix),
                "-np",
            ]
            LOGGER.debug("Running whisper-cli: %s", " ".join(cmd))
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=self._env,
                )
            except subprocess.CalledProcessError as exc:
                LOGGER.error(
                    "whisper-cli failed: %s",
                    exc.stderr.decode("utf-8", errors="ignore"),
                )
                raise RuntimeError("Failed to transcribe audio") from exc

            json_path = Path(f"{tmp_prefix}.json")
            if not json_path.exists():
                raise RuntimeError("whisper-cli finished but JSON result not found")

            with open(json_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)

        text = " ".join(
            segment.get("text", "").strip()
            for segment in payload.get("transcription", [])
        ).strip()
        language = payload.get("result", {}).get("language")
        return WhisperResult(text=text, language=language, raw_json=payload)
