from __future__ import annotations

import json
import subprocess
from pathlib import Path


class WhisperError(RuntimeError):
    pass


def transcribe(
    audio_file: Path,
    whisper_bin: Path,
    model_path: Path,
    language: str = "ru",
    threads: int = 8,
) -> str:
    result_path = audio_file.with_suffix(".json")
    cmd = [
        str(whisper_bin),
        "-m",
        str(model_path),
        "-f",
        str(audio_file),
        "-l",
        language,
        "-t",
        str(threads),
        "-of",
        str(result_path),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:  # pragma: no cover - system dependency
        raise WhisperError("Не найден бинарник whisper.cpp. Соберите проект по инструкции.") from exc
    except subprocess.CalledProcessError as exc:  # pragma: no cover - runtime error
        raise WhisperError(exc.stderr.decode("utf-8", errors="ignore")) from exc

    if not result_path.exists():  # pragma: no cover - defensive
        raise WhisperError("whisper.cpp не сохранил результат")

    data = json.loads(result_path.read_text(encoding="utf-8"))
    text = data.get("text")
    if not text and "segments" in data:
        text = " ".join(seg.get("text", "") for seg in data["segments"])

    result_path.unlink(missing_ok=True)
    if not text:
        raise WhisperError("Не удалось распознать речь")
    return text.strip()
