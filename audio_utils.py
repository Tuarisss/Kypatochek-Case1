"""Audio helper utilities."""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger(__name__)


def convert_ogg_to_wav(
    input_path: Path,
    *,
    ffmpeg_binary: str = "ffmpeg",
    output_path: Optional[Path] = None,
) -> Path:
    """Convert an OGG/Opus file to mono 16kHz WAV via ffmpeg."""

    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    if output_path is None:
        with tempfile.NamedTemporaryFile(prefix="ai_omg_audio_", suffix=".wav", delete=False) as tmp_file:
            output_path = Path(tmp_file.name)
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_binary,
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    LOGGER.debug("Running ffmpeg: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg binary is missing. Install ffmpeg or set FFMPEG_BIN env variable"
        ) from exc
    except subprocess.CalledProcessError as exc:
        LOGGER.error("ffmpeg failed: %s", exc.stderr.decode("utf-8", errors="ignore"))
        raise RuntimeError("ffmpeg failed to convert audio") from exc

    return output_path
