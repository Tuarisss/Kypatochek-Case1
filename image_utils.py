"""Image helper utilities for multimodal prompts."""
from __future__ import annotations

import base64
import mimetypes
from pathlib import Path


def image_file_to_data_url(image_path: Path) -> str:
    """Convert image file to inline data URL for LM Studio."""

    image_path = image_path.resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type:
        suffix = image_path.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}:
            mime_type = f"image/{suffix.lstrip('.')}"
        else:
            mime_type = "image/jpeg"

    data = image_path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
