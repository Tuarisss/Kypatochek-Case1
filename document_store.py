"""Very small utility to retrieve normative documents snippets."""
from __future__ import annotations

import logging
import random
import re
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError

LOGGER = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".txt", ".md", ".rtf", ".pdf"}
PDF_PAGE_LIMIT = int(os.environ.get("PDF_PAGE_LIMIT", "40"))


@dataclass
class DocumentChunk:
    path: Path
    text: str
    score: float = 0.0

    def pretty_header(self) -> str:
        rel_path = self.path.name
        return f"{rel_path} (score {self.score:.2f})"


class DocumentStore:
    """Naive full-text loader with keyword scoring."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.chunks: List[DocumentChunk] = []
        if self.root_dir.exists():
            self.reload()

    def reload(self) -> None:
        self.chunks.clear()
        for file_path in sorted(self._iter_files()):
            raw_text = self._read_file_text(file_path)
            if raw_text is None:
                continue
            for chunk in self._split_into_chunks(raw_text):
                text = chunk.strip()
                if not text:
                    continue
                self.chunks.append(DocumentChunk(path=file_path, text=text))
        LOGGER.info("Loaded %s text chunks from %s", len(self.chunks), self.root_dir)

    def _iter_files(self) -> Iterable[Path]:
        if not self.root_dir.exists():
            return []
        for path in self.root_dir.glob("**/*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield path

    def _read_file_text(self, file_path: Path) -> Optional[str]:
        try:
            if file_path.suffix.lower() == ".pdf":
                return self._read_pdf(file_path)
            return file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("Failed to load %s: %s", file_path, exc)
            return None

    @staticmethod
    def _read_pdf(file_path: Path) -> str:
        try:
            reader = PdfReader(str(file_path), strict=False)
        except PdfReadError as exc:
            LOGGER.warning("Failed to open PDF %s: %s", file_path, exc)
            return ""
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("Unexpected PDF error %s: %s", file_path, exc)
            return ""

        text_parts = []
        for page_idx, page in enumerate(reader.pages, start=1):
            if page_idx > PDF_PAGE_LIMIT:
                LOGGER.info(
                    "Truncated %s to first %s pages (set PDF_PAGE_LIMIT env var to adjust)",
                    file_path,
                    PDF_PAGE_LIMIT,
                )
                break
            try:
                text_parts.append(page.extract_text() or "")
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning(
                    "Failed to extract text from %s page %s: %s", file_path, page_idx, exc
                )
                continue
        return "\n".join(text_parts)

    @staticmethod
    def _split_into_chunks(text: str, max_len: int = 1200) -> Iterable[str]:
        paragraphs = re.split(r"\n{2,}", text)
        buffer: List[str] = []
        length = 0

        def emit() -> Optional[str]:
            nonlocal buffer, length
            if not buffer:
                return None
            chunk = "\n\n".join(buffer)
            buffer = []
            length = 0
            return chunk

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            while len(paragraph) > max_len:
                head = paragraph[:max_len]
                paragraph = paragraph[max_len:]
                buffer.append(head)
                length += len(head)
                if length >= max_len:
                    chunk = emit()
                    if chunk:
                        yield chunk
            buffer.append(paragraph)
            length += len(paragraph)
            if length >= max_len:
                chunk = emit()
                if chunk:
                    yield chunk
        if buffer:
            chunk = emit()
            if chunk:
                yield chunk

    def search(self, query: str, limit: int = 3) -> List[DocumentChunk]:
        if not query.strip() or not self.chunks:
            return []
        words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
        if not words:
            return []
        scored: List[DocumentChunk] = []
        for chunk in self.chunks:
            text_lower = chunk.text.lower()
            matches = sum(text_lower.count(word) for word in words)
            if matches == 0:
                continue
            score = matches / len(words)
            scored.append(DocumentChunk(path=chunk.path, text=chunk.text, score=score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]

    def describe(self) -> str:
        if not self.chunks:
            return "Документы: отсутствуют. Добавьте файлы в папку knowledge_base."
        files = self.list_files()
        lines = ["Документы:"]
        for idx, path in enumerate(files, start=1):
            lines.append(f"{idx}) {path.name}")
        return "\n".join(lines)

    def document_count(self) -> int:
        if not self.chunks:
            return 0
        return len({chunk.path for chunk in self.chunks})

    def list_files(self) -> List[Path]:
        return sorted({chunk.path for chunk in self.chunks})

    def sample_chunks(self, count: int = 2) -> List[DocumentChunk]:
        if not self.chunks:
            return []
        return random.sample(self.chunks, min(count, len(self.chunks)))
