"""Very small utility to retrieve normative documents snippets."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

LOGGER = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".txt", ".md", ".rtf"}


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
            try:
                raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning("Failed to load %s: %s", file_path, exc)
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

    @staticmethod
    def _split_into_chunks(text: str, max_len: int = 1200) -> Iterable[str]:
        paragraphs = re.split(r"\n{2,}", text)
        buffer = []
        length = 0
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            buffer.append(paragraph)
            length += len(paragraph)
            if length >= max_len:
                yield "\n\n".join(buffer)
                buffer = []
                length = 0
        if buffer:
            yield "\n\n".join(buffer)

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
            return "0 documents loaded. Place .txt/.md files into the knowledge_base folder."
        files = sorted({chunk.path for chunk in self.chunks})
        return "Loaded files:\n" + "\n".join(f"- {path.name}" for path in files)

    def document_count(self) -> int:
        if not self.chunks:
            return 0
        return len({chunk.path for chunk in self.chunks})
