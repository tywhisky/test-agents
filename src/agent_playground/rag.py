from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
import re
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class RagChunk:
    id: str
    content: str
    document_id: str
    namespace: str
    heading_path: str | None = None
    start: int = 0
    end: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def convert_to_markdown(path: str) -> str:
    """Convert a supported document to Markdown or fall back to plain text."""
    source = Path(path)
    if not source.exists():
        return ""

    markitdown_text = _try_markitdown(source)
    if markitdown_text:
        return markitdown_text
    return _fallback_text_reader(source)


def split_paragraphs_with_headings(text: str) -> list[dict[str, Any]]:
    """Split Markdown text into paragraphs while preserving heading ancestry."""
    lines = text.splitlines()
    heading_stack: list[str] = []
    paragraphs: list[dict[str, Any]] = []
    buffer: list[str] = []
    char_pos = 0

    def flush_buffer(end_pos: int) -> None:
        nonlocal buffer
        if not buffer:
            return
        content = "\n".join(buffer).strip()
        buffer = []
        if not content:
            return
        paragraphs.append(
            {
                "content": content,
                "heading_path": " > ".join(heading_stack) if heading_stack else None,
                "start": max(0, end_pos - len(content)),
                "end": end_pos,
            }
        )

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            flush_buffer(char_pos)
            level = len(stripped) - len(stripped.lstrip("#"))
            title = stripped.lstrip("#").strip()
            if title:
                if level <= len(heading_stack):
                    heading_stack = heading_stack[: level - 1]
                heading_stack.append(title)
            char_pos += len(line) + 1
            continue

        if not stripped:
            flush_buffer(char_pos)
        else:
            buffer.append(line)
        char_pos += len(line) + 1

    flush_buffer(char_pos)

    if not paragraphs and text.strip():
        paragraphs.append(
            {"content": text.strip(), "heading_path": None, "start": 0, "end": len(text)}
        )
    return paragraphs


def chunk_markdown(
    text: str,
    chunk_tokens: int = 300,
    overlap_tokens: int = 50,
) -> list[dict[str, Any]]:
    """Build retrieval chunks from Markdown paragraphs."""
    paragraphs = split_paragraphs_with_headings(text)
    if not paragraphs:
        return []
    return chunk_paragraphs(paragraphs, chunk_tokens, overlap_tokens)


def chunk_paragraphs(
    paragraphs: list[dict[str, Any]],
    chunk_tokens: int,
    overlap_tokens: int,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0
    index = 0

    while index < len(paragraphs):
        paragraph = paragraphs[index]
        paragraph_tokens = approx_token_len(paragraph["content"]) or 1

        if current_tokens + paragraph_tokens <= chunk_tokens or not current:
            current.append(paragraph)
            current_tokens += paragraph_tokens
            index += 1
            continue

        chunks.append(_build_chunk(current))
        current, current_tokens = _overlap_tail(current, overlap_tokens)

    if current:
        chunks.append(_build_chunk(current))

    return chunks


def approx_token_len(text: str) -> int:
    """Estimate token count for mixed scripts without external tokenizers."""
    cjk = sum(1 for char in text if is_cjk(char))
    non_cjk_tokens = len(re.findall(r"[A-Za-z0-9_]+", text))
    return cjk + non_cjk_tokens


def is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
        or 0xF900 <= code <= 0xFAFF
    )


class InMemoryVectorStore:
    """Small lexical vector store for local RAG examples and tests."""

    def __init__(self) -> None:
        self._chunks: list[RagChunk] = []

    def add_chunks(
        self,
        chunks: list[dict[str, Any]],
        namespace: str = "default",
        document_id: str | None = None,
    ) -> list[RagChunk]:
        stored_chunks = []
        resolved_document_id = document_id or "document"
        for chunk in chunks:
            chunk_document_id = str(chunk.get("document_id") or resolved_document_id)
            stored = RagChunk(
                id=str(chunk.get("id") or uuid4()),
                content=str(chunk["content"]),
                document_id=chunk_document_id,
                namespace=namespace,
                heading_path=chunk.get("heading_path"),
                start=int(chunk.get("start", 0)),
                end=int(chunk.get("end", 0)),
                metadata=dict(chunk.get("metadata", {})),
            )
            self._chunks.append(stored)
            stored_chunks.append(stored)
        return stored_chunks

    def search(
        self,
        query: str,
        limit: int = 5,
        namespace: str | None = None,
        min_score: float = 0.0,
        expansions: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        queries = [query] + list(expansions or [])
        best_hits: dict[str, dict[str, Any]] = {}

        for candidate_query in queries:
            query_vector = text_to_vector(candidate_query)
            for chunk in self._chunks:
                if namespace and chunk.namespace != namespace:
                    continue
                score = cosine_similarity(query_vector, text_to_vector(chunk.content))
                if score < min_score:
                    continue
                hit = _chunk_to_hit(chunk, score)
                if chunk.id not in best_hits or score > best_hits[chunk.id]["score"]:
                    best_hits[chunk.id] = hit

        hits = list(best_hits.values())
        hits.sort(key=lambda item: item["score"], reverse=True)
        return hits[:limit]

    def stats(self, namespace: str | None = None) -> dict[str, int]:
        chunks = [
            chunk
            for chunk in self._chunks
            if namespace is None or chunk.namespace == namespace
        ]
        return {
            "documents": len({chunk.document_id for chunk in chunks}),
            "chunks": len(chunks),
        }


def text_to_vector(text: str) -> Counter[str]:
    return Counter(tokenize(text))


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in re.finditer(r"[A-Za-z0-9_]+", text)]


def cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    shared = set(left) & set(right)
    dot_product = sum(left[token] * right[token] for token in shared)
    left_norm = sqrt(sum(value * value for value in left.values()))
    right_norm = sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)


def _try_markitdown(source: Path) -> str:
    try:
        from markitdown import MarkItDown
    except ImportError:
        return ""

    try:
        result = MarkItDown().convert(str(source))
    except Exception:
        return ""
    text = getattr(result, "text_content", "")
    return text.strip() if isinstance(text, str) else ""


def _fallback_text_reader(source: Path) -> str:
    try:
        return source.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return source.read_text(encoding="latin-1")
    except OSError:
        return ""


def _build_chunk(paragraphs: list[dict[str, Any]]) -> dict[str, Any]:
    content = "\n\n".join(paragraph["content"] for paragraph in paragraphs)
    heading_path = next(
        (
            paragraph["heading_path"]
            for paragraph in reversed(paragraphs)
            if paragraph.get("heading_path")
        ),
        None,
    )
    return {
        "content": content,
        "start": paragraphs[0]["start"],
        "end": paragraphs[-1]["end"],
        "heading_path": heading_path,
    }


def _overlap_tail(
    paragraphs: list[dict[str, Any]],
    overlap_tokens: int,
) -> tuple[list[dict[str, Any]], int]:
    if overlap_tokens <= 0:
        return [], 0

    kept: list[dict[str, Any]] = []
    kept_tokens = 0
    for paragraph in reversed(paragraphs):
        paragraph_tokens = approx_token_len(paragraph["content"]) or 1
        if kept_tokens + paragraph_tokens > overlap_tokens:
            break
        kept.append(paragraph)
        kept_tokens += paragraph_tokens
    kept.reverse()
    if len(kept) == len(paragraphs):
        return [], 0
    return kept, kept_tokens


def _chunk_to_hit(chunk: RagChunk, score: float) -> dict[str, Any]:
    return {
        "id": chunk.id,
        "score": score,
        "content": chunk.content,
        "document_id": chunk.document_id,
        "heading_path": chunk.heading_path,
        "start": chunk.start,
        "end": chunk.end,
        "metadata": dict(chunk.metadata),
    }
