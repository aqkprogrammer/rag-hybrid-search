"""Structure-aware recursive chunking with token-based sizing and sentence-aligned overlap.

Strategy (see README "Chunking strategy"):

1. Loaders emit `Section`s - text under a single heading breadcrumb (or a single PDF page).
   Chunks never span two sections, so every chunk has one unambiguous breadcrumb.
2. A section is split into *blocks* (paragraphs, list runs, fenced code blocks kept intact).
3. Blocks larger than the budget are split recursively: lines -> sentences -> words, stopping
   at the coarsest separator that fits.
4. Units are packed greedily up to `chunk_size` tokens. Each new chunk starts with an overlap
   tail of whole *sentences* from the previous chunk (<= `overlap` tokens), so context carries
   across boundaries without cutting sentences in half.
5. A tiny trailing chunk is merged into its predecessor; a tiny "intro" section that is
   immediately followed by one of its sub-sections is folded into that sub-section.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from hybrid_rag.schemas import FILTERABLE_KEYS, Chunk, LoadedDocument, Section
from hybrid_rag.text import TokenCounter, split_sentences

_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_BLANK_RE = re.compile(r"^\s*$")


@dataclass(frozen=True)
class _Unit:
    text: str
    sep: str  # separator to use *before* this unit when joined to a previous one
    tokens: int


class StructureAwareChunker:
    def __init__(
        self,
        counter: TokenCounter,
        chunk_size: int = 320,
        overlap: int = 48,
        min_tokens: int = 24,
    ) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.counter = counter
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_tokens = min_tokens

    # -- public -------------------------------------------------------------------------------
    def chunk(self, doc: LoadedDocument, doc_id: str) -> list[Chunk]:
        base_meta: dict[str, Any] = {
            "doc_id": doc_id,
            "source": doc.source,
            "title": doc.title,
            "doc_type": doc.doc_type,
        }
        for key in FILTERABLE_KEYS:
            if key in doc.metadata and key not in base_meta:
                value = doc.metadata[key]
                if isinstance(value, str | int | float | bool):
                    base_meta[key] = value

        chunks: list[Chunk] = []
        for section in self._fold_tiny_sections(doc.sections):
            for text in self.split_section(section.text):
                idx = len(chunks)
                meta = {
                    **base_meta,
                    "heading_path": " > ".join(section.heading_path),
                    "chunk_index": idx,
                }
                if section.page is not None:
                    meta["page"] = section.page
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}-{idx:04d}",
                        doc_id=doc_id,
                        index=idx,
                        text=text,
                        heading_path=list(section.heading_path),
                        token_count=self.counter.count(text),
                        page=section.page,
                        metadata=meta,
                    )
                )
        return chunks

    def split_section(self, text: str) -> list[str]:
        units: list[_Unit] = []
        for block in self._blocks(text):
            units.extend(self._split_recursive(block, level=0, sep="\n\n"))
        return self._pack(units)

    # -- internals ----------------------------------------------------------------------------
    def _fold_tiny_sections(self, sections: list[Section]) -> list[Section]:
        out: list[Section] = []
        pending: Section | None = None
        for sec in sections:
            if pending is not None:
                is_child = (
                    bool(pending.heading_path)
                    and sec.heading_path[: len(pending.heading_path)] == pending.heading_path
                    and len(sec.heading_path) > len(pending.heading_path)
                    and sec.page == pending.page
                )
                if is_child:
                    sec = Section(
                        heading_path=sec.heading_path,
                        text=f"{pending.text}\n\n{sec.text}",
                        page=sec.page,
                    )
                else:
                    out.append(pending)
                pending = None
            if self.counter.count(sec.text) < self.min_tokens:
                pending = sec
            else:
                out.append(sec)
        if pending is not None:
            out.append(pending)
        return out

    @staticmethod
    def _blocks(text: str) -> list[str]:
        """Paragraph-level blocks; fenced code blocks are atomic even if they contain blanks."""
        blocks: list[str] = []
        buf: list[str] = []
        in_fence = False
        for line in text.splitlines():
            if _FENCE_RE.match(line):
                in_fence = not in_fence
                buf.append(line)
                continue
            if not in_fence and _BLANK_RE.match(line):
                if buf:
                    blocks.append("\n".join(buf).strip("\n"))
                    buf = []
                continue
            buf.append(line)
        if buf:
            blocks.append("\n".join(buf).strip("\n"))
        return [b for b in blocks if b.strip()]

    def _split_recursive(self, text: str, level: int, sep: str) -> list[_Unit]:
        tokens = self.counter.count(text)
        if tokens <= self.chunk_size:
            return [_Unit(text, sep, tokens)]

        if level == 0:  # lines
            parts, child_sep = [p for p in text.split("\n") if p.strip()], "\n"
        elif level == 1:  # sentences
            parts, child_sep = split_sentences(text), " "
        else:  # hard word windows - last resort for pathological input
            return self._word_windows(text, sep)

        if len(parts) <= 1:
            return self._split_recursive(text, level + 1, sep)
        units: list[_Unit] = []
        for i, part in enumerate(parts):
            units.extend(self._split_recursive(part, level + 1, sep if i == 0 else child_sep))
        return units

    def _word_windows(self, text: str, sep: str) -> list[_Unit]:
        words = text.split()
        units: list[_Unit] = []
        buf: list[str] = []
        for word in words:
            candidate = " ".join([*buf, word])
            if buf and self.counter.count(candidate) > self.chunk_size:
                chunk = " ".join(buf)
                units.append(_Unit(chunk, sep if not units else " ", self.counter.count(chunk)))
                buf = [word]
            else:
                buf.append(word)
        if buf:
            chunk = " ".join(buf)
            units.append(_Unit(chunk, sep if not units else " ", self.counter.count(chunk)))
        return units

    def _pack(self, units: list[_Unit]) -> list[str]:
        chunks: list[str] = []
        current: list[_Unit] = []
        current_tokens = 0
        # Tokens contributed by the overlap tail at the head of `current`; a chunk made *only*
        # of overlap carries no new information and must never be emitted.
        overlap_tokens = 0

        for unit in units:
            projected = current_tokens + unit.tokens + (1 if current else 0)
            if current and projected > self.chunk_size and current_tokens > overlap_tokens:
                chunks.append(self._join(current))
                current = self._overlap_tail(current)
                current_tokens = sum(u.tokens for u in current)
                overlap_tokens = current_tokens
                if current and current_tokens + unit.tokens > self.chunk_size:
                    current, current_tokens, overlap_tokens = [], 0, 0
            current.append(unit)
            current_tokens += unit.tokens

        if current and current_tokens > overlap_tokens:
            text = self._join(current)
            new_tokens = current_tokens - overlap_tokens
            if chunks and new_tokens < self.min_tokens:
                # Tiny remainder: append the *new* part to the previous chunk instead.
                chunks[-1] = self._join(
                    [_Unit(chunks[-1], "", 0), *current[_n_units(current, overlap_tokens) :]]
                )
            else:
                chunks.append(text)
        return chunks

    def _overlap_tail(self, units: list[_Unit]) -> list[_Unit]:
        if self.overlap <= 0:
            return []
        sentences: list[str] = []
        for u in units[-3:]:
            sentences.extend(split_sentences(u.text))
        tail: list[_Unit] = []
        budget = self.overlap
        for sentence in reversed(sentences):
            t = self.counter.count(sentence)
            if t > budget:
                break
            tail.insert(0, _Unit(sentence, " ", t))
            budget -= t
        if tail:
            tail[0] = _Unit(tail[0].text, "", tail[0].tokens)
            # Separate the carried-over context from the new content by a paragraph break.
        return tail

    @staticmethod
    def _join(units: list[_Unit]) -> str:
        out = ""
        for i, u in enumerate(units):
            out += u.text if i == 0 else (u.sep or "\n\n") + u.text
        return out.strip()


def _n_units(units: list[_Unit], overlap_tokens: int) -> int:
    """Number of leading units in `units` that belong to the overlap tail."""
    total = 0
    for i, u in enumerate(units):
        if total >= overlap_tokens:
            return i
        total += u.tokens
    return len(units)
