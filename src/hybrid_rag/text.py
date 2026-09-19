"""Text utilities shared by chunking, BM25, reranking and citation verification.

Two distinct notions of "token" live here:

* `TokenCounter` counts *model* tokens (tiktoken BPE) - used to size chunks against LLM /
  embedding context budgets.
* `analyze()` produces normalised *lexical terms* (lower-cased, stop-words removed, lightly
  stemmed) - used by BM25, the lexical reranker and the citation verifier.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Protocol

from hybrid_rag.logging_setup import get_logger

log = get_logger(__name__)

STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further
    had has have having he her here hers herself him himself his how i if in into is it its itself
    just me more most my myself no nor not now of off on once only or other our ours ourselves out
    over own same she should so some such than that the their theirs them themselves then there
    these they this those through to too under until up very was we were what when where which
    while who whom why will with would you your yours yourself yourselves s t also may must shall
    per via etc e g ie eg one any every get got
    """.split()  # noqa: SIM905
)

_WORD_RE = re.compile(r"[a-z0-9]+(?:[-_.'][a-z0-9]+)*", re.IGNORECASE)
_NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*(?:%|\b)")


class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int: ...


class SimpleTokenCounter:
    """Dependency-free approximation of BPE token counts (~1.3 tokens per word + punctuation)."""

    name = "simple"
    _piece_re = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def count(self, text: str) -> int:
        total = 0
        for piece in self._piece_re.findall(text):
            # Long words are split into several BPE tokens by real tokenizers.
            total += 1 + len(piece) // 7 if piece[0].isalnum() else 1
        return total


class TiktokenCounter:
    name = "tiktoken"

    def __init__(self, encoding: str = "cl100k_base") -> None:
        import tiktoken

        self._enc = tiktoken.get_encoding(encoding)

    def count(self, text: str) -> int:
        return len(self._enc.encode(text, disallowed_special=()))


@lru_cache(maxsize=4)
def get_token_counter(kind: str = "tiktoken", encoding: str = "cl100k_base") -> TokenCounter:
    if kind == "tiktoken":
        try:
            return TiktokenCounter(encoding)
        except Exception as exc:  # network-less first run, unknown encoding, ...
            log.warning("tiktoken_unavailable_falling_back", error=str(exc))
    return SimpleTokenCounter()


def stem(word: str) -> str:
    """Tiny, predictable suffix stripper (a pragmatic subset of Porter step 1)."""
    if len(word) <= 3 or word.isdigit():
        return word
    for suffix, repl in (
        ("ies", "y"),
        ("sses", "ss"),
        ("ing", ""),
        ("edly", ""),
        ("ed", ""),
        ("s", ""),
    ):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            if suffix == "s" and word.endswith(("ss", "us", "is")):
                return word
            return word[: len(word) - len(suffix)] + repl
    return word


def words(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def analyze(text: str) -> list[str]:
    """Lower-case, split, drop stop-words, stem. Used for every lexical comparison."""
    return [stem(w) for w in words(text) if w not in STOPWORDS]


def numbers(text: str) -> set[str]:
    """Numeric literals in `text` (normalised: thousands separators removed)."""
    return {n.replace(",", "") for n in _NUMBER_RE.findall(text)}


_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_ABBREVIATIONS = ("e.g.", "i.e.", "etc.", "vs.", "Mr.", "Ms.", "Dr.", "No.", "approx.", "Inc.")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9*_`])")


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences; newlines (list items, table rows) are hard boundaries."""
    out: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        protected = line
        for i, abbr in enumerate(_ABBREVIATIONS):
            protected = protected.replace(abbr, f"\x00{i}\x00")
        for part in _SENTENCE_BOUNDARY_RE.split(protected):
            restored = part
            for i, abbr in enumerate(_ABBREVIATIONS):
                restored = restored.replace(f"\x00{i}\x00", abbr)
            restored = restored.strip()
            if restored:
                out.append(restored)
    return out


def strip_markdown(line: str) -> str:
    """Remove list bullets, heading hashes, emphasis and table pipes from a line of markdown."""
    line = re.sub(r"^\s{0,3}(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>\s*)", "", line)
    line = line.replace("**", "").replace("__", "").replace("`", "")
    line = re.sub(r"\s*\|\s*", " | ", line).strip(" |")
    return line.strip()


def citation_regex() -> re.Pattern[str]:
    return _CITATION_RE


_BLOCK_START_RE = re.compile(r"^\s*(?:[-*+]\s|\d+[.)]\s|\||#{1,6}\s|>|```|~~~)")


def unwrap_lines(text: str) -> list[str]:
    """Join hard-wrapped lines into logical lines (paragraphs, list items, table rows).

    Source documents are often wrapped at ~100 columns; sentence-level processing must see
    "... completes in under 3 minutes." as one sentence, not two fragments.
    """
    logical: list[str] = []
    in_fence = False
    prev_blank = True
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            logical.append(line)
            prev_blank = False
            continue
        if not line.strip():
            prev_blank = True
            continue
        is_start = in_fence or prev_blank or bool(_BLOCK_START_RE.match(line)) or not logical
        if is_start or logical[-1].lstrip().startswith(("|", "#", "```", "~~~")):
            logical.append(line)
        else:
            logical[-1] = f"{logical[-1]} {line.strip()}"
        prev_blank = False
    return logical
