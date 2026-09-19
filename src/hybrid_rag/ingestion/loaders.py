"""Format-specific loaders that turn raw bytes into structure-preserving `Section`s.

Every loader emits the same intermediate representation - a list of sections, each carrying its
heading breadcrumb - so the chunker can be format-agnostic and structure-aware at the same time.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any

import yaml
from bs4 import BeautifulSoup, Tag

from hybrid_rag.schemas import LoadedDocument, Section

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".text": "text",
    ".pdf": "pdf",
    ".html": "html",
    ".htm": "html",
}

# Metadata keys recognised in HTML <meta> tags and plain-text header blocks.
_META_KEYS = {"title", "department", "owner", "classification", "version", "last_reviewed"}


class UnsupportedFormatError(ValueError):
    pass


class EmptyDocumentError(ValueError):
    pass


def detect_doc_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    try:
        return SUPPORTED_EXTENSIONS[ext]
    except KeyError as exc:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFormatError(f"unsupported file type '{ext}' (allowed: {allowed})") from exc


def load_document(
    filename: str, data: bytes, extra_metadata: dict[str, Any] | None = None
) -> LoadedDocument:
    """Dispatch to the right loader based on file extension."""
    doc_type = detect_doc_type(filename)
    loader = {
        "markdown": load_markdown,
        "text": load_text,
        "pdf": load_pdf,
        "html": load_html,
    }[doc_type]
    doc = loader(filename, data)
    if extra_metadata:
        doc.metadata.update({k: v for k, v in extra_metadata.items() if v not in (None, "")})
        if extra_metadata.get("title"):
            doc.title = str(extra_metadata["title"])
    # JSON round-trip: YAML dates etc. become strings, so stored and freshly loaded metadata
    # compare equal (required for idempotent re-ingest).
    doc.metadata = json.loads(json.dumps(doc.metadata, default=str))
    doc.sections = [s for s in doc.sections if s.text.strip()]
    if not doc.sections:
        raise EmptyDocumentError(f"no extractable text in '{filename}'")
    return doc


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _title_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    return re.sub(r"[-_]+", " ", stem).strip().title() or filename


# ---------------------------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------------------------
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return {}, text
    if not isinstance(meta, dict):
        return {}, text
    return {str(k): v for k, v in meta.items()}, text[match.end() :]


def markdown_sections(body: str) -> tuple[str | None, list[Section]]:
    """Split markdown into sections keyed by their heading breadcrumb (code fences respected)."""
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    in_fence = False
    h1: str | None = None

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            sections.append(Section(heading_path=[h for _, h in stack], text=text))
        buf.clear()

    for line in body.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            buf.append(line)
            continue
        heading = None if in_fence else _HEADING_RE.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            if level == 1 and h1 is None:
                h1 = title
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            buf.append(line)
    flush()
    # The document title (single H1) is stored as doc metadata; drop it from breadcrumbs.
    if h1 is not None:
        for s in sections:
            if s.heading_path and s.heading_path[0] == h1:
                s.heading_path = s.heading_path[1:]
    return h1, sections


def load_markdown(filename: str, data: bytes) -> LoadedDocument:
    meta, body = parse_frontmatter(_decode(data))
    h1, sections = markdown_sections(body)
    title = str(meta.pop("title", None) or h1 or _title_from_filename(filename))
    return LoadedDocument(
        source=filename, title=title, doc_type="markdown", sections=sections, metadata=meta
    )


# ---------------------------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------------------------
_UNDERLINE_RE = re.compile(r"^(=+|-+)\s*$")


_HEADER_FIELD_RE = re.compile(r"^([A-Za-z][A-Za-z _-]{1,30}):\s*(.+?)\s*$")


def load_text(filename: str, data: bytes) -> LoadedDocument:
    """Plain text. Recognises setext-style (underlined) and ALL-CAPS lines as headings, and a
    leading ``Key: value`` header block (e.g. ``Department: Finance``) as metadata."""
    text = _decode(data).replace("\r\n", "\n")
    lines = text.split("\n")
    meta: dict[str, Any] = {}
    sections: list[Section] = []
    current: list[str] = []
    heading: str | None = None
    title: str | None = None

    def flush() -> None:
        body = "\n".join(current).strip()
        if body:
            sections.append(Section(heading_path=[heading] if heading else [], text=body))
        current.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        stripped = line.strip()
        is_setext = bool(stripped) and bool(_UNDERLINE_RE.match(nxt.strip())) and len(nxt) >= 3
        is_caps = (
            bool(stripped)
            and stripped.isupper()
            and 2 <= len(stripped.split()) <= 10
            and not stripped.endswith((".", ","))
        )
        field = (
            _HEADER_FIELD_RE.match(stripped)
            if not sections and not "".join(current).strip()
            else None
        )
        if field and field.group(1).strip().lower().replace(" ", "_") in _META_KEYS:
            meta[field.group(1).strip().lower().replace(" ", "_")] = field.group(2)
            i += 1
            continue
        if is_setext or is_caps:
            flush()
            name = stripped.title() if is_caps else stripped
            if title is None and (is_setext and nxt.strip().startswith("=")):
                title = name
            else:
                heading = name
            i += 2 if is_setext else 1
            continue
        current.append(line)
        i += 1
    flush()
    return LoadedDocument(
        source=filename,
        title=title or _title_from_filename(filename),
        doc_type="text",
        sections=sections,
        metadata=meta,
    )


# ---------------------------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------------------------
_BLOCK_TAGS = ("p", "li", "pre", "blockquote", "tr", "dd", "dt", "figcaption")
_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")


def load_html(filename: str, data: bytes) -> LoadedDocument:
    soup = BeautifulSoup(_decode(data), "html.parser")
    for bad in soup(["script", "style", "noscript", "nav", "footer", "header", "svg", "form"]):
        bad.decompose()

    meta: dict[str, Any] = {}
    for m in soup.find_all("meta"):
        name, content = m.get("name"), m.get("content")
        if isinstance(name, str) and isinstance(content, str) and name in _META_KEYS:
            meta[name] = content

    page_title = soup.title.get_text(strip=True) if soup.title else None
    root = soup.find("main") or soup.find("article") or soup.body or soup
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    h1: str | None = None

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            sections.append(Section(heading_path=[h for _, h in stack], text=text))
        buf.clear()

    for el in root.find_all([*_HEADING_TAGS, *_BLOCK_TAGS]) if isinstance(root, Tag) else []:
        if el.name in _HEADING_TAGS:
            flush()
            level = int(el.name[1])
            title = el.get_text(" ", strip=True)
            if level == 1 and h1 is None:
                h1 = title
                continue
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            continue
        # Skip blocks nested in another block we already capture (e.g. <p> inside <li>).
        if el.find_parent(_BLOCK_TAGS) is not None:
            continue
        if el.name == "tr":
            cells = [c.get_text(" ", strip=True) for c in el.find_all(["td", "th"])]
            text = " | ".join(c for c in cells if c)
        else:
            text = el.get_text(" ", strip=True)
        if not text:
            continue
        if el.name == "li":
            text = f"- {text}"
        elif el.name == "pre":
            text = el.get_text("\n", strip=False).strip("\n")
        buf.append(text if el.name in ("li", "tr") else f"{text}\n")
    flush()

    title = str(meta.pop("title", None) or h1 or page_title or _title_from_filename(filename))
    return LoadedDocument(
        source=filename, title=title, doc_type="html", sections=sections, metadata=meta
    )


# ---------------------------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------------------------
def load_pdf(filename: str, data: bytes) -> LoadedDocument:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
    except PdfReadError as exc:
        raise EmptyDocumentError(f"could not parse PDF '{filename}': {exc}") from exc

    meta: dict[str, Any] = {}
    info = reader.metadata
    pdf_title = info.title if info and info.title else None
    if info and info.author:
        meta["owner"] = str(info.author)
    if info and info.subject:
        meta["department"] = str(info.subject)

    sections: list[Section] = []
    for page_no, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        text = _clean_pdf_text(raw)
        if text:
            sections.append(Section(heading_path=[f"Page {page_no}"], text=text, page=page_no))
    return LoadedDocument(
        source=filename,
        title=str(pdf_title or _title_from_filename(filename)),
        doc_type="pdf",
        sections=sections,
        metadata=meta,
    )


def _clean_pdf_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # de-hyphenate line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
