from pathlib import Path

import pytest

from hybrid_rag.ingestion.loaders import (
    EmptyDocumentError,
    UnsupportedFormatError,
    load_document,
)

SAMPLE = Path(__file__).resolve().parent.parent / "sample_docs"


def test_markdown_frontmatter_and_breadcrumbs() -> None:
    md = b"""---
title: Test Policy
department: HR
last_reviewed: 2026-01-01
---
# Ignored H1

Intro text.

## Leave
### Sick
Ten days.

```python
# not a heading
x = 1
```
## Other
Body.
"""
    doc = load_document("policy.md", md)
    assert doc.title == "Test Policy"
    assert doc.metadata == {"department": "HR", "last_reviewed": "2026-01-01"}
    paths = [s.heading_path for s in doc.sections]
    assert paths == [[], ["Leave", "Sick"], ["Other"]]
    assert "# not a heading" in doc.sections[1].text


def test_markdown_title_falls_back_to_h1_then_filename() -> None:
    assert load_document("a.md", b"# Real Title\n\nText").title == "Real Title"
    assert load_document("my-doc_name.md", b"Just text").title == "My Doc Name"


def test_html_structure_meta_and_tables() -> None:
    doc = load_document(
        "access.html", (SAMPLE / "security/access-management-and-offboarding.html").read_bytes()
    )
    assert doc.title == "Access Management & Offboarding Standard"
    assert doc.metadata["department"] == "Security"
    paths = {tuple(s.heading_path) for s in doc.sections}
    assert ("Offboarding", "Involuntary departures") in paths
    text = doc.full_text
    assert "Intranet home" not in text  # nav stripped
    assert "Credential rotation | Keys are rotated at least every 90 days" in text


def test_text_header_block_and_caps_headings() -> None:
    doc = load_document(
        "expense.txt", (SAMPLE / "finance/expense-and-travel-policy.txt").read_bytes()
    )
    assert doc.title == "Expense and Travel Policy"
    assert doc.metadata["department"] == "Finance"
    assert ["Approval Limits"] in [s.heading_path for s in doc.sections]


def test_pdf_pages_and_metadata() -> None:
    doc = load_document("byod.pdf", (SAMPLE / "it/laptop-and-byod-standard.pdf").read_bytes())
    assert doc.doc_type == "pdf"
    assert doc.title == "Laptop and BYOD Standard"
    assert [s.page for s in doc.sections] == [1, 2]
    assert "every 3 years" in doc.full_text


def test_extra_metadata_overrides() -> None:
    doc = load_document("x.md", b"# T\n\nbody", {"department": "Legal", "owner": None})
    assert doc.metadata == {"department": "Legal"}


def test_unsupported_and_empty() -> None:
    with pytest.raises(UnsupportedFormatError):
        load_document("x.docx", b"data")
    with pytest.raises(EmptyDocumentError):
        load_document("x.md", b"   \n\n")
    with pytest.raises(EmptyDocumentError):
        load_document("x.pdf", b"not a pdf")
