from itertools import pairwise

from hybrid_rag.ingestion.chunker import StructureAwareChunker
from hybrid_rag.schemas import LoadedDocument, Section
from hybrid_rag.text import SimpleTokenCounter, split_sentences

COUNTER = SimpleTokenCounter()


def _doc(*sections: Section) -> LoadedDocument:
    return LoadedDocument(source="t.md", title="T", doc_type="markdown", sections=list(sections))


def _long_text(n: int) -> str:
    return " ".join(f"Sentence number {i} talks about policy item {i}." for i in range(n))


def test_chunks_respect_token_budget_and_carry_breadcrumbs() -> None:
    chunker = StructureAwareChunker(COUNTER, chunk_size=60, overlap=15, min_tokens=5)
    chunks = chunker.chunk(_doc(Section(heading_path=["A", "B"], text=_long_text(40))), "doc1")
    assert len(chunks) > 3
    # Small slack for the tiny-remainder merge at the end of a section.
    assert all(c.token_count <= 60 + 5 for c in chunks)
    assert all(c.heading_path == ["A", "B"] for c in chunks)
    assert chunks[0].metadata["heading_path"] == "A > B"
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].chunk_id == "doc1-0000"


def test_overlap_is_sentence_aligned() -> None:
    chunker = StructureAwareChunker(COUNTER, chunk_size=60, overlap=20, min_tokens=5)
    chunks = chunker.chunk(_doc(Section(text=_long_text(30))), "d")
    for prev, nxt in pairwise(chunks):
        # The next chunk opens with whole sentences copied from the end of the previous one.
        assert split_sentences(prev.text)[-1] in nxt.text
        assert split_sentences(nxt.text)[0] in prev.text


def test_no_overlap_when_disabled() -> None:
    chunker = StructureAwareChunker(COUNTER, chunk_size=60, overlap=0, min_tokens=5)
    chunks = chunker.chunk(_doc(Section(text=_long_text(30))), "d")
    joined = " ".join(c.text for c in chunks)
    assert joined.count("Sentence number 5 ") == 1


def test_chunks_never_span_sections() -> None:
    chunker = StructureAwareChunker(COUNTER, chunk_size=200, overlap=20, min_tokens=3)
    chunks = chunker.chunk(
        _doc(
            Section(heading_path=["One"], text="Alpha paragraph with enough words here."),
            Section(heading_path=["Two"], text="Beta paragraph with enough words here."),
        ),
        "d",
    )
    assert [c.heading_path for c in chunks] == [["One"], ["Two"]]


def test_code_fence_kept_intact() -> None:
    code = "```bash\necho one\n\necho two\n```"
    chunker = StructureAwareChunker(COUNTER, chunk_size=200, overlap=10, min_tokens=3)
    chunks = chunker.chunk(_doc(Section(text=f"Run this:\n\n{code}\n\nDone.")), "d")
    assert len(chunks) == 1
    assert code in chunks[0].text


def test_tiny_parent_section_folds_into_child() -> None:
    chunker = StructureAwareChunker(COUNTER, chunk_size=200, overlap=10, min_tokens=10)
    chunks = chunker.chunk(
        _doc(
            Section(heading_path=["Leave"], text="Overview."),
            Section(heading_path=["Leave", "Sick"], text=_long_text(3)),
        ),
        "d",
    )
    assert len(chunks) == 1
    assert chunks[0].heading_path == ["Leave", "Sick"]
    assert chunks[0].text.startswith("Overview.")


def test_giant_word_falls_back_to_word_windows() -> None:
    chunker = StructureAwareChunker(COUNTER, chunk_size=30, overlap=5, min_tokens=2)
    text = " ".join(["word"] * 200)  # no sentence or line boundaries at all
    chunks = chunker.chunk(_doc(Section(text=text)), "d")
    assert len(chunks) > 3
    assert all(c.token_count <= 32 for c in chunks)


def test_filterable_metadata_is_copied() -> None:
    doc = _doc(Section(text="Some text about things and more."))
    doc.metadata = {"department": "HR", "tags": ["a"], "owner": "X"}
    chunk = StructureAwareChunker(COUNTER, 100, 10, 2).chunk(doc, "d")[0]
    assert chunk.metadata["department"] == "HR"
    assert chunk.metadata["owner"] == "X"
    assert "tags" not in chunk.metadata
    assert chunk.contextual_text.startswith("T\n\n")
