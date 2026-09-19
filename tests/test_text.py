from hybrid_rag.text import (
    SimpleTokenCounter,
    analyze,
    numbers,
    split_sentences,
    stem,
    strip_markdown,
    unwrap_lines,
)


def test_analyze_removes_stopwords_and_stems() -> None:
    assert analyze("The employees are accruing days") == ["employee", "accru", "day"]


def test_stem_is_conservative() -> None:
    assert stem("policies") == "policy"
    assert stem("process") == "process"
    assert stem("bus") == "bus"
    assert stem("2026") == "2026"


def test_split_sentences_handles_abbreviations_and_newlines() -> None:
    text = "Use a laptop, e.g. a MacBook. It must be encrypted.\n- Lock the screen"
    assert split_sentences(text) == [
        "Use a laptop, e.g. a MacBook.",
        "It must be encrypted.",
        "- Lock the screen",
    ]


def test_numbers_normalises_thousands_separators() -> None:
    assert numbers("A $1,000 stipend and 20 weeks, 99.9% uptime") == {"1000", "20", "99.9%"}


def test_unwrap_lines_joins_soft_wrapped_paragraphs_but_keeps_lists() -> None:
    text = "A rollback normally completes in\nunder 3 minutes.\n\n- item one\n- item two\n| a | b |"
    assert unwrap_lines(text) == [
        "A rollback normally completes in under 3 minutes.",
        "- item one",
        "- item two",
        "| a | b |",
    ]


def test_strip_markdown() -> None:
    assert strip_markdown("- **Public**: approved") == "Public: approved"
    assert strip_markdown("## Heading") == "Heading"


def test_simple_token_counter_is_monotonic() -> None:
    c = SimpleTokenCounter()
    assert c.count("") == 0
    assert c.count("hello world") < c.count("hello world, this is a longer sentence")
