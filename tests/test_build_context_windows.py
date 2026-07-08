"""Tests for context window extraction."""

from frc_sentiment.transform.build_mention_contexts import (
    build_context_for_mention,
    word_spans,
)


def test_word_spans_returns_offsets_for_tokens() -> None:
    """Word spans return start and end offsets for each non-whitespace token."""
    assert word_spans("un  deux trois") == [(0, 2), (4, 8), (9, 14)]


def test_build_context_for_mention_middle_of_text() -> None:
    """A context window includes requested words before and after a mention."""
    text = "un deux trois robespierre quatre cinq six"
    start = text.index("robespierre")
    end = start + len("robespierre")

    context = build_context_for_mention(
        clean_text=text,
        match_start_char=start,
        match_end_char=end,
        words_before=2,
        words_after=2,
    )

    assert context["context_window"] == "deux trois robespierre quatre cinq"
    assert context["words_before_actual"] == 2
    assert context["words_after_actual"] == 2
    assert context["context_word_count"] == 5


def test_build_context_for_mention_near_start() -> None:
    """A mention near the beginning uses fewer before-words than requested."""
    text = "robespierre quatre cinq six"
    start = text.index("robespierre")
    end = start + len("robespierre")

    context = build_context_for_mention(
        clean_text=text,
        match_start_char=start,
        match_end_char=end,
        words_before=3,
        words_after=2,
    )

    assert context["context_window"] == "robespierre quatre cinq"
    assert context["words_before_actual"] == 0
    assert context["words_after_actual"] == 2
    assert context["context_word_count"] == 3


def test_build_context_for_mention_near_end() -> None:
    """A mention near the end uses fewer after-words than requested."""
    text = "un deux trois danton"
    start = text.index("danton")
    end = start + len("danton")

    context = build_context_for_mention(
        clean_text=text,
        match_start_char=start,
        match_end_char=end,
        words_before=2,
        words_after=3,
    )

    assert context["context_window"] == "deux trois danton"
    assert context["words_before_actual"] == 2
    assert context["words_after_actual"] == 0
    assert context["context_word_count"] == 3


def test_build_context_for_mention_multiword_variant() -> None:
    """A multiword matched variant counts the full mention span."""
    text = "avant louis capet après"
    start = text.index("louis")
    end = start + len("louis capet")

    context = build_context_for_mention(
        clean_text=text,
        match_start_char=start,
        match_end_char=end,
        words_before=1,
        words_after=1,
    )

    assert context["context_window"] == "avant louis capet après"
    assert context["words_before_actual"] == 1
    assert context["words_after_actual"] == 1
    assert context["context_word_count"] == 4


def test_build_context_for_mention_invalid_position_returns_empty_context() -> None:
    """Invalid mention offsets return an empty context."""
    context = build_context_for_mention(
        clean_text="un deux trois",
        match_start_char=100,
        match_end_char=110,
        words_before=2,
        words_after=2,
    )

    assert context == {
        "context_window": "",
        "context_start_char": 0,
        "context_end_char": 0,
        "words_before_actual": 0,
        "words_after_actual": 0,
        "context_word_count": 0,
    }


def test_build_context_for_mention_empty_text_returns_empty_context() -> None:
    """Empty or null text returns an empty context."""
    context = build_context_for_mention(
        clean_text=None,
        match_start_char=0,
        match_end_char=1,
        words_before=2,
        words_after=2,
    )

    assert context["context_window"] == ""
    assert context["context_word_count"] == 0