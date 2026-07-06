"""Tests for rule-based date candidate generation."""

from frc_sentiment.transform.build_date_candidates import (
    build_rule_candidates_for_document,
    source_excerpt,
)


def test_source_excerpt_collapses_whitespace() -> None:
    """Source excerpts should be compact enough for review tables."""
    assert source_excerpt("  un\n\n deux   trois  ") == "un deux trois"


def test_source_excerpt_truncates_long_text() -> None:
    """Long source text should be truncated."""
    excerpt = source_excerpt("a" * 600, max_length=10)

    assert excerpt == "aaaaaaaaaa..."


def test_build_rule_candidates_includes_metadata_title_and_ocr_candidates() -> None:
    """Rule candidates should preserve all non-unknown source candidates."""
    candidates = build_rule_candidates_for_document(
        publication_date_raw="1792",
        title="Adresse du 10 août 1792",
        ocr_front_matter="Imprimé le 9 thermidor an II.",
    )

    assert len(candidates) == 3

    assert [candidate["source_field"] for candidate in candidates] == [
        "metadata_date",
        "title",
        "ocr_front_matter",
    ]

    assert candidates[0]["date_precision"] == "year"
    assert candidates[1]["publication_date"] == "1792-08-10"
    assert candidates[2]["publication_date"] == "1794-07-27"


def test_build_rule_candidates_skips_unknown_sources() -> None:
    """Unknown source fields should not create candidate rows."""
    candidates = build_rule_candidates_for_document(
        publication_date_raw=None,
        title="Sans date claire",
        ocr_front_matter=None,
    )

    assert candidates == []


def test_build_rule_candidates_marks_confidence_by_source() -> None:
    """Candidate confidence should reflect source type."""
    candidates = build_rule_candidates_for_document(
        publication_date_raw="1792",
        title="Adresse du 10 août 1792",
        ocr_front_matter="Imprimé le 10 août 1792.",
    )

    confidence_by_source = {
        candidate["source_field"]: candidate["confidence"] for candidate in candidates
    }

    assert confidence_by_source == {
        "metadata_date": "high",
        "title": "medium",
        "ocr_front_matter": "low",
    }