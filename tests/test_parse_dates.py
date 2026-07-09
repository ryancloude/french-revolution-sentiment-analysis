"""Tests for publication date extraction."""

from frc_sentiment.transform.build_document_model import parse_publication_date


def test_parse_year_only_from_metadata_date() -> None:
    """A year-only metadata date should not become a fake exact date."""
    parsed = parse_publication_date(publication_date_raw="1789", title=None)

    assert parsed["publication_year"] == 1789
    assert parsed["publication_month"] is None
    assert parsed["publication_day"] is None
    assert parsed["publication_date"] is None
    assert parsed["date_precision"] == "year"
    assert parsed["date_calendar"] == "gregorian"
    assert parsed["date_source"] == "metadata_date"


def test_parse_month_year_from_metadata_date() -> None:
    """A month-year date should preserve month without faking a day."""
    parsed = parse_publication_date(publication_date_raw="août 1792", title=None)

    assert parsed["publication_year"] == 1792
    assert parsed["publication_month"] == 8
    assert parsed["publication_day"] is None
    assert parsed["publication_date"] is None
    assert parsed["date_precision"] == "month"


def test_parse_iso_date_from_metadata_date() -> None:
    """An ISO-style date should parse to an exact date."""
    parsed = parse_publication_date(publication_date_raw="1792-08-10", title=None)

    assert parsed["publication_year"] == 1792
    assert parsed["publication_month"] == 8
    assert parsed["publication_day"] == 10
    assert parsed["publication_date"] == "1792-08-10"
    assert parsed["date_precision"] == "day"


def test_parse_french_day_month_year_from_title() -> None:
    """The parser should fall back to the title if metadata date is missing."""
    parsed = parse_publication_date(
        publication_date_raw=None,
        title="Discours prononcé le 14 juillet 1789",
    )

    assert parsed["publication_year"] == 1789
    assert parsed["publication_month"] == 7
    assert parsed["publication_day"] == 14
    assert parsed["publication_date"] == "1789-07-14"
    assert parsed["date_precision"] == "day"
    assert parsed["date_source"] == "title"


def test_convert_revolutionary_calendar_date() -> None:
    """A full French Republican date should convert to Gregorian date."""
    parsed = parse_publication_date(
        publication_date_raw=None,
        title="Rapport du 9 thermidor an II",
    )

    assert parsed["publication_date"] == "1794-07-27"
    assert parsed["publication_year"] == 1794
    assert parsed["publication_month"] == 7
    assert parsed["publication_day"] == 27
    assert parsed["date_precision"] == "day"
    assert parsed["date_calendar"] == "french_republican"


def test_detect_revolutionary_month_without_converting() -> None:
    """A Republican month without a day should be detected but not converted."""
    parsed = parse_publication_date(
        publication_date_raw=None,
        title="Rapport de thermidor an II",
    )

    assert parsed["publication_year"] is None
    assert parsed["publication_month"] == 11
    assert parsed["publication_day"] is None
    assert parsed["publication_date"] is None
    assert parsed["date_precision"] == "revolutionary_month"
    assert parsed["date_calendar"] == "french_republican"


def test_unparseable_date_returns_unknown() -> None:
    """Unparseable dates should return explicit unknown fields."""
    parsed = parse_publication_date(publication_date_raw=None, title="Sans date claire")

    assert parsed["publication_year"] is None
    assert parsed["publication_month"] is None
    assert parsed["publication_day"] is None
    assert parsed["publication_date"] is None
    assert parsed["date_precision"] == "unknown"

def test_parse_date_from_ocr_front_matter_when_metadata_and_title_missing() -> None:
    """The parser should use OCR front matter as a fallback source."""
    parsed = parse_publication_date(
        publication_date_raw=None,
        title="Sans date claire",
        ocr_front_matter="Imprimé à Paris le 10 août 1792 avec permission.",
    )

    assert parsed["publication_year"] == 1792
    assert parsed["publication_month"] == 8
    assert parsed["publication_day"] == 10
    assert parsed["publication_date"] == "1792-08-10"
    assert parsed["date_precision"] == "day"
    assert parsed["date_source"] == "ocr_front_matter"
    assert parsed["date_parse_confidence"] == "low"


def test_metadata_date_takes_priority_over_ocr_front_matter() -> None:
    """Metadata date should win over OCR front matter when both are present."""
    parsed = parse_publication_date(
        publication_date_raw="1789",
        title=None,
        ocr_front_matter="Fait à Paris le 9 thermidor an II.",
    )

    assert parsed["publication_year"] == 1789
    assert parsed["publication_month"] is None
    assert parsed["publication_day"] is None
    assert parsed["publication_date"] is None
    assert parsed["date_precision"] == "year"
    assert parsed["date_source"] == "metadata_date"


def test_parse_revolutionary_date_from_ocr_front_matter() -> None:
    """The parser should convert full Revolutionary calendar dates found in OCR front matter."""
    parsed = parse_publication_date(
        publication_date_raw=None,
        title=None,
        ocr_front_matter="Fait à Paris le 9 thermidor an II.",
    )

    assert parsed["publication_date"] == "1794-07-27"
    assert parsed["publication_year"] == 1794
    assert parsed["publication_month"] == 7
    assert parsed["publication_day"] == 27
    assert parsed["date_precision"] == "day"
    assert parsed["date_source"] == "ocr_front_matter"
    assert parsed["date_calendar"] == "french_republican"


def test_ocr_front_matter_can_improve_metadata_year_when_year_matches() -> None:
    """OCR front matter can improve a year-only metadata date when the year agrees."""
    parsed = parse_publication_date(
        publication_date_raw="1792",
        title=None,
        ocr_front_matter="Imprimé à Paris le 10 août 1792.",
    )

    assert parsed["publication_year"] == 1792
    assert parsed["publication_month"] == 8
    assert parsed["publication_day"] == 10
    assert parsed["publication_date"] == "1792-08-10"
    assert parsed["date_precision"] == "day"
    assert parsed["date_source"] == "ocr_front_matter"


def test_ocr_front_matter_does_not_override_conflicting_metadata_year() -> None:
    """OCR front matter should not override a conflicting metadata year."""
    parsed = parse_publication_date(
        publication_date_raw="1789",
        title=None,
        ocr_front_matter="Imprimé à Paris le 10 août 1792.",
    )

    assert parsed["publication_year"] == 1789
    assert parsed["publication_month"] is None
    assert parsed["publication_day"] is None
    assert parsed["publication_date"] is None
    assert parsed["date_precision"] == "year"
    assert parsed["date_source"] == "metadata_date"

def test_parse_historical_aoust_spelling() -> None:
    """Historical spelling aoust should parse as August."""
    parsed = parse_publication_date(
        publication_date_raw=None,
        title="Lettre du 10 aoust 1792",
    )

    assert parsed["publication_date"] == "1792-08-10"
    assert parsed["publication_month"] == 8
    assert parsed["date_precision"] == "day"


def test_parse_numeric_old_french_month_7bre() -> None:
    """Old French 7bre notation should parse as September."""
    parsed = parse_publication_date(
        publication_date_raw=None,
        title="Adresse du 5 7bre 1791",
    )

    assert parsed["publication_date"] == "1791-09-05"
    assert parsed["publication_month"] == 9
    assert parsed["date_precision"] == "day"


def test_parse_abbreviated_revolutionary_month() -> None:
    """Abbreviated Revolutionary calendar month should convert when day and year exist."""
    parsed = parse_publication_date(
        publication_date_raw=None,
        title="Rapport du 9 therm an II",
    )

    assert parsed["publication_date"] == "1794-07-27"
    assert parsed["date_calendar"] == "french_republican"
    assert parsed["date_precision"] == "day"

def test_ocr_mangled_month_does_not_crash_parser() -> None:
    """OCR-mangled month-like text should not crash date parsing."""
    parsed = parse_publication_date(
        publication_date_raw=None,
        title="Adresse du 10 avru 1792",
    )

    assert parsed["publication_year"] == 1792
    assert parsed["publication_month"] is None
    assert parsed["publication_day"] is None
    assert parsed["date_precision"] == "year"

def test_invalid_gregorian_day_downgrades_to_month_precision() -> None:
    """Invalid dates like 31 November should preserve month precision but not crash."""
    parsed = parse_publication_date(
        publication_date_raw=None,
        title="Adresse du 31 novembre 1788",
    )

    assert parsed["publication_year"] == 1788
    assert parsed["publication_month"] == 11
    assert parsed["publication_day"] is None
    assert parsed["publication_date"] is None
    assert parsed["date_precision"] == "month"
    assert parsed["date_calendar"] == "gregorian"