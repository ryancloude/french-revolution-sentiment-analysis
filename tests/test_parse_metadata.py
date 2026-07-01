"""Tests for metadata XML parsing."""

from frc_sentiment.transform.parse_metadata import parse_metadata_xml


def test_parse_metadata_xml_success() -> None:
    """Metadata XML fields are parsed into the expected tuple positions."""
    raw_xml = """
    <metadata>
      <language>fre</language>
      <title>Discours sur la Révolution</title>
      <creator>Jean Test</creator>
      <date>1789</date>
      <identifier>discours1789test</identifier>
      <identifier-access>http://archive.org/details/discours1789test</identifier-access>
    </metadata>
    """

    parsed = parse_metadata_xml(raw_xml)

    assert parsed == (
        "discours1789test",
        "Discours sur la Révolution",
        "Jean Test",
        "1789",
        "fre",
        "discours1789test",
        "http://archive.org/details/discours1789test",
        "success",
        None,
    )


def test_parse_metadata_xml_uses_author_when_creator_missing() -> None:
    """The parser falls back to the author tag if creator is unavailable."""
    raw_xml = """
    <metadata>
      <title>Test title</title>
      <author>Anonymous Author</author>
      <date>1792</date>
      <identifier>test1792</identifier>
    </metadata>
    """

    parsed = parse_metadata_xml(raw_xml)

    assert parsed[2] == "Anonymous Author"


def test_parse_metadata_xml_cleans_extra_whitespace() -> None:
    """Text values are trimmed and internal whitespace is normalized."""
    raw_xml = """
    <metadata>
      <title>
        Discours

        avec     espaces
      </title>
      <date>  1793  </date>
      <identifier>  test1793  </identifier>
    </metadata>
    """

    parsed = parse_metadata_xml(raw_xml)

    assert parsed[0] == "test1793"
    assert parsed[1] == "Discours avec espaces"
    assert parsed[3] == "1793"


def test_parse_metadata_xml_empty_document_returns_parse_error() -> None:
    """Empty XML input returns a parse_error status instead of raising."""
    parsed = parse_metadata_xml("")

    assert parsed[7] == "parse_error"
    assert parsed[8] == "Empty XML document"


def test_parse_metadata_xml_malformed_document_returns_parse_error() -> None:
    """Malformed XML returns a parse_error status instead of raising."""
    parsed = parse_metadata_xml("<metadata><title>Broken</metadata>")

    assert parsed[7] == "parse_error"
    assert parsed[8] is not None