"""Tests for metadata XML parsing."""

from frc_sentiment.transform.build_document_model import parse_metadata_xml


def test_parse_metadata_xml_success() -> None:
    """Metadata XML fields are parsed into the expected tuple positions."""
    raw_xml = """
    <metadata>
      <language>fre</language>
      <title>Discours sur la Révolution</title>
      <creator>Jean Test</creator>
      <subject>France -- History -- Revolution, 1789-1799</subject>
      <subject>Political pamphlets</subject>
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
        "Jean Test",
        ["Jean Test"],
        [],
        ["France -- History -- Revolution, 1789-1799", "Political pamphlets"],
        "France -- History -- Revolution, 1789-1799; Political pamphlets",
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

    assert parsed[2] is None
    assert parsed[3] == "Anonymous Author"
    assert parsed[4] == []
    assert parsed[5] == ["Anonymous Author"]


def test_parse_metadata_xml_preserves_multiple_creators_and_subjects() -> None:
    """Repeated creator and subject tags are preserved as arrays."""
    raw_xml = """
    <metadata>
      <title>Test title</title>
      <creator>First Creator</creator>
      <creator>Second Creator</creator>
      <subject>Subject A</subject>
      <subject>Subject B</subject>
      <identifier>test-multiple</identifier>
    </metadata>
    """

    parsed = parse_metadata_xml(raw_xml)

    assert parsed[2] == "First Creator"
    assert parsed[3] == "First Creator"
    assert parsed[4] == ["First Creator", "Second Creator"]
    assert parsed[6] == ["Subject A", "Subject B"]
    assert parsed[7] == "Subject A; Subject B"


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
    assert parsed[8] == "1793"


def test_parse_metadata_xml_empty_document_returns_parse_error() -> None:
    """Empty XML input returns a parse_error status instead of raising."""
    parsed = parse_metadata_xml("")

    assert parsed[12] == "parse_error"
    assert parsed[13] == "Empty XML document"


def test_parse_metadata_xml_malformed_document_returns_parse_error() -> None:
    """Malformed XML returns a parse_error status instead of raising."""
    parsed = parse_metadata_xml("<metadata><title>Broken</metadata>")

    assert parsed[12] == "parse_error"
    assert parsed[13] is not None