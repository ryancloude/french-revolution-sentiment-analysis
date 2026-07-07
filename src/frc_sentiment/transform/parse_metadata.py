"""Parse bronze metadata XML into the silver documents table."""

from __future__ import annotations

import argparse
import re
import unicodedata

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

VALID_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")

PARSED_METADATA_SCHEMA = T.StructType(
    [
        T.StructField("metadata_document_id", T.StringType(), True),
        T.StructField("title", T.StringType(), True),
        T.StructField("creator", T.StringType(), True),
        T.StructField("author", T.StringType(), True),
        T.StructField("creators", T.ArrayType(T.StringType()), True),
        T.StructField("metadata_authors", T.ArrayType(T.StringType()), True),
        T.StructField("subjects", T.ArrayType(T.StringType()), True),
        T.StructField("subjects_raw", T.StringType(), True),
        T.StructField("publication_date_raw", T.StringType(), True),
        T.StructField("language", T.StringType(), True),
        T.StructField("internet_archive_id", T.StringType(), True),
        T.StructField("source_url", T.StringType(), True),
        T.StructField("metadata_parse_status", T.StringType(), False),
        T.StructField("metadata_parse_error", T.StringType(), True),
    ]
)

DATE_PARSE_SCHEMA = T.StructType(
    [
        T.StructField("publication_year", T.IntegerType(), True),
        T.StructField("publication_month", T.IntegerType(), True),
        T.StructField("publication_day", T.IntegerType(), True),
        T.StructField("publication_date", T.StringType(), True),
        T.StructField("date_precision", T.StringType(), False),
        T.StructField("date_source", T.StringType(), False),
        T.StructField("date_calendar", T.StringType(), False),
        T.StructField("date_parse_confidence", T.StringType(), False),
        T.StructField("date_parse_notes", T.StringType(), True),
    ]
)


def quote_identifier(identifier: str) -> str:
    """Validate and quote a Unity Catalog identifier."""
    if not VALID_IDENTIFIER.fullmatch(identifier):
        raise ValueError(f"Invalid Unity Catalog identifier: {identifier!r}")

    return f"`{identifier}`"


def table_name(catalog: str, schema: str, table: str) -> str:
    """Build a fully qualified table name."""
    return ".".join(
        [
            quote_identifier(catalog),
            quote_identifier(schema),
            quote_identifier(table),
        ]
    )

FRENCH_MONTHS = {
    "janvier": 1,
    "janv": 1,
    "janv.": 1,
    "fevrier": 2,
    "fevr": 2,
    "fevr.": 2,
    "fev": 2,
    "fev.": 2,
    "mars": 3,
    "avril": 4,
    "avr": 4,
    "avr.": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7,
    "juil": 7,
    "juil.": 7,
    "aout": 8,
    "aoust": 8,
    "septembre": 9,
    "sept": 9,
    "sept.": 9,
    "7bre": 9,
    "octobre": 10,
    "oct": 10,
    "oct.": 10,
    "8bre": 10,
    "novembre": 11,
    "nov": 11,
    "nov.": 11,
    "9bre": 11,
    "decembre": 12,
    "dec": 12,
    "dec.": 12,
    "10bre": 12,
    "xbre": 12,
}

REVOLUTIONARY_MONTHS = {
    "vendemiaire": 1,
    "vend": 1,
    "vend.": 1,
    "brumaire": 2,
    "brum": 2,
    "brum.": 2,
    "frimaire": 3,
    "frim": 3,
    "frim.": 3,
    "nivose": 4,
    "niv": 4,
    "niv.": 4,
    "pluviose": 5,
    "pluv": 5,
    "pluv.": 5,
    "ventose": 6,
    "vent": 6,
    "vent.": 6,
    "germinal": 7,
    "germ": 7,
    "germ.": 7,
    "floreal": 8,
    "flor": 8,
    "flor.": 8,
    "prairial": 9,
    "prair": 9,
    "prair.": 9,
    "messidor": 10,
    "mess": 10,
    "mess.": 10,
    "thermidor": 11,
    "therm": 11,
    "therm.": 11,
    "fructidor": 12,
    "fruct": 12,
    "fruct.": 12,
}


def strip_accents(value: str) -> str:
    """Remove accents for simpler French date matching."""
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(character for character in normalized if not unicodedata.combining(character))


def normalize_date_text(value: str | None) -> str:
    """Normalize date text for regex-based date parsing."""
    if not value:
        return ""

    no_accents = strip_accents(value.lower())
    no_accents = no_accents.replace("ſ", "s")
    normalized = no_accents.replace(",", " ").replace(".", " ").replace("'", " ")
    return " ".join(normalized.split())


def roman_to_int(value: str) -> int | None:
    """Convert a Roman numeral to an integer."""
    roman_values = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total = 0
    previous = 0

    for character in reversed(value.lower()):
        current = roman_values.get(character)
        if current is None:
            return None

        if current < previous:
            total -= current
        else:
            total += current
            previous = current

    return total or None


def parse_revolutionary_year(value: str) -> int | None:
    """Parse a French Republican calendar year written as a number or Roman numeral."""
    if value.isdigit():
        return int(value)

    return roman_to_int(value)


def date_result(
    publication_year: int | None,
    publication_month: int | None,
    publication_day: int | None,
    publication_date: str | None,
    date_precision: str,
    date_source: str,
    date_calendar: str,
    date_parse_confidence: str,
    date_parse_notes: str | None = None,
) -> dict[str, str | int | None]:
    """Build a consistent date parse result."""
    return {
        "publication_year": publication_year,
        "publication_month": publication_month,
        "publication_day": publication_day,
        "publication_date": publication_date,
        "date_precision": date_precision,
        "date_source": date_source,
        "date_calendar": date_calendar,
        "date_parse_confidence": date_parse_confidence,
        "date_parse_notes": date_parse_notes,
    }


def convert_revolutionary_date(
    republican_year: int,
    republican_month: int,
    republican_day: int,
) -> tuple[int, int, int]:
    """Convert a French Republican date to Gregorian year, month, day."""
    from convertdate import french_republican

    return french_republican.to_gregorian(
        republican_year,
        republican_month,
        republican_day,
    )


def parse_date_from_text(
    value: str | None,
    source: str,
    confidence: str,
) -> dict[str, str | int | None]:
    """Parse Gregorian or French Republican date information from one text field."""
    normalized = normalize_date_text(value)

    if not normalized:
        return date_result(None, None, None, None, "unknown", "none", "unknown", "low")

    iso_match = re.search(r"\b(1[5-9]\d{2}|20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", normalized)
    if iso_match:
        year = int(iso_match.group(1))
        month = int(iso_match.group(2))
        day = int(iso_match.group(3))

        if 1 <= month <= 12 and 1 <= day <= 31:
            return date_result(
                year,
                month,
                day,
                f"{year:04d}-{month:02d}-{day:02d}",
                "day",
                source,
                "gregorian",
                confidence,
            )

    revolutionary_day_match = re.search(
        r"\b(\d{1,2})(?:er)?\s+("
        + "|".join(REVOLUTIONARY_MONTHS)
        + r")\s+an\s+([ivxlcdm]+|\d+)\b",
        normalized,
    )
    if revolutionary_day_match:
        republican_day = int(revolutionary_day_match.group(1))
        republican_month = REVOLUTIONARY_MONTHS[revolutionary_day_match.group(2)]
        republican_year = parse_revolutionary_year(revolutionary_day_match.group(3))

        if republican_year is not None and 1 <= republican_day <= 30:
            year, month, day = convert_revolutionary_date(
                republican_year,
                republican_month,
                republican_day,
            )
            return date_result(
                year,
                month,
                day,
                f"{year:04d}-{month:02d}-{day:02d}",
                "day",
                source,
                "french_republican",
                confidence,
                "Converted from French Republican calendar using convertdate.",
            )

    french_date_match = re.search(
        r"\b(\d{1,2})(?:er)?\s+("
        + "|".join(FRENCH_MONTHS)
        + r")\s+(1[5-9]\d{2}|20\d{2})\b",
        normalized,
    )
    if french_date_match:
        day = int(french_date_match.group(1))
        month = FRENCH_MONTHS[french_date_match.group(2)]
        year = int(french_date_match.group(3))

        if 1 <= day <= 31:
            return date_result(
                year,
                month,
                day,
                f"{year:04d}-{month:02d}-{day:02d}",
                "day",
                source,
                "gregorian",
                confidence,
            )

    french_month_year_match = re.search(
        r"\b("
        + "|".join(FRENCH_MONTHS)
        + r")\s+(1[5-9]\d{2}|20\d{2})\b",
        normalized,
    )
    if french_month_year_match:
        month = FRENCH_MONTHS[french_month_year_match.group(1)]
        year = int(french_month_year_match.group(2))
        return date_result(
            year,
            month,
            None,
            None,
            "month",
            source,
            "gregorian",
            confidence,
        )

    revolutionary_month_match = re.search(
        r"\b("
        + "|".join(REVOLUTIONARY_MONTHS)
        + r")\s+an\s+([ivxlcdm]+|\d+)\b",
        normalized,
    )
    if revolutionary_month_match:
        return date_result(
            None,
            REVOLUTIONARY_MONTHS[revolutionary_month_match.group(1)],
            None,
            None,
            "revolutionary_month",
            source,
            "french_republican",
            "low",
            "French Republican month detected, but no day was available for exact conversion.",
        )

    year_match = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", normalized)
    if year_match:
        year = int(year_match.group(1))
        return date_result(
            year,
            None,
            None,
            None,
            "year",
            source,
            "gregorian",
            confidence,
        )

    return date_result(None, None, None, None, "unknown", "none", "unknown", "low")


def date_precision_rank(parsed_date: dict[str, str | int | None]) -> int:
    """Rank parsed dates by usefulness for chronological analysis."""
    precision = parsed_date["date_precision"]

    if precision == "day":
        return 3
    if precision == "month":
        return 2
    if precision in {"year", "revolutionary_month"}:
        return 1

    return 0


def has_conflicting_year(
    candidate: dict[str, str | int | None],
    baseline_year: int | None,
) -> bool:
    """Return True when a candidate date conflicts with a known metadata year."""
    candidate_year = candidate["publication_year"]

    return (
        baseline_year is not None
        and candidate_year is not None
        and candidate_year != baseline_year
    )


def append_date_note(
    parsed_date: dict[str, str | int | None],
    note: str,
) -> dict[str, str | int | None]:
    """Append a note to a parsed date result."""
    updated = parsed_date.copy()
    existing_notes = updated.get("date_parse_notes")

    if existing_notes:
        updated["date_parse_notes"] = f"{existing_notes} {note}"
    else:
        updated["date_parse_notes"] = note

    return updated


def parse_publication_date(
    publication_date_raw: str | None,
    title: str | None,
    ocr_front_matter: str | None = None,
) -> dict[str, str | int | None]:
    """Parse the best available publication date from metadata, title, and OCR front matter."""
    metadata_result = parse_date_from_text(
        publication_date_raw,
        source="metadata_date",
        confidence="high",
    )
    title_result = parse_date_from_text(
        title,
        source="title",
        confidence="medium",
    )
    ocr_result = parse_date_from_text(
        ocr_front_matter,
        source="ocr_front_matter",
        confidence="low",
    )

    candidates = [metadata_result, title_result, ocr_result]

    metadata_year = metadata_result["publication_year"]

    non_conflicting_candidates = [
        candidate
        for candidate in candidates
        if candidate["date_precision"] != "unknown"
        and not has_conflicting_year(candidate, metadata_year)
    ]

    if not non_conflicting_candidates:
        return date_result(None, None, None, None, "unknown", "none", "unknown", "low")

    best_candidate = max(
        non_conflicting_candidates,
        key=lambda candidate: (
            date_precision_rank(candidate),
            # Tie-breaker: prefer more authoritative sources.
            {"metadata_date": 3, "title": 2, "ocr_front_matter": 1}.get(
                str(candidate["date_source"]),
                0,
            ),
        ),
    )

    if (
        best_candidate["date_source"] != "metadata_date"
        and metadata_year is not None
        and best_candidate["publication_year"] == metadata_year
    ):
        return append_date_note(
            best_candidate,
            "More precise date matched metadata year.",
        )

    return best_candidate

def parse_metadata_xml(raw_xml: str | None) -> tuple:
    """Parse one Internet Archive metadata XML document."""
    import xml.etree.ElementTree as ET

    def clean_text(value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = " ".join(value.split())
        return cleaned or None

    def empty_parse_result(
        status: str,
        error: str | None,
    ) -> tuple:
        return (
            None,  # metadata_document_id
            None,  # title
            None,  # creator
            None,  # author
            [],  # creators
            [],  # metadata_authors
            [],  # subjects
            None,  # subjects_raw
            None,  # publication_date_raw
            None,  # language
            None,  # internet_archive_id
            None,  # source_url
            status,
            error,
        )

    if not raw_xml:
        return empty_parse_result("parse_error", "Empty XML document")

    try:
        root = ET.fromstring(raw_xml)
    except Exception as exc:
        return empty_parse_result("parse_error", str(exc)[:500])

    def get_first(*tags: str) -> str | None:
        for tag in tags:
            element = root.find(tag)
            if element is not None:
                value = clean_text(element.text)
                if value:
                    return value

        return None

    def get_all(tag: str) -> list[str]:
        values = []

        for element in root.findall(tag):
            value = clean_text(element.text)
            if value and value not in values:
                values.append(value)

        return values

    creators = get_all("creator")
    metadata_authors = get_all("author")
    subjects = get_all("subject")

    creator = creators[0] if creators else None

    # Keep the existing broad "author" field for compatibility.
    # Prefer creator because Internet Archive metadata commonly uses creator
    # as the primary author/creator field.
    author = creator or (metadata_authors[0] if metadata_authors else None)

    metadata_document_id = get_first("identifier")
    title = get_first("title")
    publication_date_raw = get_first("date")
    language = get_first("language")
    internet_archive_id = get_first("identifier")
    source_url = get_first("identifier-access")
    subjects_raw = "; ".join(subjects) if subjects else None

    return (
        metadata_document_id,
        title,
        creator,
        author,
        creators,
        metadata_authors,
        subjects,
        subjects_raw,
        publication_date_raw,
        language,
        internet_archive_id,
        source_url,
        "success",
        None,
    )


def build_silver_documents(
    spark: SparkSession,
    catalog: str,
    schema: str,
) -> DataFrame:
    """Build the silver documents DataFrame from bronze metadata and OCR tables."""
    bronze_metadata_table = table_name(catalog, schema, "bronze_metadata")
    bronze_ocr_text_table = table_name(catalog, schema, "bronze_ocr_text")

    parse_metadata_udf = F.udf(parse_metadata_xml, PARSED_METADATA_SCHEMA)

    parsed_metadata = (
        spark.table(bronze_metadata_table)
        .withColumn("parsed", parse_metadata_udf(F.col("raw_metadata_xml")))
        .select(
            F.col("document_id"),
            F.col("file_path").alias("metadata_file_path"),
            F.col("parsed.metadata_document_id"),
            F.col("parsed.title"),
            F.col("parsed.creator"),
            F.col("parsed.author"),
            F.col("parsed.creators"),
            F.col("parsed.metadata_authors"),
            F.col("parsed.subjects"),
            F.col("parsed.subjects_raw"),
            F.col("parsed.publication_date_raw"),
            F.col("parsed.language"),
            F.col("parsed.internet_archive_id"),
            F.col("parsed.source_url"),
            F.col("parsed.metadata_parse_status"),
            F.col("parsed.metadata_parse_error"),
        )
    )

    ocr_documents = (
        spark.table(bronze_ocr_text_table)
        .select(
            "document_id",
            F.substring(F.col("raw_text"), 1, 2000).alias("ocr_front_matter"),
        )
        .distinct()
        .withColumn("has_ocr_text", F.lit(True))
    )

    parsed_metadata = parsed_metadata.join(
        ocr_documents,
        on="document_id",
        how="left",
    )

    parse_publication_date_udf = F.udf(parse_publication_date, DATE_PARSE_SCHEMA)

    parsed_metadata = (
        parsed_metadata.withColumn(
            "parsed_date",
            parse_publication_date_udf(
                F.col("publication_date_raw"),
                F.col("title"),
                F.col("ocr_front_matter"),
            ),
        )
        .withColumn("publication_year", F.col("parsed_date.publication_year"))
        .withColumn("publication_month", F.col("parsed_date.publication_month"))
        .withColumn("publication_day", F.col("parsed_date.publication_day"))
        .withColumn("publication_date", F.to_date(F.col("parsed_date.publication_date")))
        .withColumn("date_precision", F.col("parsed_date.date_precision"))
        .withColumn("date_source", F.col("parsed_date.date_source"))
        .withColumn("date_calendar", F.col("parsed_date.date_calendar"))
        .withColumn("date_parse_confidence", F.col("parsed_date.date_parse_confidence"))
        .withColumn("date_parse_notes", F.col("parsed_date.date_parse_notes"))
        .drop("parsed_date")
    )

    return (
        parsed_metadata.withColumn("has_ocr_text", F.coalesce(F.col("has_ocr_text"), F.lit(False)))
        .withColumn(
            "document_id_matches_metadata",
            F.when(F.col("metadata_document_id").isNull(), F.lit(None).cast("boolean"))
            .otherwise(F.col("document_id") == F.col("metadata_document_id")),
        )
        .withColumn("parsed_at", F.current_timestamp())
        .select(
            "document_id",
            "metadata_document_id",
            "document_id_matches_metadata",
            "title",
            "creator",
            "author",
            "creators",
            "metadata_authors",
            "subjects",
            "subjects_raw",
            "publication_year",
            "publication_month",
            "publication_day",
            "publication_date",
            "publication_date_raw",
            "date_precision",
            "date_source",
            "date_calendar",
            "date_parse_confidence",
            "date_parse_notes",
            "language",
            "internet_archive_id",
            "source_url",
            "has_ocr_text",
            "ocr_front_matter",
            "metadata_parse_status",
            "metadata_parse_error",
            "metadata_file_path",
            "parsed_at",
        )
    )


def write_silver_documents(df: DataFrame, full_table_name: str) -> None:
    """Write the silver documents Delta table."""
    row_count = df.count()

    if row_count == 0:
        raise RuntimeError(f"No rows found for {full_table_name}")

    (
        df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(full_table_name)
    )

    print(f"Wrote {row_count} rows to {full_table_name}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    return parser.parse_args()


def main() -> None:
    """Parse bronze metadata into silver documents."""
    args = parse_args()
    spark = SparkSession.builder.getOrCreate()

    silver_documents = build_silver_documents(
        spark=spark,
        catalog=args.catalog,
        schema=args.schema,
    )

    write_silver_documents(
        silver_documents,
        table_name(args.catalog, args.schema, "silver_documents"),
    )


if __name__ == "__main__":
    main()