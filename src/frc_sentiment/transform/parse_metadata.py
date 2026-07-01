"""Parse bronze metadata XML into the silver documents table."""

from __future__ import annotations

import argparse
import re

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

VALID_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")

PARSED_METADATA_SCHEMA = T.StructType(
    [
        T.StructField("metadata_document_id", T.StringType(), True),
        T.StructField("title", T.StringType(), True),
        T.StructField("author", T.StringType(), True),
        T.StructField("publication_date_raw", T.StringType(), True),
        T.StructField("language", T.StringType(), True),
        T.StructField("internet_archive_id", T.StringType(), True),
        T.StructField("source_url", T.StringType(), True),
        T.StructField("metadata_parse_status", T.StringType(), False),
        T.StructField("metadata_parse_error", T.StringType(), True),
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


def parse_metadata_xml(raw_xml: str | None) -> tuple[str | None, ...]:
    """Parse one Internet Archive metadata XML document."""
    import xml.etree.ElementTree as ET

    def clean_text(value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = " ".join(value.split())
        return cleaned or None

    if not raw_xml:
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "parse_error",
            "Empty XML document",
        )

    try:
        root = ET.fromstring(raw_xml)
    except Exception as exc:
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "parse_error",
            str(exc)[:500],
        )

    def get_first(*tags: str) -> str | None:
        for tag in tags:
            element = root.find(tag)
            if element is not None:
                value = clean_text(element.text)
                if value:
                    return value

        return None

    metadata_document_id = get_first("identifier")
    title = get_first("title")
    author = get_first("creator", "author")
    publication_date_raw = get_first("date")
    language = get_first("language")
    internet_archive_id = get_first("identifier")
    source_url = get_first("identifier-access")

    return (
        metadata_document_id,
        title,
        author,
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
            F.col("parsed.author"),
            F.col("parsed.publication_date_raw"),
            F.col("parsed.language"),
            F.col("parsed.internet_archive_id"),
            F.col("parsed.source_url"),
            F.col("parsed.metadata_parse_status"),
            F.col("parsed.metadata_parse_error"),
        )
    )

    year_candidate = F.regexp_extract(
        F.coalesce(F.col("publication_date_raw"), F.lit("")),
        r"(1[5-9]\d{2}|20\d{2})",
        1,
    )

    parsed_metadata = parsed_metadata.withColumn(
        "publication_year",
        F.when(year_candidate != "", year_candidate.cast("int")).otherwise(
            F.lit(None).cast("int")
        ),
    )

    ocr_documents = (
        spark.table(bronze_ocr_text_table)
        .select("document_id")
        .distinct()
        .withColumn("has_ocr_text", F.lit(True))
    )

    return (
        parsed_metadata.join(ocr_documents, on="document_id", how="left")
        .withColumn("has_ocr_text", F.coalesce(F.col("has_ocr_text"), F.lit(False)))
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
            "author",
            "publication_year",
            "publication_date_raw",
            "language",
            "internet_archive_id",
            "source_url",
            "has_ocr_text",
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