"""Load raw OCR text and metadata files from a Databricks volume into bronze Delta tables."""

from __future__ import annotations

import argparse
import re

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

VALID_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
VALID_SOURCE_SUBDIR = re.compile(r"^[A-Za-z0-9_/-]+$")


def quote_identifier(identifier: str) -> str:
    """Validate and quote a Unity Catalog identifier."""
    if not VALID_IDENTIFIER.fullmatch(identifier):
        raise ValueError(f"Invalid Unity Catalog identifier: {identifier!r}")

    return f"`{identifier}`"


def validate_source_subdir(source_subdir: str) -> str:
    """Validate a simple volume-relative source directory."""
    if not VALID_SOURCE_SUBDIR.fullmatch(source_subdir):
        raise ValueError(f"Invalid source subdirectory: {source_subdir!r}")

    return source_subdir.strip("/")


def table_name(catalog: str, schema: str, table: str) -> str:
    """Build a fully qualified table name."""
    return ".".join(
        [
            quote_identifier(catalog),
            quote_identifier(schema),
            quote_identifier(table),
        ]
    )


def volume_path(catalog: str, schema: str, volume: str, source_subdir: str) -> str:
    """Build the Databricks volume path for the raw sample files."""
    source_subdir = validate_source_subdir(source_subdir)
    return f"/Volumes/{catalog}/{schema}/{volume}/{source_subdir}"


def read_metadata_files(spark: SparkSession, base_path: str) -> DataFrame:
    """Read raw metadata XML files as one row per file."""
    path = f"{base_path}/metadata/*.xml"

    return (
        spark.read.format("binaryFile")
        .load(path)
        .select(
            F.regexp_extract(F.col("path"), r"([^/\\]+)_meta\.xml$", 1).alias("document_id"),
            F.col("path").alias("file_path"),
            F.decode(F.col("content"), "UTF-8").alias("raw_metadata_xml"),
            F.current_timestamp().alias("ingested_at"),
        )
        .where(F.col("document_id") != "")
    )


def read_ocr_text_files(spark: SparkSession, base_path: str) -> DataFrame:
    """Read raw OCR text files as one row per file."""
    path = f"{base_path}/ocr_text/*.txt"

    return (
        spark.read.format("binaryFile")
        .load(path)
        .select(
            F.regexp_extract(F.col("path"), r"([^/\\]+)_djvu\.txt$", 1).alias("document_id"),
            F.col("path").alias("file_path"),
            F.decode(F.col("content"), "UTF-8").alias("raw_text"),
            F.lit("ocr_text").alias("source_file_type"),
            F.current_timestamp().alias("ingested_at"),
        )
        .where(F.col("document_id") != "")
    )


def write_delta_table(df: DataFrame, full_table_name: str) -> None:
    """Overwrite a bronze Delta table."""
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
    parser.add_argument("--volume", required=True)
    parser.add_argument("--source-subdir", default="sample")
    return parser.parse_args()


def main() -> None:
    """Load bronze metadata and OCR text tables."""
    args = parse_args()
    spark = SparkSession.builder.getOrCreate()

    base_path = volume_path(
        catalog=args.catalog,
        schema=args.schema,
        volume=args.volume,
        source_subdir=args.source_subdir,
    )

    metadata = read_metadata_files(spark, base_path)
    ocr_text = read_ocr_text_files(spark, base_path)

    write_delta_table(
        metadata,
        table_name(args.catalog, args.schema, "bronze_metadata"),
    )

    write_delta_table(
        ocr_text,
        table_name(args.catalog, args.schema, "bronze_ocr_text"),
    )


if __name__ == "__main__":
    main()