"""Clean OCR text from bronze into the silver clean text table."""

from __future__ import annotations

import argparse
import re

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

VALID_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


def quote_identifier(identifier: str) -> str:
    """Validate and quote a Unity Catalog identifier."""
    if not VALID_IDENTIFIER.fullmatch(identifier):
        raise ValueError(f"Invalid Unity Catalog identifier: {identifier!r}")

    return f"`{identifier}`"


def table_name(catalog: str, schema: str, table: str) -> str:
    """Build a fully qualified Unity Catalog table name."""
    return ".".join(
        [
            quote_identifier(catalog),
            quote_identifier(schema),
            quote_identifier(table),
        ]
    )


def build_silver_clean_text(
    spark: SparkSession,
    catalog: str,
    schema: str,
) -> DataFrame:
    """Build cleaned OCR text with basic text quality flags."""
    bronze_ocr_text_table = table_name(catalog, schema, "bronze_ocr_text")
    silver_documents_table = table_name(catalog, schema, "silver_documents")

    ocr_text = spark.table(bronze_ocr_text_table)

    documents = spark.table(silver_documents_table).select(
        "document_id",
        "title",
        "publication_year",
        "publication_date_raw",
        "language",
        "metadata_parse_status",
    )

    cleaned = (
        ocr_text.join(documents, on="document_id", how="left")
        .withColumn("raw_text", F.coalesce(F.col("raw_text"), F.lit("")))
        .withColumn("character_count", F.length(F.col("raw_text")))
        .withColumn("clean_text", F.trim(F.regexp_replace(F.col("raw_text"), r"\s+", " ")))
        .withColumn("clean_character_count", F.length(F.col("clean_text")))
        .withColumn("clean_text_lower", F.lower(F.col("clean_text")))
        .withColumn(
            "word_count",
            F.when(F.length(F.col("clean_text")) == 0, F.lit(0)).otherwise(
                F.size(F.split(F.col("clean_text"), r"\s+"))
            ),
        )
        .withColumn(
            "contains_encoding_artifacts",
            F.col("raw_text").rlike(r"Ã|Â|â€|�"),
        )
        .withColumn(
            "ocr_quality_flag",
            F.when(F.col("character_count") == 0, F.lit("empty_text"))
            .when(F.col("word_count") < 100, F.lit("very_short_text"))
            .when(F.col("contains_encoding_artifacts"), F.lit("possible_encoding_artifacts"))
            .otherwise(F.lit("usable")),
        )
        .withColumn("cleaned_at", F.current_timestamp())
        .select(
            "document_id",
            "title",
            "publication_year",
            "publication_date_raw",
            "language",
            "raw_text",
            "clean_text",
            "clean_text_lower",
            "word_count",
            "character_count",
            "clean_character_count",
            "contains_encoding_artifacts",
            "ocr_quality_flag",
            "metadata_parse_status",
            "file_path",
            "source_file_type",
            "ingested_at",
            "cleaned_at",
        )
    )

    return cleaned


def build_text_quality_summary(clean_text: DataFrame) -> DataFrame:
    """Build a compact quality summary table for the cleaned OCR text."""
    return (
        clean_text.groupBy("ocr_quality_flag")
        .agg(
            F.count("*").alias("document_count"),
            F.min("word_count").alias("min_word_count"),
            F.expr("percentile_approx(word_count, 0.5)").alias("median_word_count"),
            F.avg("word_count").alias("avg_word_count"),
            F.max("word_count").alias("max_word_count"),
            F.sum(F.when(F.col("publication_year").isNull(), 1).otherwise(0)).alias(
                "documents_missing_year"
            ),
            F.sum(F.when(F.col("contains_encoding_artifacts"), 1).otherwise(0)).alias(
                "documents_with_encoding_artifacts"
            ),
        )
        .withColumn("summary_created_at", F.current_timestamp())
        .orderBy("ocr_quality_flag")
    )


def write_delta_table(df: DataFrame, full_table_name: str) -> None:
    """Overwrite a Delta table."""
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
    """Create silver clean text and text quality summary tables."""
    args = parse_args()
    spark = SparkSession.builder.getOrCreate()

    clean_text = build_silver_clean_text(
        spark=spark,
        catalog=args.catalog,
        schema=args.schema,
    )

    quality_summary = build_text_quality_summary(clean_text)

    write_delta_table(
        clean_text,
        table_name(args.catalog, args.schema, "silver_clean_text"),
    )

    write_delta_table(
        quality_summary,
        table_name(args.catalog, args.schema, "silver_text_quality_summary"),
    )


if __name__ == "__main__":
    main()