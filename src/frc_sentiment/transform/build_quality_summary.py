"""Build compact data quality summary tables for the current pipeline outputs."""

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


def scalar_count(spark: SparkSession, full_table_name: str) -> int:
    """Return the row count for a table."""
    return spark.table(full_table_name).count()


def scalar_filtered_count(spark: SparkSession, full_table_name: str, condition: str) -> int:
    """Return the row count for a table after applying a SQL condition."""
    return spark.table(full_table_name).where(condition).count()


def make_metric_rows(spark: SparkSession, rows: list[tuple[str, str, int]]) -> DataFrame:
    """Create a DataFrame from metric rows."""
    return spark.createDataFrame(
        rows,
        schema="metric_group string, metric_name string, metric_value long",
    ).withColumn("summary_created_at", F.current_timestamp())


def build_data_quality_summary(spark: SparkSession, catalog: str, schema: str) -> DataFrame:
    """Build one-row-per-metric pipeline quality summary."""
    bronze_metadata = table_name(catalog, schema, "bronze_metadata")
    bronze_ocr_text = table_name(catalog, schema, "bronze_ocr_text")
    silver_documents = table_name(catalog, schema, "silver_documents")
    silver_clean_text = table_name(catalog, schema, "silver_clean_text")
    silver_figures = table_name(catalog, schema, "silver_figures")
    silver_entity_mentions = table_name(catalog, schema, "silver_entity_mentions")
    silver_context_windows = table_name(catalog, schema, "silver_context_windows")

    rows = [
        (
            "bronze",
            "bronze_metadata_rows",
            scalar_count(spark, bronze_metadata),
        ),
        (
            "bronze",
            "bronze_ocr_text_rows",
            scalar_count(spark, bronze_ocr_text),
        ),
        (
            "metadata",
            "silver_documents_rows",
            scalar_count(spark, silver_documents),
        ),
        (
            "metadata",
            "documents_with_valid_year",
            scalar_filtered_count(spark, silver_documents, "publication_year IS NOT NULL"),
        ),
        (
            "metadata",
            "documents_missing_year",
            scalar_filtered_count(spark, silver_documents, "publication_year IS NULL"),
        ),
        (
            "metadata",
            "documents_with_ocr_text",
            scalar_filtered_count(spark, silver_documents, "has_ocr_text = true"),
        ),
        (
            "metadata",
            "documents_without_ocr_text",
            scalar_filtered_count(spark, silver_documents, "has_ocr_text = false"),
        ),
        (
            "text_quality",
            "silver_clean_text_rows",
            scalar_count(spark, silver_clean_text),
        ),
        (
            "text_quality",
            "usable_ocr_documents",
            scalar_filtered_count(spark, silver_clean_text, "ocr_quality_flag = 'usable'"),
        ),
        (
            "text_quality",
            "very_short_text_documents",
            scalar_filtered_count(
                spark,
                silver_clean_text,
                "ocr_quality_flag = 'very_short_text'",
            ),
        ),
        (
            "text_quality",
            "empty_text_documents",
            scalar_filtered_count(spark, silver_clean_text, "ocr_quality_flag = 'empty_text'"),
        ),
        (
            "text_quality",
            "possible_encoding_artifact_documents",
            scalar_filtered_count(
                spark,
                silver_clean_text,
                "ocr_quality_flag = 'possible_encoding_artifacts'",
            ),
        ),
        (
            "figures",
            "figure_variant_rows",
            scalar_count(spark, silver_figures),
        ),
        (
            "figures",
            "distinct_figures",
            spark.table(silver_figures).select("figure_id").distinct().count(),
        ),
        (
            "entity_mentions",
            "entity_mentions",
            scalar_count(spark, silver_entity_mentions),
        ),
        (
            "entity_mentions",
            "high_confidence_mentions",
            scalar_filtered_count(
                spark,
                silver_entity_mentions,
                "match_confidence = 'high'",
            ),
        ),
        (
            "entity_mentions",
            "medium_confidence_mentions",
            scalar_filtered_count(
                spark,
                silver_entity_mentions,
                "match_confidence = 'medium'",
            ),
        ),
        (
            "entity_mentions",
            "low_confidence_mentions",
            scalar_filtered_count(
                spark,
                silver_entity_mentions,
                "match_confidence = 'low'",
            ),
        ),
        (
            "context_windows",
            "context_windows",
            scalar_count(spark, silver_context_windows),
        ),
        (
            "context_windows",
            "empty_context_windows",
            scalar_filtered_count(spark, silver_context_windows, "context_window = ''"),
        ),
    ]

    return make_metric_rows(spark, rows)


def build_figure_mention_summary(spark: SparkSession, catalog: str, schema: str) -> DataFrame:
    """Build mention counts by figure and matched variant."""
    silver_entity_mentions = table_name(catalog, schema, "silver_entity_mentions")

    return (
        spark.table(silver_entity_mentions)
        .groupBy(
            "figure_id",
            "canonical_name",
            "matched_variant",
            "variant_type",
            "match_confidence",
        )
        .agg(
            F.count("*").alias("mention_count"),
            F.countDistinct("document_id").alias("document_count"),
            F.min("publication_year").alias("min_publication_year"),
            F.max("publication_year").alias("max_publication_year"),
        )
        .withColumn("summary_created_at", F.current_timestamp())
        .orderBy(F.desc("mention_count"), "canonical_name", "matched_variant")
    )


def write_delta_table(df: DataFrame, full_table_name: str) -> None:
    """Overwrite a Delta table."""
    row_count = df.count()

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
    """Create data quality summary tables."""
    args = parse_args()
    spark = SparkSession.builder.getOrCreate()

    data_quality_summary = build_data_quality_summary(
        spark=spark,
        catalog=args.catalog,
        schema=args.schema,
    )
    figure_mention_summary = build_figure_mention_summary(
        spark=spark,
        catalog=args.catalog,
        schema=args.schema,
    )

    write_delta_table(
        data_quality_summary,
        table_name(args.catalog, args.schema, "silver_data_quality_summary"),
    )
    write_delta_table(
        figure_mention_summary,
        table_name(args.catalog, args.schema, "silver_figure_mention_summary"),
    )


if __name__ == "__main__":
    main()