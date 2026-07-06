"""Build gold figure mention counts by publication period."""

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


def build_figure_mentions_by_period(
    spark: SparkSession,
    catalog: str,
    schema: str,
) -> DataFrame:
    """Aggregate figure mentions into a dashboard-ready table."""
    mentions_table = table_name(catalog, schema, "silver_entity_mentions")

    mentions = spark.table(mentions_table)

    return (
        mentions.groupBy(
            "publication_year",
            "publication_month",
            "date_precision",
            "date_calendar",
            "date_extractor_name",
            "date_confidence",
            "conflicts_with_metadata_year",
            "figure_id",
            "canonical_name",
            "match_confidence",
        )
        .agg(
            F.count("*").alias("mention_count"),
            F.countDistinct("document_id").alias("document_count"),
            F.countDistinct("matched_variant").alias("matched_variant_count"),
            F.min("publication_date").alias("min_publication_date"),
            F.max("publication_date").alias("max_publication_date"),
        )
        .withColumn(
            "period_label",
            F.when(
                F.col("publication_year").isNull(),
                F.lit("Unknown"),
            )
            .when(
                F.col("publication_month").isNull(),
                F.col("publication_year").cast("string"),
            )
            .otherwise(
                F.concat_ws(
                    "-",
                    F.col("publication_year").cast("string"),
                    F.lpad(F.col("publication_month").cast("string"), 2, "0"),
                )
            ),
        )
        .withColumn(
            "is_high_confidence_match",
            F.col("match_confidence") == F.lit("high"),
        )
        .withColumn(
            "is_analysis_ready",
            (F.col("publication_year").isNotNull())
            & (F.col("conflicts_with_metadata_year") == F.lit(False)),
        )
        .withColumn("built_at", F.current_timestamp())
        .select(
            "period_label",
            "publication_year",
            "publication_month",
            "date_precision",
            "date_calendar",
            "date_extractor_name",
            "date_confidence",
            "conflicts_with_metadata_year",
            "figure_id",
            "canonical_name",
            "match_confidence",
            "is_high_confidence_match",
            "is_analysis_ready",
            "mention_count",
            "document_count",
            "matched_variant_count",
            "min_publication_date",
            "max_publication_date",
            "built_at",
        )
        .orderBy(
            "publication_year",
            "publication_month",
            "canonical_name",
            "match_confidence",
        )
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
    """Build the gold figure mentions by period table."""
    args = parse_args()
    spark = SparkSession.builder.getOrCreate()

    gold_table = build_figure_mentions_by_period(
        spark=spark,
        catalog=args.catalog,
        schema=args.schema,
    )

    write_delta_table(
        gold_table,
        table_name(args.catalog, args.schema, "gold_figure_mentions_by_period"),
    )


if __name__ == "__main__":
    main()