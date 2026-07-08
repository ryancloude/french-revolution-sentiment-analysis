"""Build gold figure stance metrics by metadata subject and publication period."""

from __future__ import annotations

import argparse
import re

from pyspark.sql import Column, DataFrame, SparkSession
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


def count_when(condition: Column) -> Column:
    """Count rows matching a condition."""
    return F.sum(F.when(condition, F.lit(1)).otherwise(F.lit(0)))


def safe_ratio(numerator: str, denominator: str) -> Column:
    """Return numerator / denominator, or null when denominator is zero."""
    return F.when(
        F.col(denominator) > F.lit(0),
        F.col(numerator).cast("double") / F.col(denominator).cast("double"),
    ).otherwise(F.lit(None).cast("double"))


def build_figure_stance_by_subject_period(
    spark: SparkSession,
    catalog: str,
    schema: str,
) -> DataFrame:
    """Aggregate document-level figure stance by metadata subject and period."""
    document_stance_table = table_name(catalog, schema, "gold_figure_stance_by_document")
    bridge_subjects_table = table_name(catalog, schema, "silver_bridge_document_subjects")
    subjects_table = table_name(catalog, schema, "silver_dim_subjects")

    document_stance = spark.table(document_stance_table)

    document_subjects = (
        spark.table(bridge_subjects_table)
        .join(
            spark.table(subjects_table),
            on="subject_id",
            how="left",
        )
        .select(
            "document_id",
            "subject_id",
            "subject",
            "subject_source",
            "subject_position",
        )
    )

    subject_rows = document_stance.join(
        document_subjects,
        on="document_id",
        how="left",
    ).withColumn(
        "subject",
        F.coalesce(F.col("subject"), F.lit("Unknown")),
    )

    group_columns = [
        "period_label",
        "publication_year",
        "publication_month",
        "subject_id",
        "subject",
        "subject_source",
        "figure_id",
        "canonical_name",
    ]

    return (
        subject_rows.groupBy(*group_columns)
        .agg(
            F.countDistinct("document_id").alias("document_count"),
            F.sum("mention_count").alias("mention_count"),
            F.sum("included_mention_count").alias("included_mention_count"),
            F.avg(
                F.when(F.col("is_analysis_ready"), F.col("avg_stance_score"))
            ).alias("avg_stance_score"),
            F.avg(
                F.when(F.col("is_analysis_ready"), F.col("avg_abs_stance_score"))
            ).alias("avg_abs_stance_score"),
            count_when(
                F.col("is_analysis_ready")
                & (F.col("document_stance_label") == F.lit("positive"))
            ).alias("positive_document_count"),
            count_when(
                F.col("is_analysis_ready")
                & (F.col("document_stance_label") == F.lit("negative"))
            ).alias("negative_document_count"),
            count_when(
                F.col("is_analysis_ready")
                & (F.col("document_stance_label") == F.lit("neutral"))
            ).alias("neutral_document_count"),
            F.sum("positive_mention_count").alias("positive_mention_count"),
            F.sum("negative_mention_count").alias("negative_mention_count"),
            F.sum("neutral_or_unclear_mention_count").alias(
                "neutral_or_unclear_mention_count"
            ),
            F.sum("weak_stance_mention_count").alias("weak_stance_mention_count"),
            F.sum("moderate_stance_mention_count").alias(
                "moderate_stance_mention_count"
            ),
            F.sum("strong_stance_mention_count").alias("strong_stance_mention_count"),
            F.sum("strong_negative_mention_count").alias(
                "strong_negative_mention_count"
            ),
            F.sum("high_confidence_count").alias("high_confidence_count"),
            F.sum("medium_confidence_count").alias("medium_confidence_count"),
            F.sum("low_confidence_count").alias("low_confidence_count"),
            F.sum("direct_mention_count").alias("direct_mention_count"),
            F.sum("indirect_mention_count").alias("indirect_mention_count"),
            F.sum("not_relevant_mention_count").alias("not_relevant_mention_count"),
            F.sum("included_direct_mention_count").alias(
                "included_direct_mention_count"
            ),
            F.sum("included_indirect_mention_count").alias(
                "included_indirect_mention_count"
            ),
            F.sum("included_not_relevant_mention_count").alias(
                "included_not_relevant_mention_count"
            ),
            count_when(F.col("is_analysis_ready")).alias("analysis_ready_document_count"),
            count_when(~F.col("is_analysis_ready")).alias(
                "not_analysis_ready_document_count"
            ),
            F.min("publication_date").alias("min_publication_date"),
            F.max("publication_date").alias("max_publication_date"),
        )
        .withColumn(
            "included_mention_share",
            safe_ratio("included_mention_count", "mention_count"),
        )
        .withColumn(
            "positive_mention_share",
            safe_ratio("positive_mention_count", "included_mention_count"),
        )
        .withColumn(
            "negative_mention_share",
            safe_ratio("negative_mention_count", "included_mention_count"),
        )
        .withColumn(
            "neutral_or_unclear_mention_share",
            safe_ratio(
                "neutral_or_unclear_mention_count",
                "included_mention_count",
            ),
        )
        .withColumn(
            "strong_negative_mention_share",
            safe_ratio("strong_negative_mention_count", "included_mention_count"),
        )
        .withColumn(
            "direct_relevance_share",
            safe_ratio("included_direct_mention_count", "included_mention_count"),
        )
        .withColumn(
            "analysis_ready_document_share",
            safe_ratio("analysis_ready_document_count", "document_count"),
        )
        .withColumn(
            "is_analysis_ready",
            (F.col("publication_year").isNotNull())
            & (F.col("analysis_ready_document_count") > F.lit(0)),
        )
        .withColumn("built_at", F.current_timestamp())
        .select(
            "period_label",
            "publication_year",
            "publication_month",
            "subject_id",
            "subject",
            "subject_source",
            "figure_id",
            "canonical_name",
            "is_analysis_ready",
            "document_count",
            "analysis_ready_document_count",
            "not_analysis_ready_document_count",
            "analysis_ready_document_share",
            "mention_count",
            "included_mention_count",
            "included_mention_share",
            "avg_stance_score",
            "avg_abs_stance_score",
            "positive_document_count",
            "negative_document_count",
            "neutral_document_count",
            "positive_mention_count",
            "negative_mention_count",
            "neutral_or_unclear_mention_count",
            "positive_mention_share",
            "negative_mention_share",
            "neutral_or_unclear_mention_share",
            "weak_stance_mention_count",
            "moderate_stance_mention_count",
            "strong_stance_mention_count",
            "strong_negative_mention_count",
            "strong_negative_mention_share",
            "high_confidence_count",
            "medium_confidence_count",
            "low_confidence_count",
            "direct_mention_count",
            "indirect_mention_count",
            "not_relevant_mention_count",
            "included_direct_mention_count",
            "included_indirect_mention_count",
            "included_not_relevant_mention_count",
            "direct_relevance_share",
            "min_publication_date",
            "max_publication_date",
            "built_at",
        )
        .orderBy(
            "publication_year",
            "publication_month",
            "subject",
            "canonical_name",
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
    """Build the gold figure stance by subject period table."""
    args = parse_args()
    spark = SparkSession.builder.getOrCreate()

    gold_table = build_figure_stance_by_subject_period(
        spark=spark,
        catalog=args.catalog,
        schema=args.schema,
    )

    write_delta_table(
        gold_table,
        table_name(args.catalog, args.schema, "gold_figure_stance_by_subject_period"),
    )


if __name__ == "__main__":
    main()