"""Build gold figure mention counts by publication period."""

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


def build_figure_mentions_by_period(
    spark: SparkSession,
    catalog: str,
    schema: str,
) -> DataFrame:
    """Aggregate figure mentions into a dashboard-ready table."""
    mentions_table = table_name(catalog, schema, "silver_fact_figure_mentions")
    dates_table = table_name(catalog, schema, "silver_dim_dates")
    stance_categories_table = table_name(catalog, schema, "silver_dim_stance_categories")

    mentions = spark.table(mentions_table).drop(
        "publication_year",
        "publication_month",
        "publication_day",
        "publication_date",
        "date_precision",
        "date_calendar",
    )

    dates = spark.table(dates_table).select(
        "date_key",
        "period_label",
        "publication_year",
        "publication_month",
        "publication_day",
        "publication_date",
        "date_precision",
        "date_calendar",
    )

    stance_categories = spark.table(stance_categories_table).select(
        "stance_category_id",
        "stance_label",
        "stance_intensity",
        "stance_confidence",
        "target_relevance",
    )

    mention_rows = (
        mentions.join(dates, on="date_key", how="left")
        .join(stance_categories, on="stance_category_id", how="left")
        .withColumn(
            "is_included_stance",
            (F.col("stance_confidence").isin("medium", "high"))
            & F.col("stance_score").isNotNull(),
        )
    )

    return (
        mention_rows.groupBy(
            "period_label",
            "publication_year",
            "publication_month",
            "date_precision",
            "date_calendar",
            "figure_id",
            "canonical_name",
            "match_confidence",
        )
        .agg(
            F.count("*").alias("mention_count"),
            F.countDistinct("document_id").alias("document_count"),
            F.countDistinct("variant_id").alias("matched_variant_count"),
            count_when(F.col("is_high_confidence_match")).alias(
                "high_confidence_match_count"
            ),
            count_when(F.col("is_analysis_ready")).alias(
                "analysis_ready_mention_count"
            ),
            F.countDistinct(
                F.when(F.col("is_analysis_ready"), F.col("document_id"))
            ).alias("analysis_ready_document_count"),
            count_when(F.col("is_included_stance")).alias(
                "included_stance_mention_count"
            ),
            F.countDistinct(
                F.when(F.col("is_included_stance"), F.col("document_id"))
            ).alias("included_stance_document_count"),
            count_when(
                F.col("is_included_stance")
                & (F.col("stance_label") == F.lit("positive"))
            ).alias("positive_mention_count"),
            count_when(
                F.col("is_included_stance")
                & (F.col("stance_label") == F.lit("negative"))
            ).alias("negative_mention_count"),
            count_when(
                F.col("is_included_stance")
                & (F.col("stance_label") == F.lit("neutral_or_unclear"))
            ).alias("neutral_or_unclear_mention_count"),
            count_when(
                F.col("is_included_stance")
                & (F.col("stance_intensity") == F.lit("weak"))
            ).alias("weak_stance_mention_count"),
            count_when(
                F.col("is_included_stance")
                & (F.col("stance_intensity") == F.lit("moderate"))
            ).alias("moderate_stance_mention_count"),
            count_when(
                F.col("is_included_stance")
                & (F.col("stance_intensity") == F.lit("strong"))
            ).alias("strong_stance_mention_count"),
            count_when(F.col("stance_confidence") == F.lit("high")).alias(
                "high_stance_confidence_count"
            ),
            count_when(F.col("stance_confidence") == F.lit("medium")).alias(
                "medium_stance_confidence_count"
            ),
            count_when(F.col("stance_confidence") == F.lit("low")).alias(
                "low_stance_confidence_count"
            ),
            count_when(F.col("target_relevance") == F.lit("direct")).alias(
                "direct_mention_count"
            ),
            count_when(F.col("target_relevance") == F.lit("indirect")).alias(
                "indirect_mention_count"
            ),
            count_when(F.col("target_relevance") == F.lit("not_relevant")).alias(
                "not_relevant_mention_count"
            ),
            F.min("publication_date").alias("min_publication_date"),
            F.max("publication_date").alias("max_publication_date"),
        )
        .withColumn(
            "is_high_confidence_match_group",
            F.col("match_confidence") == F.lit("high"),
        )
        .withColumn(
            "is_analysis_ready",
            (F.col("publication_year").isNotNull())
            & (F.col("analysis_ready_mention_count") > F.lit(0)),
        )
        .withColumn(
            "included_stance_mention_share",
            safe_ratio("included_stance_mention_count", "mention_count"),
        )
        .withColumn(
            "positive_mention_share",
            safe_ratio("positive_mention_count", "included_stance_mention_count"),
        )
        .withColumn(
            "negative_mention_share",
            safe_ratio("negative_mention_count", "included_stance_mention_count"),
        )
        .withColumn(
            "neutral_or_unclear_mention_share",
            safe_ratio(
                "neutral_or_unclear_mention_count",
                "included_stance_mention_count",
            ),
        )
        .withColumn("built_at", F.current_timestamp())
        .select(
            "period_label",
            "publication_year",
            "publication_month",
            "date_precision",
            "date_calendar",
            "figure_id",
            "canonical_name",
            "match_confidence",
            "is_high_confidence_match_group",
            "is_analysis_ready",
            "mention_count",
            "document_count",
            "matched_variant_count",
            "high_confidence_match_count",
            "analysis_ready_mention_count",
            "analysis_ready_document_count",
            "included_stance_mention_count",
            "included_stance_document_count",
            "included_stance_mention_share",
            "positive_mention_count",
            "negative_mention_count",
            "neutral_or_unclear_mention_count",
            "positive_mention_share",
            "negative_mention_share",
            "neutral_or_unclear_mention_share",
            "weak_stance_mention_count",
            "moderate_stance_mention_count",
            "strong_stance_mention_count",
            "high_stance_confidence_count",
            "medium_stance_confidence_count",
            "low_stance_confidence_count",
            "direct_mention_count",
            "indirect_mention_count",
            "not_relevant_mention_count",
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