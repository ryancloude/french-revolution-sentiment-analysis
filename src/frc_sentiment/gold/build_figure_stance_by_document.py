"""Build gold figure stance metrics by source document."""

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


def add_period_label(df: DataFrame) -> DataFrame:
    """Add a dashboard-friendly period label."""
    return df.withColumn(
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


def build_figure_stance_by_document(
    spark: SparkSession,
    catalog: str,
    schema: str,
) -> DataFrame:
    """Aggregate context-level stance scores into document-level figure stance."""
    stance_table = table_name(catalog, schema, "silver_context_stance_ai_query")
    documents_table = table_name(catalog, schema, "silver_documents")

    stance_rows = (
        spark.table(stance_table)
        .withColumn(
            "is_included",
            (F.col("stance_confidence").isin("medium", "high"))
            & F.col("stance_score").isNotNull(),
        )
        .transform(add_period_label)
    )

    documents = spark.table(documents_table).select(
        "document_id",
        F.col("title").alias("document_title"),
        "creator",
        "author",
        "creators",
        "metadata_authors",
        "subjects",
        "subjects_raw",
        "language",
        "internet_archive_id",
        "source_url",
        "metadata_parse_status",
    )

    group_columns = [
        "document_id",
        "figure_id",
        "canonical_name",
    ]

    aggregated = stance_rows.groupBy(*group_columns).agg(
        F.first("period_label", ignorenulls=True).alias("period_label"),
        F.first("publication_year", ignorenulls=True).alias("publication_year"),
        F.first("publication_month", ignorenulls=True).alias("publication_month"),
        F.first("publication_day", ignorenulls=True).alias("publication_day"),
        F.first("publication_date", ignorenulls=True).alias("publication_date"),
        F.first("date_precision", ignorenulls=True).alias("date_precision"),
        F.first("date_calendar", ignorenulls=True).alias("date_calendar"),
        F.first("date_extractor_name", ignorenulls=True).alias("date_extractor_name"),
        F.first("date_source_field", ignorenulls=True).alias("date_source_field"),
        F.first("date_confidence", ignorenulls=True).alias("date_confidence"),
        F.max(F.col("conflicts_with_metadata_year").cast("int")).alias(
            "has_metadata_year_conflict_int"
        ),
        F.first("title", ignorenulls=True).alias("stance_title"),
        F.first("ocr_quality_flag", ignorenulls=True).alias("ocr_quality_flag"),
        F.count("*").alias("mention_count"),
        count_when(F.col("is_included")).alias("included_mention_count"),
        F.avg(F.when(F.col("is_included"), F.col("stance_score"))).alias(
            "avg_stance_score"
        ),
        F.avg(F.when(F.col("is_included"), F.abs(F.col("stance_score")))).alias(
            "avg_abs_stance_score"
        ),
        F.min(F.when(F.col("is_included"), F.col("stance_score"))).alias(
            "min_stance_score"
        ),
        F.max(F.when(F.col("is_included"), F.col("stance_score"))).alias(
            "max_stance_score"
        ),
        count_when(
            F.col("is_included") & (F.col("stance_label") == F.lit("positive"))
        ).alias("positive_mention_count"),
        count_when(
            F.col("is_included") & (F.col("stance_label") == F.lit("negative"))
        ).alias("negative_mention_count"),
        count_when(
            F.col("is_included")
            & (F.col("stance_label") == F.lit("neutral_or_unclear"))
        ).alias("neutral_or_unclear_mention_count"),
        count_when(
            F.col("is_included") & (F.col("stance_intensity") == F.lit("weak"))
        ).alias("weak_stance_mention_count"),
        count_when(
            F.col("is_included") & (F.col("stance_intensity") == F.lit("moderate"))
        ).alias("moderate_stance_mention_count"),
        count_when(
            F.col("is_included") & (F.col("stance_intensity") == F.lit("strong"))
        ).alias("strong_stance_mention_count"),
        count_when(
            F.col("is_included")
            & (F.col("stance_label") == F.lit("negative"))
            & (F.col("stance_intensity") == F.lit("strong"))
        ).alias("strong_negative_mention_count"),
        count_when(F.col("stance_confidence") == F.lit("high")).alias(
            "high_confidence_count"
        ),
        count_when(F.col("stance_confidence") == F.lit("medium")).alias(
            "medium_confidence_count"
        ),
        count_when(F.col("stance_confidence") == F.lit("low")).alias(
            "low_confidence_count"
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
        count_when(
            F.col("is_included") & (F.col("target_relevance") == F.lit("direct"))
        ).alias("included_direct_mention_count"),
        count_when(
            F.col("is_included") & (F.col("target_relevance") == F.lit("indirect"))
        ).alias("included_indirect_mention_count"),
        count_when(
            F.col("is_included")
            & (F.col("target_relevance") == F.lit("not_relevant"))
        ).alias("included_not_relevant_mention_count"),
    )

    return (
        aggregated.join(documents, on="document_id", how="left")
        .withColumn(
            "title",
            F.coalesce(F.col("document_title"), F.col("stance_title")),
        )
        .withColumn(
            "has_metadata_year_conflict",
            F.col("has_metadata_year_conflict_int") == F.lit(1),
        )
        .withColumn(
            "document_stance_label",
            F.when(F.col("avg_stance_score") > F.lit(0), F.lit("positive"))
            .when(F.col("avg_stance_score") < F.lit(0), F.lit("negative"))
            .when(F.col("included_mention_count") > F.lit(0), F.lit("neutral"))
            .otherwise(F.lit("not_included")),
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
            "is_analysis_ready",
            (F.col("publication_year").isNotNull())
            & (F.col("included_mention_count") > F.lit(0))
            & (~F.col("has_metadata_year_conflict")),
        )
        .withColumn("built_at", F.current_timestamp())
        .select(
            "document_id",
            "figure_id",
            "canonical_name",
            "title",
            "creator",
            "author",
            "creators",
            "metadata_authors",
            "subjects",
            "subjects_raw",
            "language",
            "internet_archive_id",
            "source_url",
            "metadata_parse_status",
            "period_label",
            "publication_year",
            "publication_month",
            "publication_day",
            "publication_date",
            "date_precision",
            "date_calendar",
            "date_extractor_name",
            "date_source_field",
            "date_confidence",
            "has_metadata_year_conflict",
            "ocr_quality_flag",
            "is_analysis_ready",
            "mention_count",
            "included_mention_count",
            "included_mention_share",
            "avg_stance_score",
            "avg_abs_stance_score",
            "min_stance_score",
            "max_stance_score",
            "document_stance_label",
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
            "built_at",
        )
        .orderBy(
            "publication_year",
            "publication_month",
            "canonical_name",
            "document_id",
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
    """Build the gold figure stance by document table."""
    args = parse_args()
    spark = SparkSession.builder.getOrCreate()

    gold_table = build_figure_stance_by_document(
        spark=spark,
        catalog=args.catalog,
        schema=args.schema,
    )

    write_delta_table(
        gold_table,
        table_name(args.catalog, args.schema, "gold_figure_stance_by_document"),
    )


if __name__ == "__main__":
    main()