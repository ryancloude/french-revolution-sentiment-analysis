"""Build gold figure stance metrics by publication period."""

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


def build_figure_stance_by_period(
    spark: SparkSession,
    catalog: str,
    schema: str,
) -> DataFrame:
    """Aggregate context-level stance scores into a dashboard-ready gold table."""
    stance_table = table_name(catalog, schema, "silver_context_stance_ai_query")

    period_group_columns = [
        "period_label",
        "publication_year",
        "publication_month",
        "figure_id",
        "canonical_name",
    ]

    stance_rows = (
        spark.table(stance_table)
        .withColumn(
            "is_included",
            (F.col("stance_confidence").isin("medium", "high"))
            & F.col("stance_score").isNotNull(),
        )
        .transform(add_period_label)
    )

    mention_agg = stance_rows.groupBy(*period_group_columns).agg(
        F.count("*").alias("mention_count"),
        F.countDistinct("document_id").alias("document_count"),
        count_when(F.col("is_included")).alias("included_mention_count"),
        F.countDistinct(
            F.when(F.col("is_included"), F.col("document_id"))
        ).alias("included_document_count"),
        F.avg(
            F.when(F.col("is_included"), F.col("stance_score"))
        ).alias("avg_stance_score_mention_weighted"),
        F.avg(
            F.when(F.col("is_included"), F.abs(F.col("stance_score")))
        ).alias("avg_abs_stance_score_mention_weighted"),
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
        count_when(F.col("date_precision") == F.lit("day")).alias(
            "day_precision_count"
        ),
        count_when(F.col("date_precision") == F.lit("month")).alias(
            "month_precision_count"
        ),
        count_when(F.col("date_precision") == F.lit("year")).alias(
            "year_precision_count"
        ),
        count_when(F.col("conflicts_with_metadata_year")).alias(
            "metadata_year_conflict_count"
        ),
        F.min("publication_date").alias("min_publication_date"),
        F.max("publication_date").alias("max_publication_date"),
    )

    document_scores = (
        stance_rows.groupBy(*period_group_columns, "document_id")
        .agg(
            F.count("*").alias("document_mention_count"),
            count_when(F.col("is_included")).alias("included_document_mention_count"),
            F.avg(
                F.when(F.col("is_included"), F.col("stance_score"))
            ).alias("avg_document_stance_score"),
            F.avg(
                F.when(F.col("is_included"), F.abs(F.col("stance_score")))
            ).alias("avg_abs_document_stance_score"),
        )
        .withColumn(
            "document_stance_label",
            F.when(F.col("avg_document_stance_score") > F.lit(0), F.lit("positive"))
            .when(F.col("avg_document_stance_score") < F.lit(0), F.lit("negative"))
            .when(F.col("included_document_mention_count") > F.lit(0), F.lit("neutral"))
            .otherwise(F.lit("not_included")),
        )
    )

    document_agg = document_scores.groupBy(*period_group_columns).agg(
        F.avg(
            F.when(
                F.col("included_document_mention_count") > F.lit(0),
                F.col("avg_document_stance_score"),
            )
        ).alias("avg_stance_score_document_weighted"),
        F.avg(
            F.when(
                F.col("included_document_mention_count") > F.lit(0),
                F.col("avg_abs_document_stance_score"),
            )
        ).alias("avg_abs_stance_score_document_weighted"),
        count_when(F.col("document_stance_label") == F.lit("positive")).alias(
            "positive_document_count"
        ),
        count_when(F.col("document_stance_label") == F.lit("negative")).alias(
            "negative_document_count"
        ),
        count_when(F.col("document_stance_label") == F.lit("neutral")).alias(
            "neutral_document_count"
        ),
    )

    return (
        mention_agg.join(
            document_agg,
            on=period_group_columns,
            how="left",
        )
        .withColumn(
            "avg_stance_score",
            F.col("avg_stance_score_document_weighted"),
        )
        .withColumn(
            "avg_abs_stance_score",
            F.col("avg_abs_stance_score_document_weighted"),
        )
        .withColumn(
            "included_mention_share",
            safe_ratio("included_mention_count", "mention_count"),
        )
        .withColumn(
            "included_document_share",
            safe_ratio("included_document_count", "document_count"),
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
            & (F.col("metadata_year_conflict_count") == F.lit(0)),
        )
        .withColumn("built_at", F.current_timestamp())
        .select(
            "period_label",
            "publication_year",
            "publication_month",
            "figure_id",
            "canonical_name",
            "is_analysis_ready",
            "mention_count",
            "document_count",
            "included_mention_count",
            "included_document_count",
            "included_mention_share",
            "included_document_share",
            "avg_stance_score",
            "avg_abs_stance_score",
            "avg_stance_score_document_weighted",
            "avg_stance_score_mention_weighted",
            "avg_abs_stance_score_document_weighted",
            "avg_abs_stance_score_mention_weighted",
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
            "positive_document_count",
            "negative_document_count",
            "neutral_document_count",
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
            "day_precision_count",
            "month_precision_count",
            "year_precision_count",
            "metadata_year_conflict_count",
            "min_publication_date",
            "max_publication_date",
            "built_at",
        )
        .orderBy(
            "publication_year",
            "publication_month",
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
    """Build the gold figure stance by period table."""
    args = parse_args()
    spark = SparkSession.builder.getOrCreate()

    gold_table = build_figure_stance_by_period(
        spark=spark,
        catalog=args.catalog,
        schema=args.schema,
    )

    write_delta_table(
        gold_table,
        table_name(args.catalog, args.schema, "gold_figure_stance_by_period"),
    )


if __name__ == "__main__":
    main()