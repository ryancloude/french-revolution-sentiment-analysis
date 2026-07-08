"""Build gold table of representative stance context passages."""

from __future__ import annotations

import argparse
import re

from pyspark.sql import DataFrame, SparkSession, Window
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


def build_top_stance_contexts(
    spark: SparkSession,
    catalog: str,
    schema: str,
) -> DataFrame:
    """Build a dashboard-ready evidence table of stance context windows."""
    mentions_table = table_name(catalog, schema, "silver_fact_figure_mentions")
    documents_table = table_name(catalog, schema, "silver_dim_documents")
    dates_table = table_name(catalog, schema, "silver_dim_dates")
    contexts_table = table_name(catalog, schema, "silver_dim_mention_contexts")
    stance_categories_table = table_name(catalog, schema, "silver_dim_stance_categories")
    stance_audit_table = table_name(catalog, schema, "silver_stance_model_audit")

    mentions = spark.table(mentions_table).drop(
        "publication_year",
        "publication_month",
        "publication_day",
        "publication_date",
        "date_precision",
        "date_calendar",
    )

    documents = spark.table(documents_table).select(
        "document_id",
        "title",
        "creator",
        "author",
        "language",
        "internet_archive_id",
        "source_url",
        "metadata_parse_status",
        "conflicts_with_metadata_year",
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

    contexts = spark.table(contexts_table).select(
        "mention_id",
        "context_window",
        "context_word_count",
        "context_start_char",
        "context_end_char",
    )

    stance_categories = spark.table(stance_categories_table).select(
        "stance_category_id",
        "stance_label",
        "stance_intensity",
        "stance_confidence",
        "target_relevance",
        "stance_score_method",
    )

    stance_audit = spark.table(stance_audit_table).select(
        "stance_audit_id",
        "mention_id",
        "stance_method",
        "stance_model",
        "model_stance_score_raw",
        "evidence_text",
        "evidence_translation_en",
        "explanation",
    )

    stance_rows = (
        mentions.join(documents, on="document_id", how="left")
        .join(dates, on="date_key", how="left")
        .join(contexts, on="mention_id", how="left")
        .join(stance_categories, on="stance_category_id", how="left")
        .join(stance_audit, on=["stance_audit_id", "mention_id"], how="left")
        .filter(F.col("stance_confidence").isin("medium", "high"))
        .filter(F.col("target_relevance").isin("direct", "indirect"))
        .filter(F.col("evidence_text").isNotNull())
        .filter(F.length(F.trim(F.col("evidence_text"))) > F.lit(0))
        .withColumn("abs_stance_score", F.abs(F.col("stance_score")))
        .withColumn(
            "confidence_rank",
            F.when(F.col("stance_confidence") == F.lit("high"), F.lit(2))
            .when(F.col("stance_confidence") == F.lit("medium"), F.lit(1))
            .otherwise(F.lit(0)),
        )
        .withColumn(
            "target_relevance_rank",
            F.when(F.col("target_relevance") == F.lit("direct"), F.lit(2))
            .when(F.col("target_relevance") == F.lit("indirect"), F.lit(1))
            .otherwise(F.lit(0)),
        )
    )

    figure_period_window = Window.partitionBy(
        "figure_id",
        "publication_year",
        "publication_month",
    )

    absolute_rank_window = figure_period_window.orderBy(
        F.desc("abs_stance_score"),
        F.desc("confidence_rank"),
        F.desc("target_relevance_rank"),
        F.asc("document_id"),
        F.asc("mention_id"),
    )

    positive_rank_window = figure_period_window.orderBy(
        F.desc("stance_score"),
        F.desc("confidence_rank"),
        F.desc("target_relevance_rank"),
        F.asc("document_id"),
        F.asc("mention_id"),
    )

    negative_rank_window = figure_period_window.orderBy(
        F.asc("stance_score"),
        F.desc("confidence_rank"),
        F.desc("target_relevance_rank"),
        F.asc("document_id"),
        F.asc("mention_id"),
    )

    return (
        stance_rows.withColumn(
            "absolute_stance_rank_by_figure_period",
            F.row_number().over(absolute_rank_window),
        )
        .withColumn(
            "positive_rank_by_figure_period",
            F.when(
                F.col("stance_label") == F.lit("positive"),
                F.row_number().over(positive_rank_window),
            ),
        )
        .withColumn(
            "negative_rank_by_figure_period",
            F.when(
                F.col("stance_label") == F.lit("negative"),
                F.row_number().over(negative_rank_window),
            ),
        )
        .withColumn(
            "is_analysis_ready",
            (F.col("publication_year").isNotNull())
            & (F.col("stance_confidence").isin("medium", "high"))
            & (F.col("target_relevance").isin("direct", "indirect"))
            & (~F.coalesce(F.col("conflicts_with_metadata_year"), F.lit(False))),
        )
        .withColumn("built_at", F.current_timestamp())
        .select(
            "stance_audit_id",
            "mention_id",
            "document_id",
            "figure_id",
            "canonical_name",
            "title",
            "creator",
            "author",
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
            "conflicts_with_metadata_year",
            "matched_variant",
            "variant_type",
            "match_confidence",
            "match_method",
            "stance_method",
            "stance_model",
            "stance_label",
            "stance_intensity",
            "stance_score",
            "abs_stance_score",
            "stance_score_method",
            "model_stance_score_raw",
            "stance_confidence",
            "target_relevance",
            "evidence_text",
            "evidence_translation_en",
            "explanation",
            "context_window",
            "context_word_count",
            "context_start_char",
            "context_end_char",
            "ocr_quality_flag",
            "absolute_stance_rank_by_figure_period",
            "positive_rank_by_figure_period",
            "negative_rank_by_figure_period",
            "is_analysis_ready",
            "built_at",
        )
        .orderBy(
            "publication_year",
            "publication_month",
            "canonical_name",
            "absolute_stance_rank_by_figure_period",
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
    """Build the gold top stance contexts table."""
    args = parse_args()
    spark = SparkSession.builder.getOrCreate()

    gold_table = build_top_stance_contexts(
        spark=spark,
        catalog=args.catalog,
        schema=args.schema,
    )

    write_delta_table(
        gold_table,
        table_name(args.catalog, args.schema, "gold_top_stance_contexts"),
    )


if __name__ == "__main__":
    main()