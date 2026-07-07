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


def build_top_stance_contexts(
    spark: SparkSession,
    catalog: str,
    schema: str,
) -> DataFrame:
    """Build a dashboard-ready evidence table of stance context windows."""
    stance_table = table_name(catalog, schema, "silver_context_stance_ai_query")
    documents_table = table_name(catalog, schema, "silver_documents")

    stance_rows = (
        spark.table(stance_table)
        .filter(F.col("stance_confidence").isin("medium", "high"))
        .filter(F.col("target_relevance").isin("direct", "indirect"))
        .filter(F.col("evidence_text").isNotNull())
        .filter(F.length(F.trim(F.col("evidence_text"))) > F.lit(0))
        .transform(add_period_label)
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
        stance_rows.join(documents, on="document_id", how="left")
        .withColumn(
            "title",
            F.coalesce(F.col("document_title"), F.col("title")),
        )
        .withColumn(
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
            & (~F.col("conflicts_with_metadata_year")),
        )
        .withColumn("built_at", F.current_timestamp())
        .select(
            "stance_candidate_id",
            "mention_id",
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