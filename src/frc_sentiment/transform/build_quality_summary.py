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


def scalar_distinct_count(
    spark: SparkSession,
    full_table_name: str,
    column_name: str,
) -> int:
    """Return the distinct count for one column."""
    return spark.table(full_table_name).select(column_name).distinct().count()


def scalar_filtered_count(
    spark: SparkSession,
    full_table_name: str,
    condition: str,
) -> int:
    """Return the row count for a table after applying a SQL condition."""
    return spark.table(full_table_name).where(condition).count()


def make_metric_rows(spark: SparkSession, rows: list[tuple[str, str, int]]) -> DataFrame:
    """Create a DataFrame from metric rows."""
    return spark.createDataFrame(
        rows,
        schema="metric_group string, metric_name string, metric_value long",
    ).withColumn("summary_created_at", F.current_timestamp())


def append_table_count(
    rows: list[tuple[str, str, int]],
    spark: SparkSession,
    metric_group: str,
    metric_name: str,
    full_table_name: str,
) -> None:
    """Append a table row-count metric."""
    rows.append(
        (
            metric_group,
            metric_name,
            scalar_count(spark, full_table_name),
        )
    )


def append_filtered_count(
    rows: list[tuple[str, str, int]],
    spark: SparkSession,
    metric_group: str,
    metric_name: str,
    full_table_name: str,
    condition: str,
) -> None:
    """Append a filtered row-count metric."""
    rows.append(
        (
            metric_group,
            metric_name,
            scalar_filtered_count(spark, full_table_name, condition),
        )
    )


def build_data_quality_summary(
    spark: SparkSession,
    catalog: str,
    schema: str,
) -> DataFrame:
    """Build one-row-per-metric pipeline quality summary."""
    bronze_metadata = table_name(catalog, schema, "bronze_metadata")
    bronze_ocr_text = table_name(catalog, schema, "bronze_ocr_text")

    silver_documents = table_name(catalog, schema, "silver_documents")
    silver_clean_text = table_name(catalog, schema, "silver_clean_text")
    silver_figures = table_name(catalog, schema, "silver_figures")
    silver_entity_mentions = table_name(catalog, schema, "silver_entity_mentions")
    silver_context_windows = table_name(catalog, schema, "silver_context_windows")
    silver_date_candidates = table_name(catalog, schema, "silver_date_candidates")
    silver_ai_date_candidates = table_name(
        catalog,
        schema,
        "silver_date_candidates_ai_extract",
    )
    silver_selected_dates = table_name(catalog, schema, "silver_selected_publication_dates")
    silver_context_stance = table_name(catalog, schema, "silver_context_stance_ai_query")

    gold_mentions_by_period = table_name(catalog, schema, "gold_figure_mentions_by_period")
    gold_stance_by_period = table_name(catalog, schema, "gold_figure_stance_by_period")
    gold_stance_by_document = table_name(catalog, schema, "gold_figure_stance_by_document")
    gold_top_contexts = table_name(catalog, schema, "gold_top_stance_contexts")
    gold_stance_by_creator = table_name(
        catalog,
        schema,
        "gold_figure_stance_by_creator_period",
    )
    gold_stance_by_subject = table_name(
        catalog,
        schema,
        "gold_figure_stance_by_subject_period",
    )

    rows: list[tuple[str, str, int]] = []

    append_table_count(rows, spark, "bronze", "bronze_metadata_rows", bronze_metadata)
    append_table_count(rows, spark, "bronze", "bronze_ocr_text_rows", bronze_ocr_text)

    append_table_count(rows, spark, "metadata", "silver_documents_rows", silver_documents)
    append_filtered_count(
        rows,
        spark,
        "metadata",
        "documents_with_valid_year",
        silver_documents,
        "publication_year IS NOT NULL",
    )
    append_filtered_count(
        rows,
        spark,
        "metadata",
        "documents_missing_year",
        silver_documents,
        "publication_year IS NULL",
    )
    append_filtered_count(
        rows,
        spark,
        "metadata",
        "documents_with_publication_month",
        silver_documents,
        "publication_month IS NOT NULL",
    )
    append_filtered_count(
        rows,
        spark,
        "metadata",
        "documents_with_publication_day",
        silver_documents,
        "publication_day IS NOT NULL",
    )
    append_filtered_count(
        rows,
        spark,
        "metadata",
        "documents_with_creator",
        silver_documents,
        "creator IS NOT NULL",
    )
    append_filtered_count(
        rows,
        spark,
        "metadata",
        "documents_with_author",
        silver_documents,
        "author IS NOT NULL",
    )
    append_filtered_count(
        rows,
        spark,
        "metadata",
        "documents_with_multiple_creators",
        silver_documents,
        "size(creators) > 1",
    )
    append_filtered_count(
        rows,
        spark,
        "metadata",
        "documents_with_subjects",
        silver_documents,
        "size(subjects) > 0",
    )
    append_filtered_count(
        rows,
        spark,
        "metadata",
        "documents_with_ocr_text",
        silver_documents,
        "has_ocr_text = true",
    )
    append_filtered_count(
        rows,
        spark,
        "metadata",
        "documents_without_ocr_text",
        silver_documents,
        "has_ocr_text = false",
    )

    append_table_count(rows, spark, "text_quality", "silver_clean_text_rows", silver_clean_text)
    append_filtered_count(
        rows,
        spark,
        "text_quality",
        "usable_ocr_documents",
        silver_clean_text,
        "ocr_quality_flag = 'usable'",
    )
    append_filtered_count(
        rows,
        spark,
        "text_quality",
        "very_short_text_documents",
        silver_clean_text,
        "ocr_quality_flag = 'very_short_text'",
    )
    append_filtered_count(
        rows,
        spark,
        "text_quality",
        "empty_text_documents",
        silver_clean_text,
        "ocr_quality_flag = 'empty_text'",
    )
    append_filtered_count(
        rows,
        spark,
        "text_quality",
        "possible_encoding_artifact_documents",
        silver_clean_text,
        "ocr_quality_flag = 'possible_encoding_artifacts'",
    )

    append_table_count(rows, spark, "figures", "figure_variant_rows", silver_figures)
    rows.append(
        (
            "figures",
            "distinct_figures",
            scalar_distinct_count(spark, silver_figures, "figure_id"),
        )
    )

    append_table_count(
        rows,
        spark,
        "entity_mentions",
        "silver_entity_mentions_rows",
        silver_entity_mentions,
    )
    append_filtered_count(
        rows,
        spark,
        "entity_mentions",
        "high_confidence_mentions",
        silver_entity_mentions,
        "match_confidence = 'high'",
    )
    append_filtered_count(
        rows,
        spark,
        "entity_mentions",
        "medium_confidence_mentions",
        silver_entity_mentions,
        "match_confidence = 'medium'",
    )
    append_filtered_count(
        rows,
        spark,
        "entity_mentions",
        "low_confidence_mentions",
        silver_entity_mentions,
        "match_confidence = 'low'",
    )

    append_table_count(
        rows,
        spark,
        "context_windows",
        "silver_context_windows_rows",
        silver_context_windows,
    )
    append_filtered_count(
        rows,
        spark,
        "context_windows",
        "empty_context_windows",
        silver_context_windows,
        "context_window = ''",
    )

    append_table_count(
        rows,
        spark,
        "date_candidates",
        "rule_based_date_candidate_rows",
        silver_date_candidates,
    )
    append_table_count(
        rows,
        spark,
        "date_candidates",
        "ai_extract_date_candidate_rows",
        silver_ai_date_candidates,
    )
    append_table_count(
        rows,
        spark,
        "date_candidates",
        "selected_publication_date_rows",
        silver_selected_dates,
    )
    append_filtered_count(
        rows,
        spark,
        "date_candidates",
        "selected_dates_with_year",
        silver_selected_dates,
        "selected_publication_year IS NOT NULL",
    )
    append_filtered_count(
        rows,
        spark,
        "date_candidates",
        "selected_dates_with_month",
        silver_selected_dates,
        "selected_publication_month IS NOT NULL",
    )
    append_filtered_count(
        rows,
        spark,
        "date_candidates",
        "selected_dates_with_day",
        silver_selected_dates,
        "selected_publication_day IS NOT NULL",
    )
    append_filtered_count(
        rows,
        spark,
        "date_candidates",
        "selected_dates_conflicting_with_metadata_year",
        silver_selected_dates,
        "conflicts_with_metadata_year = true",
    )

    append_table_count(
        rows,
        spark,
        "stance",
        "silver_context_stance_rows",
        silver_context_stance,
    )
    append_filtered_count(
        rows,
        spark,
        "stance",
        "stance_high_confidence_rows",
        silver_context_stance,
        "stance_confidence = 'high'",
    )
    append_filtered_count(
        rows,
        spark,
        "stance",
        "stance_medium_confidence_rows",
        silver_context_stance,
        "stance_confidence = 'medium'",
    )
    append_filtered_count(
        rows,
        spark,
        "stance",
        "stance_low_confidence_rows",
        silver_context_stance,
        "stance_confidence = 'low'",
    )
    append_filtered_count(
        rows,
        spark,
        "stance",
        "stance_included_rows",
        silver_context_stance,
        "stance_confidence IN ('medium', 'high')",
    )
    append_filtered_count(
        rows,
        spark,
        "stance",
        "stance_positive_rows",
        silver_context_stance,
        "stance_label = 'positive'",
    )
    append_filtered_count(
        rows,
        spark,
        "stance",
        "stance_negative_rows",
        silver_context_stance,
        "stance_label = 'negative'",
    )
    append_filtered_count(
        rows,
        spark,
        "stance",
        "stance_neutral_or_unclear_rows",
        silver_context_stance,
        "stance_label = 'neutral_or_unclear'",
    )
    append_filtered_count(
        rows,
        spark,
        "stance",
        "stance_direct_rows",
        silver_context_stance,
        "target_relevance = 'direct'",
    )
    append_filtered_count(
        rows,
        spark,
        "stance",
        "stance_indirect_rows",
        silver_context_stance,
        "target_relevance = 'indirect'",
    )
    append_filtered_count(
        rows,
        spark,
        "stance",
        "stance_not_relevant_rows",
        silver_context_stance,
        "target_relevance = 'not_relevant'",
    )

    append_table_count(
        rows,
        spark,
        "gold",
        "gold_figure_mentions_by_period_rows",
        gold_mentions_by_period,
    )
    append_table_count(
        rows,
        spark,
        "gold",
        "gold_figure_stance_by_period_rows",
        gold_stance_by_period,
    )
    append_table_count(
        rows,
        spark,
        "gold",
        "gold_figure_stance_by_document_rows",
        gold_stance_by_document,
    )
    append_table_count(
        rows,
        spark,
        "gold",
        "gold_top_stance_contexts_rows",
        gold_top_contexts,
    )
    append_table_count(
        rows,
        spark,
        "gold",
        "gold_figure_stance_by_creator_period_rows",
        gold_stance_by_creator,
    )
    append_table_count(
        rows,
        spark,
        "gold",
        "gold_figure_stance_by_subject_period_rows",
        gold_stance_by_subject,
    )
    append_filtered_count(
        rows,
        spark,
        "gold",
        "analysis_ready_figure_stance_period_rows",
        gold_stance_by_period,
        "is_analysis_ready = true",
    )
    append_filtered_count(
        rows,
        spark,
        "gold",
        "analysis_ready_figure_stance_document_rows",
        gold_stance_by_document,
        "is_analysis_ready = true",
    )
    append_filtered_count(
        rows,
        spark,
        "gold",
        "analysis_ready_top_stance_context_rows",
        gold_top_contexts,
        "is_analysis_ready = true",
    )
    append_filtered_count(
        rows,
        spark,
        "gold",
        "analysis_ready_creator_period_rows",
        gold_stance_by_creator,
        "is_analysis_ready = true",
    )
    append_filtered_count(
        rows,
        spark,
        "gold",
        "analysis_ready_subject_period_rows",
        gold_stance_by_subject,
        "is_analysis_ready = true",
    )

    return make_metric_rows(spark, rows)


def build_figure_mention_summary(
    spark: SparkSession,
    catalog: str,
    schema: str,
) -> DataFrame:
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


def build_stance_distribution_summary(
    spark: SparkSession,
    catalog: str,
    schema: str,
) -> DataFrame:
    """Build stance output distribution summary for model monitoring."""
    silver_context_stance = table_name(catalog, schema, "silver_context_stance_ai_query")

    return (
        spark.table(silver_context_stance)
        .groupBy(
            "figure_id",
            "canonical_name",
            "stance_label",
            "stance_intensity",
            "stance_score",
            "stance_confidence",
            "target_relevance",
        )
        .agg(
            F.count("*").alias("stance_row_count"),
            F.countDistinct("document_id").alias("document_count"),
            F.countDistinct("mention_id").alias("mention_count"),
            F.avg("abs_stance_score").alias("avg_abs_stance_score")
            if "abs_stance_score" in spark.table(silver_context_stance).columns
            else F.avg(F.abs(F.col("stance_score"))).alias("avg_abs_stance_score"),
        )
        .withColumn("summary_created_at", F.current_timestamp())
        .orderBy(
            F.desc("stance_row_count"),
            "canonical_name",
            "stance_label",
            "stance_intensity",
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
    stance_distribution_summary = build_stance_distribution_summary(
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
    write_delta_table(
        stance_distribution_summary,
        table_name(args.catalog, args.schema, "silver_stance_distribution_summary"),
    )


if __name__ == "__main__":
    main()