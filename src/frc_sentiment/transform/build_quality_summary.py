"""Build compact data quality summary tables for the dimensional pipeline outputs."""

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
    rows.append((metric_group, metric_name, scalar_count(spark, full_table_name)))


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

    silver_dim_documents = table_name(catalog, schema, "silver_dim_documents")
    silver_dim_dates = table_name(catalog, schema, "silver_dim_dates")
    silver_dim_document_text = table_name(catalog, schema, "silver_dim_document_text")
    silver_fact_documents = table_name(catalog, schema, "silver_fact_documents")
    silver_dim_creators = table_name(catalog, schema, "silver_dim_creators")
    silver_bridge_document_creators = table_name(
        catalog,
        schema,
        "silver_bridge_document_creators",
    )
    silver_dim_subjects = table_name(catalog, schema, "silver_dim_subjects")
    silver_bridge_document_subjects = table_name(
        catalog,
        schema,
        "silver_bridge_document_subjects",
    )
    silver_dim_figures = table_name(catalog, schema, "silver_dim_figures")
    silver_dim_figure_variants = table_name(
        catalog,
        schema,
        "silver_dim_figure_variants",
    )
    silver_publication_date_candidates = table_name(
        catalog,
        schema,
        "silver_publication_date_candidates",
    )
    silver_fact_figure_mentions = table_name(
        catalog,
        schema,
        "silver_fact_figure_mentions",
    )
    silver_dim_mention_contexts = table_name(
        catalog,
        schema,
        "silver_dim_mention_contexts",
    )
    silver_dim_stance_categories = table_name(
        catalog,
        schema,
        "silver_dim_stance_categories",
    )
    silver_stance_model_audit = table_name(catalog, schema, "silver_stance_model_audit")

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

    append_table_count(
        rows,
        spark,
        "documents",
        "silver_dim_documents_rows",
        silver_dim_documents,
    )
    append_table_count(
        rows,
        spark,
        "documents",
        "silver_fact_documents_rows",
        silver_fact_documents,
    )
    append_filtered_count(
        rows,
        spark,
        "documents",
        "documents_with_ocr_text",
        silver_fact_documents,
        "has_ocr_text = true",
    )
    append_filtered_count(
        rows,
        spark,
        "documents",
        "documents_without_ocr_text",
        silver_fact_documents,
        "has_ocr_text = false",
    )
    append_filtered_count(
        rows,
        spark,
        "documents",
        "documents_with_valid_publication_year",
        silver_fact_documents,
        "has_valid_publication_year = true",
    )
    append_filtered_count(
        rows,
        spark,
        "documents",
        "documents_with_publication_month",
        silver_fact_documents,
        "has_publication_month = true",
    )
    append_filtered_count(
        rows,
        spark,
        "documents",
        "documents_with_publication_day",
        silver_fact_documents,
        "has_publication_day = true",
    )
    append_filtered_count(
        rows,
        spark,
        "documents",
        "analysis_ready_documents",
        silver_fact_documents,
        "included_in_analysis_flag = true",
    )
    append_filtered_count(
        rows,
        spark,
        "documents",
        "metadata_parse_success_documents",
        silver_fact_documents,
        "metadata_parse_success_flag = true",
    )
    append_filtered_count(
        rows,
        spark,
        "documents",
        "documents_with_creator",
        silver_dim_documents,
        "creator IS NOT NULL",
    )
    append_filtered_count(
        rows,
        spark,
        "documents",
        "documents_with_author",
        silver_dim_documents,
        "author IS NOT NULL",
    )
    append_filtered_count(
        rows,
        spark,
        "documents",
        "documents_with_metadata_year_conflict",
        silver_dim_documents,
        "conflicts_with_metadata_year = true",
    )

    append_table_count(
        rows,
        spark,
        "dates",
        "silver_dim_dates_rows",
        silver_dim_dates,
    )
    append_table_count(
        rows,
        spark,
        "dates",
        "silver_publication_date_candidate_rows",
        silver_publication_date_candidates,
    )
    append_filtered_count(
        rows,
        spark,
        "dates",
        "rule_based_publication_date_candidates",
        silver_publication_date_candidates,
        "extractor_name = 'rule_based_metadata_title_ocr'",
    )
    append_filtered_count(
        rows,
        spark,
        "dates",
        "ai_publication_date_candidates",
        silver_publication_date_candidates,
        "extractor_name = 'databricks_ai_extract'",
    )
    append_filtered_count(
        rows,
        spark,
        "dates",
        "selected_publication_date_candidates",
        silver_publication_date_candidates,
        "selected_for_document = true",
    )
    append_filtered_count(
        rows,
        spark,
        "dates",
        "selected_ai_publication_date_candidates",
        silver_publication_date_candidates,
        "selected_for_document = true AND extractor_name = 'databricks_ai_extract'",
    )

    append_table_count(
        rows,
        spark,
        "text_quality",
        "silver_dim_document_text_rows",
        silver_dim_document_text,
    )
    append_filtered_count(
        rows,
        spark,
        "text_quality",
        "usable_ocr_documents",
        silver_fact_documents,
        "ocr_quality_flag = 'usable'",
    )
    append_filtered_count(
        rows,
        spark,
        "text_quality",
        "very_short_text_documents",
        silver_fact_documents,
        "ocr_quality_flag = 'very_short_text'",
    )
    append_filtered_count(
        rows,
        spark,
        "text_quality",
        "empty_text_documents",
        silver_fact_documents,
        "ocr_quality_flag = 'empty_text'",
    )
    append_filtered_count(
        rows,
        spark,
        "text_quality",
        "possible_encoding_artifact_documents",
        silver_fact_documents,
        "ocr_quality_flag = 'possible_encoding_artifacts'",
    )
    append_filtered_count(
        rows,
        spark,
        "text_quality",
        "documents_with_encoding_artifacts",
        silver_fact_documents,
        "contains_encoding_artifacts = true",
    )

    append_table_count(rows, spark, "creators", "creator_rows", silver_dim_creators)
    append_table_count(
        rows,
        spark,
        "creators",
        "document_creator_bridge_rows",
        silver_bridge_document_creators,
    )
    append_filtered_count(
        rows,
        spark,
        "creators",
        "unknown_creator_bridge_rows",
        silver_bridge_document_creators,
        "creator_source = 'unknown'",
    )

    append_table_count(rows, spark, "subjects", "subject_rows", silver_dim_subjects)
    append_table_count(
        rows,
        spark,
        "subjects",
        "document_subject_bridge_rows",
        silver_bridge_document_subjects,
    )
    append_filtered_count(
        rows,
        spark,
        "subjects",
        "unknown_subject_bridge_rows",
        silver_bridge_document_subjects,
        "subject_source = 'unknown'",
    )

    append_table_count(rows, spark, "figures", "distinct_figures", silver_dim_figures)
    append_table_count(
        rows,
        spark,
        "figures",
        "figure_variant_rows",
        silver_dim_figure_variants,
    )

    append_table_count(
        rows,
        spark,
        "mentions",
        "silver_fact_figure_mentions_rows",
        silver_fact_figure_mentions,
    )
    append_filtered_count(
        rows,
        spark,
        "mentions",
        "high_confidence_mentions",
        silver_fact_figure_mentions,
        "match_confidence = 'high'",
    )
    append_filtered_count(
        rows,
        spark,
        "mentions",
        "medium_confidence_mentions",
        silver_fact_figure_mentions,
        "match_confidence = 'medium'",
    )
    append_filtered_count(
        rows,
        spark,
        "mentions",
        "low_confidence_mentions",
        silver_fact_figure_mentions,
        "match_confidence = 'low'",
    )
    append_filtered_count(
        rows,
        spark,
        "mentions",
        "analysis_ready_mentions",
        silver_fact_figure_mentions,
        "is_analysis_ready = true",
    )

    append_table_count(
        rows,
        spark,
        "contexts",
        "silver_dim_mention_contexts_rows",
        silver_dim_mention_contexts,
    )
    append_filtered_count(
        rows,
        spark,
        "contexts",
        "empty_mention_contexts",
        silver_dim_mention_contexts,
        "context_window = ''",
    )

    append_table_count(
        rows,
        spark,
        "stance",
        "silver_dim_stance_category_rows",
        silver_dim_stance_categories,
    )
    append_table_count(
        rows,
        spark,
        "stance",
        "silver_stance_model_audit_rows",
        silver_stance_model_audit,
    )
    append_filtered_count(
        rows,
        spark,
        "stance",
        "stance_scored_mentions",
        silver_fact_figure_mentions,
        "is_stance_scored = true",
    )
    append_filtered_count(
        rows,
        spark,
        "stance",
        "medium_high_confidence_stance_mentions",
        silver_fact_figure_mentions,
        "is_medium_or_high_stance_confidence = true",
    )
    append_filtered_count(
        rows,
        spark,
        "stance",
        "direct_or_indirect_relevance_mentions",
        silver_fact_figure_mentions,
        "is_direct_or_indirect_relevance = true",
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
    mentions_table = table_name(catalog, schema, "silver_fact_figure_mentions")

    return (
        spark.table(mentions_table)
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
    mentions_table = table_name(catalog, schema, "silver_fact_figure_mentions")
    stance_categories_table = table_name(catalog, schema, "silver_dim_stance_categories")

    mentions = spark.table(mentions_table).drop("stance_score")

    stance_categories = spark.table(stance_categories_table).select(
        "stance_category_id",
        "stance_label",
        "stance_intensity",
        "stance_score",
        "stance_confidence",
        "target_relevance",
    )

    return (
        mentions.join(stance_categories, on="stance_category_id", how="left")
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
            F.avg(F.abs(F.col("stance_score"))).alias("avg_abs_stance_score"),
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