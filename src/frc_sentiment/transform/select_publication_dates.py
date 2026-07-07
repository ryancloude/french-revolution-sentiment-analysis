"""Select publication dates and update dimensional document/date tables."""

from __future__ import annotations

import argparse
import re

from pyspark.sql import SparkSession

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


def select_publication_dates(
    spark: SparkSession,
    catalog: str,
    schema: str,
) -> None:
    """Select best candidate dates and update document/date dimensional outputs."""
    documents_table = table_name(catalog, schema, "silver_dim_documents")
    dates_table = table_name(catalog, schema, "silver_dim_dates")
    fact_documents_table = table_name(catalog, schema, "silver_fact_documents")
    candidates_table = table_name(catalog, schema, "silver_publication_date_candidates")
    mentions_table = table_name(catalog, schema, "silver_fact_figure_mentions")

    spark.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW tmp_selected_publication_dates AS
        WITH base_documents AS (
            SELECT
                document_id,
                publication_date_raw,
                date_key AS existing_date_key,
                cast(
                    regexp_extract(
                        publication_date_raw,
                        '(1[5-9][0-9][0-9]|20[0-9][0-9])',
                        1
                    ) AS int
                ) AS metadata_year_from_raw
            FROM {documents_table}
        ),

        scored_candidates AS (
            SELECT
                docs.publication_date_raw,
                docs.existing_date_key,
                candidates.*,

                CASE
                    WHEN candidates.candidate_date_precision = 'day' THEN 3
                    WHEN candidates.candidate_date_precision = 'month' THEN 2
                    WHEN candidates.candidate_date_precision = 'year' THEN 1
                    ELSE 0
                END AS precision_rank,

                CASE
                    WHEN candidates.candidate_confidence = 'high' THEN 3
                    WHEN candidates.candidate_confidence = 'medium' THEN 2
                    WHEN candidates.candidate_confidence = 'low' THEN 1
                    ELSE 0
                END AS confidence_rank,

                CASE
                    WHEN candidates.extractor_name = 'rule_based_metadata_title_ocr' THEN 2
                    WHEN candidates.extractor_name = 'databricks_ai_extract' THEN 1
                    ELSE 0
                END AS extractor_rank,

                CASE
                    WHEN docs.metadata_year_from_raw IS NOT NULL
                         AND candidates.candidate_publication_year IS NOT NULL
                         AND docs.metadata_year_from_raw
                             != candidates.candidate_publication_year
                    THEN true
                    ELSE false
                END AS conflicts_with_metadata_year

            FROM {candidates_table} candidates
            INNER JOIN base_documents docs
              ON candidates.document_id = docs.document_id
            WHERE candidates.candidate_date_precision IN ('day', 'month', 'year')
        ),

        ranked_candidates AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY document_id
                    ORDER BY
                        CASE WHEN conflicts_with_metadata_year THEN 0 ELSE 1 END DESC,
                        precision_rank DESC,
                        confidence_rank DESC,
                        extractor_rank DESC,
                        coalesce(candidate_confidence_score, 0.0) DESC,
                        candidate_source_field ASC,
                        date_candidate_id ASC
                ) AS candidate_rank
            FROM scored_candidates
        )

        SELECT
            document_id,
            date_candidate_id AS selected_date_candidate_id,
            extractor_name AS selected_date_extractor_name,
            candidate_source_field AS selected_date_source_field,

            candidate_publication_year AS publication_year,
            candidate_publication_month AS publication_month,
            candidate_publication_day AS publication_day,
            candidate_publication_date AS publication_date,
            candidate_date_precision AS date_precision,
            candidate_date_calendar AS date_calendar,
            candidate_confidence AS selected_date_confidence,
            candidate_confidence_score AS selected_date_confidence_score,
            candidate_evidence AS selected_date_evidence,
            source_text_excerpt AS selected_date_source_text_excerpt,
            conflicts_with_metadata_year,

            sha2(
                concat_ws(
                    '|',
                    coalesce(cast(candidate_publication_year AS string), 'unknown'),
                    coalesce(cast(candidate_publication_month AS string), 'unknown'),
                    coalesce(cast(candidate_publication_day AS string), 'unknown'),
                    coalesce(candidate_date_precision, 'unknown'),
                    coalesce(candidate_date_calendar, 'unknown')
                ),
                256
            ) AS date_key,

            CASE
                WHEN conflicts_with_metadata_year
                THEN 'selected candidate conflicts with metadata year; review before use'
                WHEN extractor_name = 'databricks_ai_extract'
                     AND candidate_date_precision IN ('day', 'month')
                THEN 'ai candidate selected for improved precision; review evidence'
                WHEN extractor_name = 'rule_based_metadata_title_ocr'
                THEN 'rule candidate selected'
                ELSE 'candidate selected by ranking'
            END AS selected_date_notes,

            current_timestamp() AS selected_at

        FROM ranked_candidates
        WHERE candidate_rank = 1
        """
    )

    spark.sql(
        f"""
        CREATE OR REPLACE TABLE {candidates_table} AS
        SELECT
            c.date_candidate_id,
            c.document_id,
            c.extractor_name,
            c.extractor_version,
            c.candidate_source_field,
            c.candidate_publication_year,
            c.candidate_publication_month,
            c.candidate_publication_day,
            c.candidate_publication_date,
            c.candidate_date_precision,
            c.candidate_date_calendar,
            c.candidate_confidence,
            c.candidate_confidence_score,
            c.candidate_evidence,
            c.raw_extraction,
            c.source_text_excerpt,
            CASE
                WHEN s.selected_date_candidate_id = c.date_candidate_id THEN true
                ELSE false
            END AS selected_for_document,
            c.created_at
        FROM {candidates_table} c
        LEFT JOIN tmp_selected_publication_dates s
          ON c.document_id = s.document_id
        """
    )

    spark.sql(
        f"""
        CREATE OR REPLACE TABLE {documents_table} AS
        SELECT
            d.document_id,
            d.metadata_document_id,
            d.document_id_matches_metadata,
            coalesce(s.date_key, d.date_key) AS date_key,
            d.title,
            d.creator,
            d.author,
            d.publication_date_raw,

            s.selected_date_candidate_id,
            coalesce(s.selected_date_extractor_name, d.date_source) AS date_source,
            coalesce(s.selected_date_confidence, d.date_confidence) AS date_confidence,
            coalesce(s.selected_date_notes, d.date_parse_notes) AS date_parse_notes,
            s.selected_date_source_field,
            s.selected_date_evidence,
            s.selected_date_source_text_excerpt,
            coalesce(s.conflicts_with_metadata_year, false) AS conflicts_with_metadata_year,

            d.language,
            d.internet_archive_id,
            d.source_url,
            d.has_ocr_text,
            d.ocr_front_matter,
            d.metadata_parse_status,
            d.metadata_parse_error,
            d.metadata_file_path,
            d.parsed_at,
            current_timestamp() AS updated_at

        FROM {documents_table} d
        LEFT JOIN tmp_selected_publication_dates s
          ON d.document_id = s.document_id
        """
    )

    spark.sql(
        f"""
        CREATE OR REPLACE TABLE {dates_table} AS
        SELECT DISTINCT
            date_key,
            publication_year,
            publication_month,
            publication_day,
            publication_date,
            date_precision,
            date_calendar,

            CASE
                WHEN publication_year IS NULL THEN 'Unknown'
                WHEN publication_month IS NULL THEN cast(publication_year AS string)
                ELSE concat_ws(
                    '-',
                    cast(publication_year AS string),
                    lpad(cast(publication_month AS string), 2, '0')
                )
            END AS period_label,

            publication_year IS NOT NULL AS has_publication_year,
            publication_month IS NOT NULL AS has_publication_month,
            publication_day IS NOT NULL AS has_publication_day

        FROM tmp_selected_publication_dates

        UNION

        SELECT DISTINCT
            d.date_key,
            old_dates.publication_year,
            old_dates.publication_month,
            old_dates.publication_day,
            old_dates.publication_date,
            old_dates.date_precision,
            old_dates.date_calendar,
            old_dates.period_label,
            old_dates.has_publication_year,
            old_dates.has_publication_month,
            old_dates.has_publication_day
        FROM {documents_table} d
        INNER JOIN {dates_table} old_dates
          ON d.date_key = old_dates.date_key
        """
    )

    spark.sql(
        f"""
        CREATE OR REPLACE TABLE {fact_documents_table} AS
        SELECT
            f.document_id,
            coalesce(s.date_key, f.date_key) AS date_key,
            f.has_ocr_text,
            s.publication_year IS NOT NULL AS has_valid_publication_year,
            s.publication_month IS NOT NULL AS has_publication_month,
            s.publication_day IS NOT NULL AS has_publication_day,
            f.metadata_parse_success_flag,
            f.word_count,
            f.character_count,
            f.clean_character_count,
            f.contains_encoding_artifacts,
            f.ocr_quality_flag,
            (
                s.publication_year IS NOT NULL
                AND f.has_ocr_text
                AND f.ocr_quality_flag != 'empty_text'
                AND coalesce(s.conflicts_with_metadata_year, false) = false
            ) AS included_in_analysis_flag,
            f.created_at,
            current_timestamp() AS updated_at
        FROM {fact_documents_table} f
        LEFT JOIN tmp_selected_publication_dates s
          ON f.document_id = s.document_id
        """
    )

    spark.sql(
        f"""
        CREATE OR REPLACE TABLE {mentions_table} AS
        SELECT
            m.mention_id,
            m.document_id,
            m.figure_id,
            m.variant_id,
            coalesce(s.date_key, m.date_key) AS date_key,
            m.canonical_name,
            m.matched_variant,
            m.variant_normalized,
            m.variant_type,
            m.match_confidence,
            m.is_high_confidence_match,
            m.match_method,
            m.match_start_char,
            m.match_end_char,
            coalesce(s.publication_year, m.publication_year) AS publication_year,
            coalesce(s.publication_month, m.publication_month) AS publication_month,
            coalesce(s.publication_day, m.publication_day) AS publication_day,
            coalesce(s.publication_date, m.publication_date) AS publication_date,
            coalesce(s.date_precision, m.date_precision) AS date_precision,
            coalesce(s.date_calendar, m.date_calendar) AS date_calendar,
            m.ocr_quality_flag,
            m.included_in_analysis_flag,
            m.stance_category_id,
            m.stance_audit_id,
            m.stance_score,
            m.is_stance_scored,
            m.is_medium_or_high_stance_confidence,
            m.is_direct_or_indirect_relevance,
            (
                coalesce(s.publication_year, m.publication_year) IS NOT NULL
                AND m.included_in_analysis_flag
                AND coalesce(s.conflicts_with_metadata_year, false) = false
                AND coalesce(m.is_medium_or_high_stance_confidence, false)
            ) AS is_analysis_ready,
            m.extracted_at,
            m.stance_updated_at
        FROM {mentions_table} m
        LEFT JOIN tmp_selected_publication_dates s
          ON m.document_id = s.document_id
        """
    )

    print("Updated publication date selections in dimensional silver tables")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    return parser.parse_args()


def main() -> None:
    """Run publication date selection for the dimensional model."""
    args = parse_args()
    spark = SparkSession.builder.getOrCreate()

    select_publication_dates(
        spark=spark,
        catalog=args.catalog,
        schema=args.schema,
    )


if __name__ == "__main__":
    main()