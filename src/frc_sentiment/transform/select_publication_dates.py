"""Select the best publication date candidate per document."""

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
    """Create a selected publication date table from rule and AI candidates."""
    documents_table = table_name(catalog, schema, "silver_documents")
    rule_candidates_table = table_name(catalog, schema, "silver_date_candidates")
    ai_candidates_table = table_name(catalog, schema, "silver_date_candidates_ai_extract")
    output_table = table_name(catalog, schema, "silver_selected_publication_dates")

    spark.sql(
        f"""
        CREATE OR REPLACE TABLE {output_table} AS
        WITH base_documents AS (
            SELECT
                document_id,
                title,
                publication_date_raw,
                publication_year AS metadata_selected_year,
                publication_month AS metadata_selected_month,
                publication_day AS metadata_selected_day,
                publication_date AS metadata_selected_date,
                date_precision AS metadata_selected_precision,
                date_source AS metadata_selected_source,
                date_calendar AS metadata_selected_calendar
            FROM {documents_table}
        ),

        combined_candidates AS (
            SELECT
                date_candidate_id,
                document_id,
                extractor_name,
                source_field,
                publication_year,
                publication_month,
                publication_day,
                publication_date,
                date_precision,
                date_calendar,
                confidence,
                CAST(NULL AS double) AS confidence_score,
                evidence,
                source_text_excerpt,
                selected_in_silver_documents,
                candidate_created_at
            FROM {rule_candidates_table}

            UNION ALL

            SELECT
                date_candidate_id,
                document_id,
                extractor_name,
                source_field,
                publication_year,
                publication_month,
                publication_day,
                publication_date,
                date_precision,
                date_calendar,
                confidence,
                confidence_score,
                evidence,
                source_text_excerpt,
                CAST(false AS boolean) AS selected_in_silver_documents,
                candidate_created_at
            FROM {ai_candidates_table}
        ),

        scored_candidates AS (
            SELECT
                docs.title,
                docs.publication_date_raw,
                docs.metadata_selected_year,
                docs.metadata_selected_month,
                docs.metadata_selected_day,
                docs.metadata_selected_date,
                docs.metadata_selected_precision,
                docs.metadata_selected_source,
                docs.metadata_selected_calendar,

                candidates.*,

                CASE
                    WHEN candidates.date_precision = 'day' THEN 3
                    WHEN candidates.date_precision = 'month' THEN 2
                    WHEN candidates.date_precision = 'year' THEN 1
                    ELSE 0
                END AS precision_rank,

                CASE
                    WHEN candidates.confidence = 'high' THEN 3
                    WHEN candidates.confidence = 'medium' THEN 2
                    WHEN candidates.confidence = 'low' THEN 1
                    ELSE 0
                END AS confidence_rank,

                CASE
                    WHEN candidates.extractor_name = 'rules' THEN 2
                    WHEN candidates.extractor_name = 'databricks_ai_extract' THEN 1
                    ELSE 0
                END AS extractor_rank,

                CASE
                    WHEN docs.publication_date_raw IS NOT NULL
                         AND docs.metadata_selected_year IS NOT NULL
                         AND candidates.publication_year IS NOT NULL
                         AND docs.metadata_selected_year != candidates.publication_year
                    THEN true
                    ELSE false
                END AS conflicts_with_metadata_year

            FROM combined_candidates candidates
            INNER JOIN base_documents docs
                ON candidates.document_id = docs.document_id
            WHERE candidates.date_precision IN ('day', 'month', 'year')
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
                        coalesce(confidence_score, 0.0) DESC,
                        source_field ASC,
                        date_candidate_id ASC
                ) AS candidate_rank
            FROM scored_candidates
        )

        SELECT
            document_id,
            title,
            publication_date_raw,

            metadata_selected_year,
            metadata_selected_month,
            metadata_selected_day,
            metadata_selected_date,
            metadata_selected_precision,
            metadata_selected_source,
            metadata_selected_calendar,

            date_candidate_id AS selected_date_candidate_id,
            extractor_name AS selected_extractor_name,
            source_field AS selected_source_field,

            publication_year AS selected_publication_year,
            publication_month AS selected_publication_month,
            publication_day AS selected_publication_day,
            publication_date AS selected_publication_date,
            date_precision AS selected_date_precision,
            date_calendar AS selected_date_calendar,
            confidence AS selected_confidence,
            confidence_score AS selected_confidence_score,
            evidence AS selected_evidence,
            source_text_excerpt AS selected_source_text_excerpt,

            conflicts_with_metadata_year,

            CASE
                WHEN conflicts_with_metadata_year
                THEN 'selected candidate conflicts with metadata year; review before use'
                WHEN extractor_name = 'databricks_ai_extract'
                     AND date_precision IN ('day', 'month')
                THEN 'ai candidate selected for improved precision; review evidence'
                WHEN extractor_name = 'rules'
                THEN 'rule candidate selected'
                ELSE 'candidate selected by ranking'
            END AS selection_notes,

            current_timestamp() AS selected_at

        FROM ranked_candidates
        WHERE candidate_rank = 1
        """
    )

    row_count = spark.table(output_table).count()
    print(f"Wrote {row_count} rows to {output_table}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    return parser.parse_args()


def main() -> None:
    """Run publication date candidate selection."""
    args = parse_args()
    spark = SparkSession.builder.getOrCreate()

    select_publication_dates(
        spark=spark,
        catalog=args.catalog,
        schema=args.schema,
    )


if __name__ == "__main__":
    main()