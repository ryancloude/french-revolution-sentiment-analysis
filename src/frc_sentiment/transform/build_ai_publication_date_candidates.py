"""Append AI-extracted publication date candidates to the unified silver candidate table."""

from __future__ import annotations

import argparse
import re

from pyspark.sql import SparkSession

VALID_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")

AI_EXTRACT_SCHEMA = """
{
  "publication_year": {
    "type": "integer",
    "description": "Publication year if explicitly supported by the text"
  },
  "publication_month": {
    "type": "integer",
    "description": "Publication month number 1-12 if explicitly supported by the text"
  },
  "publication_day": {
    "type": "integer",
    "description": "Publication day if explicitly supported by the text"
  },
  "publication_date": {
    "type": "string",
    "description": "Gregorian date in YYYY-MM-DD if exact day is known; null otherwise"
  },
  "date_precision": {
    "type": "enum",
    "labels": ["day", "month", "year", "unknown"],
    "description": "Precision of extracted date"
  },
  "calendar": {
    "type": "enum",
    "labels": ["gregorian", "french_republican", "unknown"],
    "description": "Calendar used by the evidence text"
  },
  "evidence": {
    "type": "string",
    "description": "Exact source text supporting the date"
  },
  "confidence": {
    "type": "enum",
    "labels": ["high", "medium", "low"],
    "description": "Confidence in extraction"
  }
}
"""

AI_EXTRACT_INSTRUCTIONS = """
Extract the most likely publication date of this French Revolution pamphlet.
Prefer metadata date, then title, then OCR front matter.
Do not use dates that only describe historical events unless they appear to be the
publication or imprint date.
Do not guess.
If only a year is supported, return only the year and date_precision year.
If month and year are supported, return date_precision month.
If day, month, and year are supported, return date_precision day.
"""


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


def sql_string_literal(value: str) -> str:
    """Escape a Python string as a single-quoted SQL string literal."""
    return "'" + value.replace("'", "''") + "'"


def build_ai_extract_date_candidates(
    spark: SparkSession,
    catalog: str,
    schema: str,
) -> None:
    """Append AI-extracted publication date candidates to the unified candidate table."""
    documents_table = table_name(catalog, schema, "silver_dim_documents")
    candidates_table = table_name(catalog, schema, "silver_publication_date_candidates")

    ai_extract_schema = sql_string_literal(AI_EXTRACT_SCHEMA)
    ai_extract_instructions = sql_string_literal(AI_EXTRACT_INSTRUCTIONS)

    spark.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW tmp_ai_publication_date_candidates AS
        WITH source_documents AS (
            SELECT
                document_id,
                title,
                publication_date_raw,
                ocr_front_matter,
                concat(
                    'Metadata date: ', coalesce(publication_date_raw, ''), '\\n',
                    'Title: ', coalesce(title, ''), '\\n',
                    'OCR front matter: ', coalesce(substr(ocr_front_matter, 1, 2000), '')
                ) AS extraction_input
            FROM {documents_table}
        ),

        extracted AS (
            SELECT
                *,
                ai_extract(
                    extraction_input,
                    {ai_extract_schema},
                    options => map(
                        'version', '2.1',
                        'enableConfidenceScores', 'true',
                        'enableCitations', 'true',
                        'instructions',
                        {ai_extract_instructions}
                    )
                ) AS extracted
            FROM source_documents
        ),

        normalized AS (
            SELECT
                document_id,
                title,
                publication_date_raw,
                ocr_front_matter,
                extraction_input,
                extracted,

                cast(variant_get(extracted, '$.response.publication_year.value', 'string') AS int)
                    AS publication_year,
                cast(variant_get(extracted, '$.response.publication_month.value', 'string') AS int)
                    AS publication_month,
                cast(variant_get(extracted, '$.response.publication_day.value', 'string') AS int)
                    AS publication_day,

                coalesce(
                    to_date(
                        variant_get(extracted, '$.response.publication_date.value', 'string'),
                        'yyyy-MM-dd'
                    ),
                    to_date(
                        variant_get(extracted, '$.response.publication_date.value', 'string'),
                        'd MMMM yyyy'
                    ),
                    to_date(
                        variant_get(extracted, '$.response.publication_date.value', 'string'),
                        'MMMM d yyyy'
                    )
                ) AS publication_date,

                lower(
                    cast(
                        variant_get(extracted, '$.response.date_precision.value', 'string')
                        AS string
                    )
                ) AS date_precision_raw,
                lower(
                    cast(
                        variant_get(extracted, '$.response.calendar.value', 'string')
                        AS string
                    )
                ) AS calendar_raw,
                lower(
                    cast(
                        variant_get(extracted, '$.response.confidence.value', 'string')
                        AS string
                    )
                ) AS confidence_raw,
                cast(
                    variant_get(extracted, '$.response.evidence.value', 'string')
                    AS string
                ) AS evidence,

                greatest(
                    coalesce(
                        cast(
                            variant_get(
                                extracted,
                                '$.response.publication_date.confidence_score',
                                'double'
                            ) AS double
                        ),
                        0.0
                    ),
                    coalesce(
                        cast(
                            variant_get(
                                extracted,
                                '$.response.publication_year.confidence_score',
                                'double'
                            ) AS double
                        ),
                        0.0
                    ),
                    coalesce(
                        cast(
                            variant_get(
                                extracted,
                                '$.response.evidence.confidence_score',
                                'double'
                            ) AS double
                        ),
                        0.0
                    )
                ) AS confidence_score
            FROM extracted
        )

        SELECT
            sha2(
                concat_ws(
                    '|',
                    document_id,
                    'databricks_ai_extract',
                    coalesce(cast(publication_date AS string), ''),
                    coalesce(cast(publication_year AS string), ''),
                    coalesce(cast(publication_month AS string), ''),
                    coalesce(cast(publication_day AS string), ''),
                    coalesce(date_precision_raw, ''),
                    coalesce(calendar_raw, ''),
                    coalesce(evidence, '')
                ),
                256
            ) AS date_candidate_id,

            document_id,

            'databricks_ai_extract' AS extractor_name,
            '2.1' AS extractor_version,
            'combined_metadata_title_ocr_front_matter' AS candidate_source_field,

            publication_year AS candidate_publication_year,
            publication_month AS candidate_publication_month,
            publication_day AS candidate_publication_day,
            publication_date AS candidate_publication_date,

            CASE
                WHEN date_precision_raw IN ('day', 'month', 'year') THEN date_precision_raw
                WHEN publication_date IS NOT NULL THEN 'day'
                WHEN publication_month IS NOT NULL AND publication_year IS NOT NULL THEN 'month'
                WHEN publication_year IS NOT NULL THEN 'year'
                ELSE 'unknown'
            END AS candidate_date_precision,

            CASE
                WHEN calendar_raw LIKE '%republic%' THEN 'french_republican'
                WHEN calendar_raw LIKE '%gregorian%' THEN 'gregorian'
                ELSE 'unknown'
            END AS candidate_date_calendar,

            CASE
                WHEN confidence_raw IN ('high', 'medium', 'low') THEN confidence_raw
                WHEN confidence_score >= 0.85 THEN 'high'
                WHEN confidence_score >= 0.65 THEN 'medium'
                ELSE 'low'
            END AS candidate_confidence,

            confidence_score AS candidate_confidence_score,
            evidence AS candidate_evidence,
            to_json(extracted) AS raw_extraction,
            substr(extraction_input, 1, 2500) AS source_text_excerpt,
            CAST(false AS boolean) AS selected_for_document,
            current_timestamp() AS created_at

        FROM normalized
        WHERE
            publication_year IS NOT NULL
            OR publication_month IS NOT NULL
            OR publication_day IS NOT NULL
            OR publication_date IS NOT NULL
            OR evidence IS NOT NULL
        """
    )

    spark.sql(
        f"""
        CREATE OR REPLACE TABLE {candidates_table} AS
        SELECT
            date_candidate_id,
            document_id,
            extractor_name,
            extractor_version,
            candidate_source_field,
            candidate_publication_year,
            candidate_publication_month,
            candidate_publication_day,
            candidate_publication_date,
            candidate_date_precision,
            candidate_date_calendar,
            candidate_confidence,
            CAST(NULL AS double) AS candidate_confidence_score,
            candidate_notes AS candidate_evidence,
            CAST(NULL AS string) AS raw_extraction,
            substr(
                concat(
                    'Metadata date: ', coalesce(publication_date_raw, ''), '\\n',
                    'Title: ', coalesce(title, ''), '\\n',
                    'OCR front matter: ', coalesce(substr(ocr_front_matter, 1, 2000), '')
                ),
                1,
                2500
            ) AS source_text_excerpt,
            selected_for_document,
            created_at
        FROM {candidates_table}
        WHERE extractor_name != 'databricks_ai_extract'

        UNION ALL

        SELECT
            date_candidate_id,
            document_id,
            extractor_name,
            extractor_version,
            candidate_source_field,
            candidate_publication_year,
            candidate_publication_month,
            candidate_publication_day,
            candidate_publication_date,
            candidate_date_precision,
            candidate_date_calendar,
            candidate_confidence,
            candidate_confidence_score,
            candidate_evidence,
            raw_extraction,
            source_text_excerpt,
            selected_for_document,
            created_at
        FROM tmp_ai_publication_date_candidates
        """
    )

    ai_row_count = spark.table(candidates_table).where(
        "extractor_name = 'databricks_ai_extract'"
    ).count()
    print(f"Wrote {ai_row_count} AI date candidate rows to {candidates_table}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    return parser.parse_args()


def main() -> None:
    """Run AI date candidate extraction."""
    args = parse_args()
    spark = SparkSession.builder.getOrCreate()

    build_ai_extract_date_candidates(
        spark=spark,
        catalog=args.catalog,
        schema=args.schema,
    )


if __name__ == "__main__":
    main()