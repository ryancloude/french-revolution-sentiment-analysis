"""Classify target-specific stance in context windows using Databricks ai_query."""

from __future__ import annotations

import argparse
import re

from pyspark.sql import SparkSession

VALID_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")

AI_QUERY_MODEL = "databricks-meta-llama-3-3-70b-instruct"

STANCE_INSTRUCTIONS = """
You are classifying rhetorical stance in French Revolution pamphlet text.

Task:
Classify the stance expressed toward the target figure only.

Important rules:
- Do not classify the overall sentiment of the passage.
- Do not classify sentiment toward another person, institution, event, or idea.
- If the passage is factual, procedural, OCR-damaged, ambiguous, or does not clearly express
  stance toward the target figure, use neutral_or_unclear.
- If the figure is mentioned but the praise/blame is aimed at someone else, use
  neutral_or_unclear or target_relevance indirect.
- Do not infer beyond the text.
- Use the original French context as the basis for classification.
- Evidence text must be a short exact phrase from the original context.
- Translate only the evidence phrase into English.

Return exactly one JSON object.
Do not use markdown.
Do not wrap the JSON in ```json fences.
Do not include any text before or after the JSON object.
The first character must be { and the last character must be }.

Return only valid JSON with these fields:
{
  "stance_label": "positive | negative | neutral_or_unclear",
  "stance_score": number between -1 and 1,
  "stance_confidence": "high | medium | low",
  "target_relevance": "direct | indirect | not_relevant",
  "evidence_text": "short exact phrase from the French context, or null",
  "evidence_translation_en": "English translation of evidence_text, or null",
  "explanation": "brief English explanation"
}
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


def build_context_stance_ai_query(
    spark: SparkSession,
    catalog: str,
    schema: str,
    model: str,
    limit: int | None,
) -> None:
    """Create AI stance classifications for context windows."""
    context_windows_table = table_name(catalog, schema, "silver_context_windows")
    output_table = table_name(catalog, schema, "silver_context_stance_ai_query")

    instructions = sql_string_literal(STANCE_INSTRUCTIONS)
    model_name = sql_string_literal(model)
    limit_clause = f"LIMIT {limit}" if limit is not None else ""

    spark.sql(
        f"""
        CREATE OR REPLACE TABLE {output_table} AS
        WITH source_contexts AS (
            SELECT
                mention_id,
                document_id,
                figure_id,
                canonical_name,
                matched_variant,
                variant_type,
                match_confidence,
                match_method,
                publication_year,
                publication_month,
                publication_day,
                publication_date,
                date_precision,
                date_calendar,
                date_extractor_name,
                date_source_field,
                date_confidence,
                conflicts_with_metadata_year,
                title,
                ocr_quality_flag,
                context_word_count,
                context_window,
                concat(
                    {instructions},
                    '\\n\\nTarget figure: ', coalesce(canonical_name, ''),
                    '\\nMatched variant: ', coalesce(matched_variant, ''),
                    '\\nPublication year: ', coalesce(cast(publication_year AS string), ''),
                    '\\nPublication month: ', coalesce(cast(publication_month AS string), ''),
                    '\\nDocument title: ', coalesce(title, ''),
                    '\\n\\nContext window:\\n', coalesce(context_window, '')
                ) AS prompt
            FROM {context_windows_table}
            WHERE context_window IS NOT NULL
              AND context_window != ''
            ORDER BY mention_id
            {limit_clause}
        ),

        queried AS (
            SELECT
                *,
                ai_query({model_name}, prompt) AS raw_response
            FROM source_contexts
        ),

        cleaned AS (
            SELECT
                *,
                trim(
                    regexp_replace(
                        regexp_replace(
                            raw_response,
                            '^```(?:json)?\\s*',
                            ''
                        ),
                        '\\s*```$',
                        ''
                    )
                ) AS cleaned_raw_response
            FROM queried
        ),

        parsed AS (
            SELECT
                *,
                try_parse_json(cleaned_raw_response) AS response_json
            FROM cleaned
        ),

        normalized AS (
            SELECT
                *,
                lower(
                    cast(variant_get(response_json, '$.stance_label', 'string') AS string)
                ) AS stance_label_raw,
                cast(variant_get(response_json, '$.stance_score', 'double') AS double)
                    AS stance_score_raw,
                lower(
                    cast(variant_get(response_json, '$.stance_confidence', 'string') AS string)
                ) AS stance_confidence_raw,
                lower(
                    cast(variant_get(response_json, '$.target_relevance', 'string') AS string)
                ) AS target_relevance_raw,
                cast(variant_get(response_json, '$.evidence_text', 'string') AS string)
                    AS evidence_text,
                cast(
                    variant_get(response_json, '$.evidence_translation_en', 'string')
                    AS string
                ) AS evidence_translation_en,
                cast(variant_get(response_json, '$.explanation', 'string') AS string)
                    AS explanation
            FROM parsed
        )

        SELECT
            sha2(
                concat_ws(
                    '|',
                    mention_id,
                    'databricks_ai_query',
                    {model_name}
                ),
                256
            ) AS stance_candidate_id,

            mention_id,
            document_id,
            figure_id,
            canonical_name,
            matched_variant,
            variant_type,
            match_confidence,
            match_method,

            publication_year,
            publication_month,
            publication_day,
            publication_date,
            date_precision,
            date_calendar,
            date_extractor_name,
            date_source_field,
            date_confidence,
            conflicts_with_metadata_year,

            title,
            ocr_quality_flag,
            context_word_count,
            context_window,

            'databricks_ai_query' AS stance_method,
            {model_name} AS stance_model,

            CASE
                WHEN stance_label_raw IN ('positive', 'negative', 'neutral_or_unclear')
                    THEN stance_label_raw
                ELSE 'neutral_or_unclear'
            END AS stance_label,

            CASE
                WHEN stance_score_raw < -1 THEN -1.0
                WHEN stance_score_raw > 1 THEN 1.0
                WHEN stance_score_raw IS NULL THEN 0.0
                ELSE stance_score_raw
            END AS stance_score,

            CASE
                WHEN stance_confidence_raw IN ('high', 'medium', 'low')
                    THEN stance_confidence_raw
                ELSE 'low'
            END AS stance_confidence,

            CASE
                WHEN target_relevance_raw IN ('direct', 'indirect', 'not_relevant')
                    THEN target_relevance_raw
                ELSE 'not_relevant'
            END AS target_relevance,

            evidence_text,
            evidence_translation_en,
            explanation,
            raw_response,
            cleaned_raw_response,
            prompt,
            current_timestamp() AS stance_scored_at

        FROM normalized
        """
    )

    row_count = spark.table(output_table).count()
    print(f"Wrote {row_count} rows to {output_table}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--model", default=AI_QUERY_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Run Databricks AI stance classification."""
    args = parse_args()
    spark = SparkSession.builder.getOrCreate()

    build_context_stance_ai_query(
        spark=spark,
        catalog=args.catalog,
        schema=args.schema,
        model=args.model,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()