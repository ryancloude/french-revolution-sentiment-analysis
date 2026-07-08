"""Classify target-specific stance and update dimensional silver stance tables."""

from __future__ import annotations

import argparse
import re

from pyspark.sql import DataFrame, SparkSession

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

Stance label rules:
- positive means the passage praises, defends, legitimizes, admires, or supports the target.
- negative means the passage criticizes, condemns, mocks, attacks, blames, or delegitimizes
  the target.
- neutral_or_unclear means the stance is factual, ambiguous, unclear, not about the target,
  or too damaged by OCR to classify.

Stance intensity rules:
- none: no clear positive or negative stance.
- weak: mild praise or criticism.
- moderate: clear praise or criticism, but not extreme.
- strong: intense praise, intense condemnation, severe accusation, or highly charged rhetoric.
- If stance_label is neutral_or_unclear, stance_intensity must be none.
- If stance_label is positive or negative, stance_intensity must be weak, moderate, or strong.

Target relevance rules:
- direct: the stance clearly targets the named figure.
- indirect: the figure is relevant, but the stance is partly about a related group, event,
  institution, or action.
- not_relevant: the figure is only mentioned incidentally, or the stance is not about the figure.

Return exactly one JSON object.
Do not use markdown.
Do not wrap the JSON in ```json fences.
Do not include any text before or after the JSON object.
The first character must be { and the last character must be }.

Return only valid JSON with these fields:
{
  "stance_label": "positive | negative | neutral_or_unclear",
  "stance_intensity": "none | weak | moderate | strong",
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
) -> DataFrame:
    """Create normalized AI stance classifications for mention contexts."""
    mentions_table = table_name(catalog, schema, "silver_fact_figure_mentions")
    contexts_table = table_name(catalog, schema, "silver_dim_mention_contexts")
    documents_table = table_name(catalog, schema, "silver_dim_documents")
    temp_view = "tmp_normalized_context_stance_ai_query"

    instructions = sql_string_literal(STANCE_INSTRUCTIONS)
    model_name = sql_string_literal(model)
    limit_clause = f"LIMIT {limit}" if limit is not None else ""

    normalized = spark.sql(
        f"""
        WITH source_contexts AS (
            SELECT
                m.mention_id,
                m.document_id,
                m.figure_id,
                m.variant_id,
                m.date_key,
                m.canonical_name,
                m.matched_variant,
                m.variant_normalized,
                m.variant_type,
                m.match_confidence,
                m.match_method,
                m.match_start_char,
                m.match_end_char,
                m.publication_year,
                m.publication_month,
                m.publication_day,
                m.publication_date,
                m.date_precision,
                m.date_calendar,
                m.ocr_quality_flag,
                m.included_in_analysis_flag,
                m.is_analysis_ready AS mention_is_analysis_ready,
                d.title,
                c.context_word_count,
                c.context_window,
                c.context_start_char,
                c.context_end_char,
                concat(
                    {instructions},
                    '\\n\\nTarget figure: ', coalesce(m.canonical_name, ''),
                    '\\nMatched variant: ', coalesce(m.matched_variant, ''),
                    '\\nPublication year: ', coalesce(cast(m.publication_year AS string), ''),
                    '\\nPublication month: ', coalesce(cast(m.publication_month AS string), ''),
                    '\\nDocument title: ', coalesce(d.title, ''),
                    '\\n\\nContext window:\\n', coalesce(c.context_window, '')
                ) AS prompt
            FROM {mentions_table} m
            JOIN {contexts_table} c
              ON m.mention_id = c.mention_id
            LEFT JOIN {documents_table} d
              ON m.document_id = d.document_id
            WHERE c.context_window IS NOT NULL
              AND c.context_window != ''
            ORDER BY m.mention_id
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
                lower(
                    cast(variant_get(response_json, '$.stance_intensity', 'string') AS string)
                ) AS stance_intensity_raw,
                cast(variant_get(response_json, '$.stance_score', 'double') AS double)
                    AS model_stance_score_raw,
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
        ),

        validated AS (
            SELECT
                *,

                CASE
                    WHEN stance_label_raw IN ('positive', 'negative', 'neutral_or_unclear')
                        THEN stance_label_raw
                    ELSE 'neutral_or_unclear'
                END AS stance_label,

                CASE
                    WHEN stance_label_raw = 'neutral_or_unclear'
                        THEN 'none'
                    WHEN stance_label_raw IN ('positive', 'negative')
                         AND stance_intensity_raw IN ('weak', 'moderate', 'strong')
                        THEN stance_intensity_raw
                    WHEN stance_label_raw IN ('positive', 'negative')
                        THEN 'weak'
                    ELSE 'none'
                END AS stance_intensity,

                CASE
                    WHEN stance_confidence_raw IN ('high', 'medium', 'low')
                        THEN stance_confidence_raw
                    ELSE 'low'
                END AS stance_confidence,

                CASE
                    WHEN target_relevance_raw IN ('direct', 'indirect', 'not_relevant')
                        THEN target_relevance_raw
                    ELSE 'not_relevant'
                END AS target_relevance

            FROM normalized
        ),

        scored AS (
            SELECT
                *,
                CASE
                    WHEN stance_label = 'negative' AND stance_intensity = 'strong' THEN -1.0
                    WHEN stance_label = 'negative' AND stance_intensity = 'moderate' THEN -0.66
                    WHEN stance_label = 'negative' AND stance_intensity = 'weak' THEN -0.33
                    WHEN stance_label = 'positive' AND stance_intensity = 'weak' THEN 0.33
                    WHEN stance_label = 'positive' AND stance_intensity = 'moderate' THEN 0.66
                    WHEN stance_label = 'positive' AND stance_intensity = 'strong' THEN 1.0
                    ELSE 0.0
                END AS stance_score,
                'deterministic_label_intensity_mapping' AS stance_score_method
            FROM validated
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
            ) AS stance_audit_id,

            sha2(
                concat_ws(
                    '|',
                    stance_label,
                    stance_intensity,
                    stance_confidence,
                    target_relevance,
                    stance_score_method
                ),
                256
            ) AS stance_category_id,

            mention_id,
            document_id,
            figure_id,
            variant_id,
            date_key,
            canonical_name,
            matched_variant,
            variant_normalized,
            variant_type,
            match_confidence,
            match_method,
            match_start_char,
            match_end_char,

            publication_year,
            publication_month,
            publication_day,
            publication_date,
            date_precision,
            date_calendar,

            title,
            ocr_quality_flag,
            included_in_analysis_flag,
            mention_is_analysis_ready,
            context_word_count,
            context_window,
            context_start_char,
            context_end_char,

            'databricks_ai_query' AS stance_method,
            {model_name} AS stance_model,
            stance_label,
            stance_intensity,
            stance_confidence,
            target_relevance,
            stance_score,
            stance_score_method,
            model_stance_score_raw,

            evidence_text,
            evidence_translation_en,
            explanation,
            raw_response,
            cleaned_raw_response,
            prompt,
            current_timestamp() AS stance_scored_at
        FROM scored
        """
    )

    normalized.createOrReplaceTempView(temp_view)
    return normalized


def write_dim_stance_categories(
    spark: SparkSession,
    catalog: str,
    schema: str,
) -> None:
    """Write stance category dimension from normalized stance results."""
    temp_view = "tmp_normalized_context_stance_ai_query"
    output_table = table_name(catalog, schema, "silver_dim_stance_categories")

    spark.sql(
        f"""
        CREATE OR REPLACE TABLE {output_table} AS
        SELECT DISTINCT
            stance_category_id,
            stance_label,
            stance_intensity,
            stance_confidence,
            target_relevance,
            stance_score_method,
            stance_score,
            current_timestamp() AS created_at
        FROM {temp_view}
        """
    )

    print(f"Wrote {spark.table(output_table).count()} rows to {output_table}")


def write_stance_model_audit(
    spark: SparkSession,
    catalog: str,
    schema: str,
) -> None:
    """Write stance model audit table."""
    temp_view = "tmp_normalized_context_stance_ai_query"
    output_table = table_name(catalog, schema, "silver_stance_model_audit")

    spark.sql(
        f"""
        CREATE OR REPLACE TABLE {output_table} AS
        SELECT
            stance_audit_id,
            mention_id,
            document_id,
            figure_id,
            variant_id,
            stance_category_id,
            stance_method,
            stance_model,
            stance_score_method,
            model_stance_score_raw,
            evidence_text,
            evidence_translation_en,
            explanation,
            raw_response,
            cleaned_raw_response,
            prompt,
            stance_scored_at
        FROM {temp_view}
        """
    )

    print(f"Wrote {spark.table(output_table).count()} rows to {output_table}")


def update_fact_figure_mentions(
    spark: SparkSession,
    catalog: str,
    schema: str,
) -> None:
    """Overwrite figure mention fact table with selected stance fields added."""
    temp_view = "tmp_normalized_context_stance_ai_query"
    mentions_table = table_name(catalog, schema, "silver_fact_figure_mentions")

    spark.sql(
        f"""
        CREATE OR REPLACE TABLE {mentions_table} AS
        SELECT
            m.mention_id,
            m.document_id,
            m.figure_id,
            m.variant_id,
            m.date_key,
            m.canonical_name,
            m.matched_variant,
            m.variant_normalized,
            m.variant_type,
            m.match_confidence,
            m.is_high_confidence_match,
            m.match_method,
            m.match_start_char,
            m.match_end_char,
            m.publication_year,
            m.publication_month,
            m.publication_day,
            m.publication_date,
            m.date_precision,
            m.date_calendar,
            m.ocr_quality_flag,
            m.included_in_analysis_flag,

            s.stance_category_id,
            s.stance_audit_id,
            s.stance_score,

            CASE
                WHEN s.mention_id IS NOT NULL THEN true
                ELSE false
            END AS is_stance_scored,

            CASE
                WHEN s.stance_confidence IN ('medium', 'high') THEN true
                ELSE false
            END AS is_medium_or_high_stance_confidence,

            CASE
                WHEN s.target_relevance IN ('direct', 'indirect') THEN true
                ELSE false
            END AS is_direct_or_indirect_relevance,

            CASE
                WHEN m.is_analysis_ready
                     AND s.mention_id IS NOT NULL
                     AND s.stance_confidence IN ('medium', 'high')
                THEN true
                ELSE false
            END AS is_analysis_ready,

            m.extracted_at,
            current_timestamp() AS stance_updated_at

        FROM {mentions_table} m
        LEFT JOIN {temp_view} s
          ON m.mention_id = s.mention_id
        """
    )

    print(f"Updated {spark.table(mentions_table).count()} rows in {mentions_table}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--model", default=AI_QUERY_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Run Databricks AI stance classification into dimensional silver model."""
    args = parse_args()
    spark = SparkSession.builder.getOrCreate()

    build_context_stance_ai_query(
        spark=spark,
        catalog=args.catalog,
        schema=args.schema,
        model=args.model,
        limit=args.limit,
    )

    write_dim_stance_categories(
        spark=spark,
        catalog=args.catalog,
        schema=args.schema,
    )

    write_stance_model_audit(
        spark=spark,
        catalog=args.catalog,
        schema=args.schema,
    )

    update_fact_figure_mentions(
        spark=spark,
        catalog=args.catalog,
        schema=args.schema,
    )


if __name__ == "__main__":
    main()