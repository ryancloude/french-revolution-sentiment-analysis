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


def ensure_stance_output_tables(
    spark: SparkSession,
    catalog: str,
    schema: str,
) -> None:
    """Create empty stance output tables if they do not already exist."""
    stance_categories_table = table_name(catalog, schema, "silver_dim_stance_categories")
    stance_audit_table = table_name(catalog, schema, "silver_stance_model_audit")

    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {stance_categories_table} (
            stance_category_id STRING,
            stance_label STRING,
            stance_intensity STRING,
            stance_confidence STRING,
            target_relevance STRING,
            stance_score_method STRING,
            stance_score DOUBLE,
            created_at TIMESTAMP
        )
        USING DELTA
        """
    )

    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {stance_audit_table} (
            stance_audit_id STRING,
            mention_id STRING,
            document_id STRING,
            figure_id STRING,
            variant_id STRING,
            stance_category_id STRING,
            stance_method STRING,
            stance_model STRING,
            stance_score_method STRING,
            model_stance_score_raw DOUBLE,
            evidence_text STRING,
            evidence_translation_en STRING,
            explanation STRING,
            raw_response STRING,
            cleaned_raw_response STRING,
            prompt STRING,
            stance_scored_at TIMESTAMP
        )
        USING DELTA
        """
    )


def ensure_fact_stance_columns(
    spark: SparkSession,
    catalog: str,
    schema: str,
) -> None:
    """Add stance columns to the mention fact table if missing."""
    mentions_table = table_name(catalog, schema, "silver_fact_figure_mentions")
    existing_columns = {field.name for field in spark.table(mentions_table).schema.fields}

    columns_to_add = {
        "stance_category_id": "STRING",
        "stance_audit_id": "STRING",
        "stance_score": "DOUBLE",
        "is_stance_scored": "BOOLEAN",
        "is_medium_or_high_stance_confidence": "BOOLEAN",
        "is_direct_or_indirect_relevance": "BOOLEAN",
        "stance_updated_at": "TIMESTAMP",
    }

    for column_name, column_type in columns_to_add.items():
        if column_name not in existing_columns:
            spark.sql(
                f"""
                ALTER TABLE {mentions_table}
                ADD COLUMNS ({quote_identifier(column_name)} {column_type})
                """
            )


def build_context_stance_ai_query(
    spark: SparkSession,
    catalog: str,
    schema: str,
    model: str,
    limit: int | None,
    figure_id: str | None,
    publication_year: int | None,
    publication_month: int | None,
    match_confidence: str,
    include_scored: bool,
) -> DataFrame:
    """Create normalized AI stance classifications for a resumable mention batch."""
    mentions_table = table_name(catalog, schema, "silver_fact_figure_mentions")
    contexts_table = table_name(catalog, schema, "silver_dim_mention_contexts")
    documents_table = table_name(catalog, schema, "silver_dim_documents")
    stance_audit_table = table_name(catalog, schema, "silver_stance_model_audit")
    temp_view = "tmp_normalized_context_stance_ai_query"

    instructions = sql_string_literal(STANCE_INSTRUCTIONS)
    model_name = sql_string_literal(model)
    limit_clause = f"LIMIT {limit}" if limit is not None else ""

    figure_filter = (
        f"AND m.figure_id = {sql_string_literal(figure_id)}" if figure_id else ""
    )
    year_filter = (
        f"AND m.publication_year = {publication_year}"
        if publication_year is not None
        else ""
    )
    month_filter = (
        f"AND m.publication_month = {publication_month}"
        if publication_month is not None
        else ""
    )
    confidence_filter = (
        f"AND m.match_confidence = {sql_string_literal(match_confidence)}"
        if match_confidence != "any"
        else ""
    )
    scored_filter = "" if include_scored else "AND existing.mention_id IS NULL"

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
            LEFT JOIN {stance_audit_table} existing
              ON m.mention_id = existing.mention_id
             AND existing.stance_model = {model_name}
            WHERE c.context_window IS NOT NULL
              AND c.context_window != ''
              AND m.is_analysis_ready = true
              {scored_filter}
              {figure_filter}
              {year_filter}
              {month_filter}
              {confidence_filter}
            ORDER BY m.publication_year, m.publication_month, m.figure_id, m.mention_id
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


def merge_dim_stance_categories(
    spark: SparkSession,
    catalog: str,
    schema: str,
) -> None:
    """Merge stance category rows from the current batch."""
    temp_view = "tmp_normalized_context_stance_ai_query"
    output_table = table_name(catalog, schema, "silver_dim_stance_categories")

    spark.sql(
        f"""
        MERGE INTO {output_table} AS target
        USING (
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
        ) AS source
        ON target.stance_category_id = source.stance_category_id
        WHEN NOT MATCHED THEN INSERT *
        """
    )

    print(f"Stance categories available: {spark.table(output_table).count()}")


def merge_stance_model_audit(
    spark: SparkSession,
    catalog: str,
    schema: str,
) -> None:
    """Merge current batch stance model audit rows."""
    temp_view = "tmp_normalized_context_stance_ai_query"
    output_table = table_name(catalog, schema, "silver_stance_model_audit")

    spark.sql(
        f"""
        MERGE INTO {output_table} AS target
        USING (
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
        ) AS source
        ON target.stance_audit_id = source.stance_audit_id
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
        """
    )

    print(f"Stance audit rows available: {spark.table(output_table).count()}")


def update_fact_figure_mentions_from_audit(
    spark: SparkSession,
    catalog: str,
    schema: str,
) -> None:
    """Update mention fact stance fields from the latest audit result per mention."""
    mentions_table = table_name(catalog, schema, "silver_fact_figure_mentions")
    stance_audit_table = table_name(catalog, schema, "silver_stance_model_audit")
    stance_categories_table = table_name(catalog, schema, "silver_dim_stance_categories")

    spark.sql(
        f"""
        MERGE INTO {mentions_table} AS target
        USING (
            WITH ranked_audit AS (
                SELECT
                    audit.*,
                    row_number() OVER (
                        PARTITION BY audit.mention_id
                        ORDER BY audit.stance_scored_at DESC, audit.stance_audit_id DESC
                    ) AS audit_rank
                FROM {stance_audit_table} audit
            )

            SELECT
                ranked_audit.mention_id,
                ranked_audit.stance_audit_id,
                ranked_audit.stance_category_id,
                categories.stance_score,
                categories.stance_confidence,
                categories.target_relevance,
                ranked_audit.stance_scored_at
            FROM ranked_audit
            LEFT JOIN {stance_categories_table} categories
              ON ranked_audit.stance_category_id = categories.stance_category_id
            WHERE ranked_audit.audit_rank = 1
        ) AS source
        ON target.mention_id = source.mention_id
        WHEN MATCHED THEN UPDATE SET
            target.stance_category_id = source.stance_category_id,
            target.stance_audit_id = source.stance_audit_id,
            target.stance_score = source.stance_score,
            target.is_stance_scored = true,
            target.is_medium_or_high_stance_confidence =
                source.stance_confidence IN ('medium', 'high'),
            target.is_direct_or_indirect_relevance =
                source.target_relevance IN ('direct', 'indirect'),
            target.stance_updated_at = source.stance_scored_at
        """
    )

    print(f"Updated stance fields in {mentions_table}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--model", default=AI_QUERY_MODEL)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--figure-id", default=None)
    parser.add_argument("--publication-year", type=int, default=None)
    parser.add_argument("--publication-month", type=int, default=None)
    parser.add_argument(
        "--match-confidence",
        choices=["high", "medium", "low", "any"],
        default="high",
    )
    parser.add_argument(
        "--include-scored",
        action="store_true",
        help="Rescore mentions even if they already have an audit row for the model.",
    )
    return parser.parse_args()


def main() -> None:
    """Run Databricks AI stance classification into dimensional silver model."""
    args = parse_args()
    spark = SparkSession.builder.getOrCreate()

    ensure_stance_output_tables(
        spark=spark,
        catalog=args.catalog,
        schema=args.schema,
    )

    ensure_fact_stance_columns(
        spark=spark,
        catalog=args.catalog,
        schema=args.schema,
    )

    batch = build_context_stance_ai_query(
        spark=spark,
        catalog=args.catalog,
        schema=args.schema,
        model=args.model,
        limit=args.limit,
        figure_id=args.figure_id,
        publication_year=args.publication_year,
        publication_month=args.publication_month,
        match_confidence=args.match_confidence,
        include_scored=args.include_scored,
    )

    batch_count = batch.count()
    print(f"Scored batch rows: {batch_count}")

    if batch_count == 0:
        update_fact_figure_mentions_from_audit(
            spark=spark,
            catalog=args.catalog,
            schema=args.schema,
        )
        return

    merge_dim_stance_categories(
        spark=spark,
        catalog=args.catalog,
        schema=args.schema,
    )

    merge_stance_model_audit(
        spark=spark,
        catalog=args.catalog,
        schema=args.schema,
    )

    update_fact_figure_mentions_from_audit(
        spark=spark,
        catalog=args.catalog,
        schema=args.schema,
    )


if __name__ == "__main__":
    main()