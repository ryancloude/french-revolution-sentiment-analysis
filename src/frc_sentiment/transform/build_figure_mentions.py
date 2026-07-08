"""Extract figure mentions into the silver figure mention fact table."""

from __future__ import annotations

import argparse
import re

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

VALID_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")

MENTION_SCHEMA = T.ArrayType(
    T.StructType(
        [
            T.StructField("figure_id", T.StringType(), False),
            T.StructField("variant_id", T.StringType(), False),
            T.StructField("matched_variant", T.StringType(), False),
            T.StructField("variant_normalized", T.StringType(), False),
            T.StructField("match_start_char", T.IntegerType(), False),
            T.StructField("match_end_char", T.IntegerType(), False),
            T.StructField("match_method", T.StringType(), False),
        ]
    )
)


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


def normalize_for_regex(value: str) -> str:
    """Normalize lookup text for matching."""
    return " ".join(value.lower().split())


def build_variant_regex(variant_normalized: str) -> re.Pattern[str]:
    """Build a regex pattern for a normalized variant.

    The boundary logic avoids matching names inside longer words.
    For example, it should match 'marat' but not a longer token containing 'marat'.
    """
    normalized = normalize_for_regex(variant_normalized)
    escaped = re.escape(normalized)

    flexible_space_pattern = escaped.replace(r"\ ", r"\s+")
    pattern = rf"(?<!\w){flexible_space_pattern}(?!\w)"

    return re.compile(pattern, flags=re.IGNORECASE)


def extract_mentions_from_text(
    clean_text_lower: str | None,
    figure_variants: list[dict[str, str | None]],
) -> list[dict[str, str | int | None]]:
    """Extract all figure mentions from one cleaned document."""
    if not clean_text_lower:
        return []

    mentions = []

    for variant in figure_variants:
        variant_normalized = variant["variant_normalized"]
        if not variant_normalized:
            continue

        pattern = build_variant_regex(variant_normalized)

        for match in pattern.finditer(clean_text_lower):
            mentions.append(
                {
                    "figure_id": variant["figure_id"],
                    "variant_id": variant["variant_id"],
                    "matched_variant": variant["variant"],
                    "variant_normalized": variant_normalized,
                    "match_start_char": match.start(),
                    "match_end_char": match.end(),
                    "match_method": "dictionary_regex",
                }
            )

    mentions.sort(
        key=lambda mention: (
            int(mention["match_start_char"]),
            int(mention["match_end_char"]),
            str(mention["figure_id"]),
            str(mention["variant_id"]),
        )
    )

    return mentions


def get_figure_variants(
    spark: SparkSession,
    catalog: str,
    schema: str,
) -> list[dict[str, str | None]]:
    """Collect figure variants to the driver for dictionary matching."""
    figure_variants_table = table_name(catalog, schema, "silver_dim_figure_variants")

    rows = (
        spark.table(figure_variants_table)
        .select(
            "variant_id",
            "figure_id",
            "variant",
            "variant_normalized",
        )
        .where(F.col("variant_normalized").isNotNull())
        .collect()
    )

    return [row.asDict(recursive=True) for row in rows]


def build_fact_figure_mentions(
    spark: SparkSession,
    catalog: str,
    schema: str,
) -> DataFrame:
    """Build one row per detected figure mention."""
    document_text_table = table_name(catalog, schema, "silver_dim_document_text")
    documents_table = table_name(catalog, schema, "silver_dim_documents")
    dates_table = table_name(catalog, schema, "silver_dim_dates")
    figures_table = table_name(catalog, schema, "silver_dim_figures")
    variants_table = table_name(catalog, schema, "silver_dim_figure_variants")
    fact_documents_table = table_name(catalog, schema, "silver_fact_documents")

    figure_variants = get_figure_variants(spark, catalog, schema)

    if not figure_variants:
        raise RuntimeError("No figure variants found in silver_dim_figure_variants")

    @F.udf(returnType=MENTION_SCHEMA)
    def extract_mentions_udf(clean_text_lower: str | None) -> list[dict[str, str | int | None]]:
        return extract_mentions_from_text(clean_text_lower, figure_variants)

    document_text = spark.table(document_text_table).select(
        "document_id",
        "clean_text_lower",
    )

    documents = spark.table(documents_table).select(
        "document_id",
        "date_key",
    )

    dates = spark.table(dates_table).select(
        "date_key",
        "publication_year",
        "publication_month",
        "publication_day",
        "publication_date",
        "date_precision",
        "date_calendar",
    )

    fact_documents = spark.table(fact_documents_table).select(
        "document_id",
        "ocr_quality_flag",
        "included_in_analysis_flag",
    )

    figures = spark.table(figures_table).select(
        "figure_id",
        "canonical_name",
    )

    variants = spark.table(variants_table).select(
        "variant_id",
        "variant_type",
        "match_confidence",
    )

    source_documents = (
        document_text.join(documents, on="document_id", how="left")
        .join(dates, on="date_key", how="left")
        .join(fact_documents, on="document_id", how="left")
    )

    mentions = (
        source_documents.withColumn("mentions", extract_mentions_udf(F.col("clean_text_lower")))
        .select(
            "document_id",
            "date_key",
            "publication_year",
            "publication_month",
            "publication_day",
            "publication_date",
            "date_precision",
            "date_calendar",
            "ocr_quality_flag",
            "included_in_analysis_flag",
            F.explode_outer("mentions").alias("mention"),
        )
        .where(F.col("mention").isNotNull())
        .select(
            F.sha2(
                F.concat_ws(
                    "|",
                    F.col("document_id"),
                    F.col("mention.figure_id"),
                    F.col("mention.variant_id"),
                    F.col("mention.match_start_char").cast("string"),
                    F.col("mention.match_end_char").cast("string"),
                ),
                256,
            ).alias("mention_id"),
            "document_id",
            "date_key",
            "publication_year",
            "publication_month",
            "publication_day",
            "publication_date",
            "date_precision",
            "date_calendar",
            "ocr_quality_flag",
            "included_in_analysis_flag",
            F.col("mention.figure_id").alias("figure_id"),
            F.col("mention.variant_id").alias("variant_id"),
            F.col("mention.matched_variant").alias("matched_variant"),
            F.col("mention.variant_normalized").alias("variant_normalized"),
            F.col("mention.match_start_char").alias("match_start_char"),
            F.col("mention.match_end_char").alias("match_end_char"),
            F.col("mention.match_method").alias("match_method"),
        )
    )

    return (
        mentions.join(figures, on="figure_id", how="left")
        .join(variants, on="variant_id", how="left")
        .withColumn(
            "is_high_confidence_match",
            F.col("match_confidence") == F.lit("high"),
        )
        .withColumn(
            "is_analysis_ready",
            F.col("included_in_analysis_flag")
            & F.col("publication_year").isNotNull()
            & F.col("figure_id").isNotNull(),
        )
        .withColumn("extracted_at", F.current_timestamp())
        .select(
            "mention_id",
            "document_id",
            "figure_id",
            "variant_id",
            "date_key",
            "canonical_name",
            "matched_variant",
            "variant_normalized",
            "variant_type",
            "match_confidence",
            "is_high_confidence_match",
            "match_method",
            "match_start_char",
            "match_end_char",
            "publication_year",
            "publication_month",
            "publication_day",
            "publication_date",
            "date_precision",
            "date_calendar",
            "ocr_quality_flag",
            "included_in_analysis_flag",
            "is_analysis_ready",
            "extracted_at",
        )
        .orderBy("document_id", "match_start_char", "figure_id", "variant_id")
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
    """Extract figure mentions into a fact table."""
    args = parse_args()
    spark = SparkSession.builder.getOrCreate()

    mentions = build_fact_figure_mentions(
        spark=spark,
        catalog=args.catalog,
        schema=args.schema,
    )

    write_delta_table(
        mentions,
        table_name(args.catalog, args.schema, "silver_fact_figure_mentions"),
    )


if __name__ == "__main__":
    main()