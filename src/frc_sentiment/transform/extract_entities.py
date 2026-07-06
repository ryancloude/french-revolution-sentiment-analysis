"""Extract figure mentions from cleaned OCR text using dictionary matching."""

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
            T.StructField("canonical_name", T.StringType(), False),
            T.StructField("matched_variant", T.StringType(), False),
            T.StructField("variant_normalized", T.StringType(), False),
            T.StructField("variant_type", T.StringType(), True),
            T.StructField("match_confidence", T.StringType(), True),
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

    for figure in figure_variants:
        variant_normalized = figure["variant_normalized"]
        if not variant_normalized:
            continue

        pattern = build_variant_regex(variant_normalized)

        for match in pattern.finditer(clean_text_lower):
            mentions.append(
                {
                    "figure_id": figure["figure_id"],
                    "canonical_name": figure["canonical_name"],
                    "matched_variant": figure["variant"],
                    "variant_normalized": variant_normalized,
                    "variant_type": figure["variant_type"],
                    "match_confidence": figure["match_confidence"],
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
        )
    )

    return mentions


def get_figure_variants(
    spark: SparkSession,
    catalog: str,
    schema: str,
) -> list[dict[str, str | None]]:
    """Collect the figure lookup table to the driver."""
    figures_table = table_name(catalog, schema, "silver_figures")

    rows = (
        spark.table(figures_table)
        .select(
            "figure_id",
            "canonical_name",
            "variant",
            "variant_normalized",
            "variant_type",
            "match_confidence",
        )
        .where(F.col("variant_normalized").isNotNull())
        .collect()
    )

    return [row.asDict(recursive=True) for row in rows]


def build_entity_mentions(
    spark: SparkSession,
    catalog: str,
    schema: str,
) -> DataFrame:
    """Build the silver entity mentions table."""
    clean_text_table = table_name(catalog, schema, "silver_clean_text")
    selected_dates_table = table_name(catalog, schema, "silver_selected_publication_dates")

    figure_variants = get_figure_variants(spark, catalog, schema)

    if not figure_variants:
        raise RuntimeError("No figure variants found in silver_figures")

    @F.udf(returnType=MENTION_SCHEMA)
    def extract_mentions_udf(clean_text_lower: str | None) -> list[dict[str, str | int | None]]:
        return extract_mentions_from_text(clean_text_lower, figure_variants)

    selected_dates = spark.table(selected_dates_table).select(
        "document_id",
        "selected_publication_year",
        "selected_publication_month",
        "selected_publication_day",
        "selected_publication_date",
        "selected_date_precision",
        "selected_date_calendar",
        "selected_extractor_name",
        "selected_source_field",
        "selected_confidence",
        "selected_evidence",
        "conflicts_with_metadata_year",
    )

    documents = (
        spark.table(clean_text_table)
        .select(
            "document_id",
            "title",
            "clean_text_lower",
            "ocr_quality_flag",
        )
        .join(selected_dates, on="document_id", how="left")
    )

    mentions = (
        documents.withColumn("mentions", extract_mentions_udf(F.col("clean_text_lower")))
        .select(
            "document_id",
            "selected_publication_year",
            "selected_publication_month",
            "selected_publication_day",
            "selected_publication_date",
            "selected_date_precision",
            "selected_date_calendar",
            "selected_extractor_name",
            "selected_source_field",
            "selected_confidence",
            "selected_evidence",
            "conflicts_with_metadata_year",
            "title",
            "ocr_quality_flag",
            F.explode_outer("mentions").alias("mention"),
        )
        .where(F.col("mention").isNotNull())
        .select(
            F.sha2(
                F.concat_ws(
                    "|",
                    F.col("document_id"),
                    F.col("mention.figure_id"),
                    F.col("mention.match_start_char").cast("string"),
                    F.col("mention.match_end_char").cast("string"),
                    F.col("mention.variant_normalized"),
                ),
                256,
            ).alias("mention_id"),
            "document_id",
            F.col("selected_publication_year").alias("publication_year"),
            F.col("selected_publication_month").alias("publication_month"),
            F.col("selected_publication_day").alias("publication_day"),
            F.col("selected_publication_date").alias("publication_date"),
            F.col("selected_date_precision").alias("date_precision"),
            F.col("selected_date_calendar").alias("date_calendar"),
            F.col("selected_extractor_name").alias("date_extractor_name"),
            F.col("selected_source_field").alias("date_source_field"),
            F.col("selected_confidence").alias("date_confidence"),
            F.col("selected_evidence").alias("date_evidence"),
            "conflicts_with_metadata_year",
            "title",
            "ocr_quality_flag",
            F.col("mention.figure_id").alias("figure_id"),
            F.col("mention.canonical_name").alias("canonical_name"),
            F.col("mention.matched_variant").alias("matched_variant"),
            F.col("mention.variant_normalized").alias("variant_normalized"),
            F.col("mention.variant_type").alias("variant_type"),
            F.col("mention.match_confidence").alias("match_confidence"),
            F.col("mention.match_start_char").alias("match_start_char"),
            F.col("mention.match_end_char").alias("match_end_char"),
            F.col("mention.match_method").alias("match_method"),
            F.current_timestamp().alias("extracted_at"),
        )
    )

    return mentions


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
    """Extract figure mentions into a silver table."""
    args = parse_args()
    spark = SparkSession.builder.getOrCreate()

    mentions = build_entity_mentions(
        spark=spark,
        catalog=args.catalog,
        schema=args.schema,
    )

    write_delta_table(
        mentions,
        table_name(args.catalog, args.schema, "silver_entity_mentions"),
    )


if __name__ == "__main__":
    main()