"""Build date candidate tables from deterministic publication-date extractors."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


def add_src_to_pythonpath() -> None:
    """Add the synced bundle src directory to sys.path when running as a Databricks script."""
    argv_path = Path(sys.argv[0])
    if not argv_path.exists():
        return

    resolved_script_path = argv_path.resolve()

    # Expected path:
    # .../files/src/frc_sentiment/transform/build_date_candidates.py
    src_path = resolved_script_path.parents[2]

    if src_path.name == "src" and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


add_src_to_pythonpath()

from frc_sentiment.transform.parse_metadata import parse_date_from_text  # noqa: E402

VALID_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")

DATE_CANDIDATE_SCHEMA = T.ArrayType(
    T.StructType(
        [
            T.StructField("extractor_name", T.StringType(), False),
            T.StructField("source_field", T.StringType(), False),
            T.StructField("source_text_excerpt", T.StringType(), True),
            T.StructField("publication_year", T.IntegerType(), True),
            T.StructField("publication_month", T.IntegerType(), True),
            T.StructField("publication_day", T.IntegerType(), True),
            T.StructField("publication_date", T.StringType(), True),
            T.StructField("date_precision", T.StringType(), False),
            T.StructField("date_calendar", T.StringType(), False),
            T.StructField("confidence", T.StringType(), False),
            T.StructField("evidence", T.StringType(), True),
            T.StructField("notes", T.StringType(), True),
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


def source_excerpt(value: str | None, max_length: int = 500) -> str | None:
    """Return a compact source excerpt for review/debugging."""
    if not value:
        return None

    cleaned = " ".join(value.split())
    if len(cleaned) <= max_length:
        return cleaned

    return cleaned[:max_length].rstrip() + "..."


def build_rule_candidates_for_document(
    publication_date_raw: str | None,
    title: str | None,
    ocr_front_matter: str | None,
) -> list[dict[str, str | int | None]]:
    """Build rule-based date candidates for one document.

    This intentionally returns candidates from each source independently.
    Final date selection happens separately.
    """
    sources = [
        ("metadata_date", publication_date_raw, "high"),
        ("title", title, "medium"),
        ("ocr_front_matter", ocr_front_matter, "low"),
    ]

    candidates = []

    for source_field, source_text, confidence in sources:
        parsed = parse_date_from_text(
            source_text,
            source=source_field,
            confidence=confidence,
        )

        if parsed["date_precision"] == "unknown":
            continue

        candidates.append(
            {
                "extractor_name": "rules",
                "source_field": source_field,
                "source_text_excerpt": source_excerpt(source_text),
                "publication_year": parsed["publication_year"],
                "publication_month": parsed["publication_month"],
                "publication_day": parsed["publication_day"],
                "publication_date": parsed["publication_date"],
                "date_precision": parsed["date_precision"],
                "date_calendar": parsed["date_calendar"],
                "confidence": parsed["date_parse_confidence"],
                # Current rules parser does not yet expose the exact matched substring.
                # We keep source_text_excerpt for auditability and reserve evidence for
                # future Spark NLP / LLM / enhanced-regex extractors.
                "evidence": None,
                "notes": parsed["date_parse_notes"],
            }
        )

    return candidates


def build_date_candidates(
    spark: SparkSession,
    catalog: str,
    schema: str,
) -> DataFrame:
    """Build the silver date candidates table."""
    silver_documents_table = table_name(catalog, schema, "silver_documents")

    @F.udf(returnType=DATE_CANDIDATE_SCHEMA)
    def rule_candidates_udf(
        publication_date_raw: str | None,
        title: str | None,
        ocr_front_matter: str | None,
    ) -> list[dict[str, str | int | None]]:
        return build_rule_candidates_for_document(
            publication_date_raw=publication_date_raw,
            title=title,
            ocr_front_matter=ocr_front_matter,
        )

    documents = spark.table(silver_documents_table).select(
        "document_id",
        "title",
        "publication_date_raw",
        "ocr_front_matter",
        F.col("publication_year").alias("selected_publication_year"),
        F.col("publication_month").alias("selected_publication_month"),
        F.col("publication_day").alias("selected_publication_day"),
        F.col("publication_date").alias("selected_publication_date"),
        F.col("date_precision").alias("selected_date_precision"),
        F.col("date_source").alias("selected_date_source"),
        F.col("date_calendar").alias("selected_date_calendar"),
    )

    return (
        documents.withColumn(
            "candidates",
            rule_candidates_udf(
                F.col("publication_date_raw"),
                F.col("title"),
                F.col("ocr_front_matter"),
            ),
        )
        .select(
            "document_id",
            "title",
            "publication_date_raw",
            "selected_publication_year",
            "selected_publication_month",
            "selected_publication_day",
            "selected_publication_date",
            "selected_date_precision",
            "selected_date_source",
            "selected_date_calendar",
            F.explode_outer("candidates").alias("candidate"),
        )
        .where(F.col("candidate").isNotNull())
        .select(
            F.sha2(
                F.concat_ws(
                    "|",
                    F.col("document_id"),
                    F.col("candidate.extractor_name"),
                    F.col("candidate.source_field"),
                    F.coalesce(F.col("candidate.publication_date"), F.lit("")),
                    F.coalesce(F.col("candidate.publication_year").cast("string"), F.lit("")),
                    F.coalesce(F.col("candidate.publication_month").cast("string"), F.lit("")),
                    F.coalesce(F.col("candidate.publication_day").cast("string"), F.lit("")),
                    F.col("candidate.date_precision"),
                    F.col("candidate.date_calendar"),
                ),
                256,
            ).alias("date_candidate_id"),
            "document_id",
            "title",
            "publication_date_raw",
            F.col("candidate.extractor_name").alias("extractor_name"),
            F.col("candidate.source_field").alias("source_field"),
            F.col("candidate.source_text_excerpt").alias("source_text_excerpt"),
            F.col("candidate.publication_year").alias("publication_year"),
            F.col("candidate.publication_month").alias("publication_month"),
            F.col("candidate.publication_day").alias("publication_day"),
            F.to_date(F.col("candidate.publication_date")).alias("publication_date"),
            F.col("candidate.date_precision").alias("date_precision"),
            F.col("candidate.date_calendar").alias("date_calendar"),
            F.col("candidate.confidence").alias("confidence"),
            F.col("candidate.evidence").alias("evidence"),
            F.col("candidate.notes").alias("notes"),
            (
                (F.col("candidate.source_field") == F.col("selected_date_source"))
                & (F.col("candidate.date_precision") == F.col("selected_date_precision"))
                & (F.col("candidate.date_calendar") == F.col("selected_date_calendar"))
                & (
                    F.col("candidate.publication_year").eqNullSafe(
                        F.col("selected_publication_year")
                    )
                )
                & (
                    F.col("candidate.publication_month").eqNullSafe(
                        F.col("selected_publication_month")
                    )
                )
                & (
                    F.col("candidate.publication_day").eqNullSafe(
                        F.col("selected_publication_day")
                    )
                )
            ).alias("selected_in_silver_documents"),
            F.current_timestamp().alias("candidate_created_at"),
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
    """Create the silver date candidates table."""
    args = parse_args()
    spark = SparkSession.builder.getOrCreate()

    candidates = build_date_candidates(
        spark=spark,
        catalog=args.catalog,
        schema=args.schema,
    )

    write_delta_table(
        candidates,
        table_name(args.catalog, args.schema, "silver_date_candidates"),
    )


if __name__ == "__main__":
    main()