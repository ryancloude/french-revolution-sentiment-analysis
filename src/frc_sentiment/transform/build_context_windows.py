"""Build context windows around detected figure mentions."""

from __future__ import annotations

import argparse
import re

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

VALID_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")

CONTEXT_SCHEMA = T.StructType(
    [
        T.StructField("context_window", T.StringType(), False),
        T.StructField("context_start_char", T.IntegerType(), False),
        T.StructField("context_end_char", T.IntegerType(), False),
        T.StructField("words_before_actual", T.IntegerType(), False),
        T.StructField("words_after_actual", T.IntegerType(), False),
        T.StructField("context_word_count", T.IntegerType(), False),
    ]
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


def word_spans(text: str) -> list[tuple[int, int]]:
    """Return start and end character offsets for each non-whitespace token."""
    return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]


def build_context_for_mention(
    clean_text: str | None,
    match_start_char: int | None,
    match_end_char: int | None,
    words_before: int,
    words_after: int,
) -> dict[str, str | int]:
    """Extract a word-based context window around one mention."""
    if not clean_text or match_start_char is None or match_end_char is None:
        return {
            "context_window": "",
            "context_start_char": 0,
            "context_end_char": 0,
            "words_before_actual": 0,
            "words_after_actual": 0,
            "context_word_count": 0,
        }

    spans = word_spans(clean_text)

    if not spans:
        return {
            "context_window": "",
            "context_start_char": 0,
            "context_end_char": 0,
            "words_before_actual": 0,
            "words_after_actual": 0,
            "context_word_count": 0,
        }

    mention_word_indexes = [
        index
        for index, (start, end) in enumerate(spans)
        if start < match_end_char and end > match_start_char
    ]

    if not mention_word_indexes:
        return {
            "context_window": "",
            "context_start_char": 0,
            "context_end_char": 0,
            "words_before_actual": 0,
            "words_after_actual": 0,
            "context_word_count": 0,
        }

    first_mention_word_index = min(mention_word_indexes)
    last_mention_word_index = max(mention_word_indexes)

    context_start_word_index = max(0, first_mention_word_index - words_before)
    context_end_word_index = min(len(spans) - 1, last_mention_word_index + words_after)

    context_start_char = spans[context_start_word_index][0]
    context_end_char = spans[context_end_word_index][1]

    context_window = clean_text[context_start_char:context_end_char].strip()

    words_before_actual = first_mention_word_index - context_start_word_index
    words_after_actual = context_end_word_index - last_mention_word_index
    context_word_count = context_end_word_index - context_start_word_index + 1

    return {
        "context_window": context_window,
        "context_start_char": context_start_char,
        "context_end_char": context_end_char,
        "words_before_actual": words_before_actual,
        "words_after_actual": words_after_actual,
        "context_word_count": context_word_count,
    }


def build_context_windows(
    spark: SparkSession,
    catalog: str,
    schema: str,
    words_before: int,
    words_after: int,
) -> DataFrame:
    """Build context windows for each entity mention."""
    mentions_table = table_name(catalog, schema, "silver_entity_mentions")
    clean_text_table = table_name(catalog, schema, "silver_clean_text")

    @F.udf(returnType=CONTEXT_SCHEMA)
    def context_window_udf(
        clean_text: str | None,
        match_start_char: int | None,
        match_end_char: int | None,
    ) -> dict[str, str | int]:
        return build_context_for_mention(
            clean_text=clean_text,
            match_start_char=match_start_char,
            match_end_char=match_end_char,
            words_before=words_before,
            words_after=words_after,
        )

    mentions = spark.table(mentions_table)

    clean_text = spark.table(clean_text_table).select(
        "document_id",
        "clean_text",
        "file_path",
    )

    return (
        mentions.join(clean_text, on="document_id", how="left")
        .withColumn(
            "context",
            context_window_udf(
                F.col("clean_text"),
                F.col("match_start_char"),
                F.col("match_end_char"),
            ),
        )
        .select(
            "mention_id",
            "document_id",
            "figure_id",
            "canonical_name",
            "matched_variant",
            "variant_normalized",
            "variant_type",
            "match_confidence",
            "match_method",
            "publication_year",
            "publication_month",
            "publication_day",
            "publication_date",
            "date_precision",
            "date_calendar",
            "date_extractor_name",
            "date_source_field",
            "date_confidence",
            "date_evidence",
            "conflicts_with_metadata_year",
            "title",
            "ocr_quality_flag",
            "match_start_char",
            "match_end_char",
            F.lit(words_before).alias("words_before_requested"),
            F.lit(words_after).alias("words_after_requested"),
            F.col("context.words_before_actual").alias("words_before_actual"),
            F.col("context.words_after_actual").alias("words_after_actual"),
            F.col("context.context_word_count").alias("context_word_count"),
            F.col("context.context_start_char").alias("context_start_char"),
            F.col("context.context_end_char").alias("context_end_char"),
            F.col("context.context_window").alias("context_window"),
            "file_path",
            F.current_timestamp().alias("context_extracted_at"),
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
    parser.add_argument("--words-before", type=int, default=75)
    parser.add_argument("--words-after", type=int, default=75)
    return parser.parse_args()


def main() -> None:
    """Create silver context windows."""
    args = parse_args()
    spark = SparkSession.builder.getOrCreate()

    context_windows = build_context_windows(
        spark=spark,
        catalog=args.catalog,
        schema=args.schema,
        words_before=args.words_before,
        words_after=args.words_after,
    )

    write_delta_table(
        context_windows,
        table_name(args.catalog, args.schema, "silver_context_windows"),
    )


if __name__ == "__main__":
    main()