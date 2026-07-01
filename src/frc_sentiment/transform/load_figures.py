"""Load the manually maintained figure lookup CSV into a silver Delta table."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

VALID_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
VALID_RELATIVE_PATH = re.compile(r"^[A-Za-z0-9_./-]+$")

FIGURE_COLUMNS = [
    "figure_id",
    "canonical_name",
    "variant",
    "variant_normalized",
    "variant_type",
    "match_confidence",
    "notes",
]

FIGURE_SCHEMA = T.StructType(
    [
        T.StructField("figure_id", T.StringType(), True),
        T.StructField("canonical_name", T.StringType(), True),
        T.StructField("variant", T.StringType(), True),
        T.StructField("variant_normalized", T.StringType(), True),
        T.StructField("variant_type", T.StringType(), True),
        T.StructField("match_confidence", T.StringType(), True),
        T.StructField("notes", T.StringType(), True),
    ]
)


def quote_identifier(identifier: str) -> str:
    """Validate and quote a Unity Catalog identifier."""
    if not VALID_IDENTIFIER.fullmatch(identifier):
        raise ValueError(f"Invalid Unity Catalog identifier: {identifier!r}")

    return f"`{identifier}`"


def validate_relative_path(path: str) -> str:
    """Validate a simple bundle-relative file path."""
    if not VALID_RELATIVE_PATH.fullmatch(path):
        raise ValueError(f"Invalid relative path: {path!r}")

    if ".." in Path(path).parts:
        raise ValueError(f"Parent-directory references are not allowed: {path!r}")

    return path


def get_script_path() -> Path | None:
    """Get the executing script path in local Python or Databricks script execution."""
    file_value = globals().get("__file__")
    if file_value:
        file_path = Path(file_value)
        if file_path.exists():
            return file_path.resolve()

    argv_path = Path(sys.argv[0])
    if argv_path.exists():
        return argv_path.resolve()

    return None


def resolve_lookup_path(figures_path: str) -> Path:
    """Resolve a lookup path relative to the synced bundle files root."""
    path = Path(figures_path)

    if path.is_absolute():
        return path

    validate_relative_path(figures_path)

    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate.resolve()

    script_path = get_script_path()
    if script_path is None:
        raise FileNotFoundError(
            f"Could not resolve {figures_path!r}. Current working directory is {Path.cwd()}"
        )

    # Local repo:
    #   src/frc_sentiment/transform/load_figures.py -> repo root is parents[3]
    #
    # Databricks bundle:
    #   /Workspace/.../files/src/frc_sentiment/transform/load_figures.py
    #   /Workspace/.../files/data/lookup/figures.csv
    bundle_files_root = script_path.parents[3]
    bundle_candidate = bundle_files_root / path

    if not bundle_candidate.exists():
        raise FileNotFoundError(
            f"Figure lookup CSV not found at {bundle_candidate}. "
            f"Script path was {script_path}; current working directory is {Path.cwd()}"
        )

    return bundle_candidate


def table_name(catalog: str, schema: str, table: str) -> str:
    """Build a fully qualified Unity Catalog table name."""
    return ".".join(
        [
            quote_identifier(catalog),
            quote_identifier(schema),
            quote_identifier(table),
        ]
    )


def clean_csv_value(value: str | None) -> str | None:
    """Trim CSV values and convert blank strings to null."""
    if value is None:
        return None

    cleaned = value.strip()
    return cleaned or None


def read_figure_rows(figures_path: Path) -> list[dict[str, str | None]]:
    """Read the figure lookup CSV on the driver."""
    if not figures_path.exists():
        raise FileNotFoundError(f"Figure lookup CSV not found: {figures_path}")

    with figures_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        fieldnames = reader.fieldnames or []
        missing_columns = sorted(set(FIGURE_COLUMNS) - set(fieldnames))
        if missing_columns:
            raise RuntimeError(f"Figure lookup CSV is missing columns: {missing_columns}")

        rows = [
            {column: clean_csv_value(row.get(column)) for column in FIGURE_COLUMNS}
            for row in reader
        ]

    if not rows:
        raise RuntimeError(f"Figure lookup CSV has no data rows: {figures_path}")

    return rows


def normalize_variant(column: F.Column) -> F.Column:
    """Normalize figure variants for case-insensitive text matching."""
    return F.trim(F.regexp_replace(F.lower(column), r"\s+", " "))


def load_figures(spark: SparkSession, figures_path: Path) -> DataFrame:
    """Read and validate the figure lookup CSV."""
    rows = read_figure_rows(figures_path)

    figures = (
        spark.createDataFrame(rows, schema=FIGURE_SCHEMA)
        .withColumn(
            "variant_normalized",
            F.when(
                F.col("variant_normalized").isNull() | (F.col("variant_normalized") == ""),
                normalize_variant(F.col("variant")),
            ).otherwise(normalize_variant(F.col("variant_normalized"))),
        )
        .withColumn("loaded_at", F.current_timestamp())
    )

    invalid_rows = figures.where(
        (F.col("figure_id").isNull())
        | (F.col("figure_id") == "")
        | (F.col("canonical_name").isNull())
        | (F.col("canonical_name") == "")
        | (F.col("variant").isNull())
        | (F.col("variant") == "")
        | (F.col("variant_normalized").isNull())
        | (F.col("variant_normalized") == "")
        | (~F.col("match_confidence").isin("high", "medium", "low"))
    )

    invalid_count = invalid_rows.count()
    if invalid_count > 0:
        invalid_rows.show(truncate=False)
        raise RuntimeError(f"Found {invalid_count} invalid figure lookup rows")

    duplicate_count = (
        figures.groupBy("figure_id", "variant_normalized")
        .count()
        .where(F.col("count") > 1)
        .count()
    )

    if duplicate_count > 0:
        raise RuntimeError("Found duplicate figure_id + variant_normalized rows")

    return figures


def write_delta_table(df: DataFrame, full_table_name: str) -> None:
    """Overwrite a Delta table."""
    row_count = df.count()

    if row_count == 0:
        raise RuntimeError(f"No rows found for {full_table_name}")

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
    parser.add_argument("--figures-path", required=True)
    return parser.parse_args()


def main() -> None:
    """Load the figure lookup CSV into the silver layer."""
    args = parse_args()
    spark = SparkSession.builder.getOrCreate()

    figures_path = resolve_lookup_path(args.figures_path)
    print(f"Reading figure lookup from: {figures_path}")

    figures = load_figures(spark, figures_path)

    write_delta_table(
        figures,
        table_name(args.catalog, args.schema, "silver_figures"),
    )


if __name__ == "__main__":
    main()