"""Create the project's Unity Catalog schema and managed volume."""

from __future__ import annotations

import argparse
import re

VALID_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


def quote_identifier(identifier: str) -> str:
    """Validate and quote a Unity Catalog identifier."""
    if not VALID_IDENTIFIER.fullmatch(identifier):
        raise ValueError(f"Invalid Unity Catalog identifier: {identifier!r}")

    return f"`{identifier}`"


def parse_args() -> argparse.Namespace:
    """Parse infrastructure configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--volume", required=True)
    return parser.parse_args()


def main() -> None:
    """Create the schema and volume if they do not already exist."""
    from pyspark.sql import SparkSession

    args = parse_args()
    spark = SparkSession.builder.getOrCreate()

    catalog = quote_identifier(args.catalog)
    schema = quote_identifier(args.schema)
    volume = quote_identifier(args.volume)

    schema_name = f"{catalog}.{schema}"
    volume_name = f"{schema_name}.{volume}"

    spark.sql(
        f"""
        CREATE SCHEMA IF NOT EXISTS {schema_name}
        COMMENT 'French Revolution sentiment analysis project'
        """
    )

    spark.sql(
        f"""
        CREATE VOLUME IF NOT EXISTS {volume_name}
        COMMENT 'Raw OCR and metadata source files'
        """
    )

    print(f"Schema ready: {schema_name}")
    print(f"Volume ready: {volume_name}")


if __name__ == "__main__":
    main()