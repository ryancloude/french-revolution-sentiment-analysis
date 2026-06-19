"""Download a reproducible sample of Newberry metadata and OCR text files."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

BASE_URL = (
    "https://raw.githubusercontent.com/"
    "NewberryDIS/frc-data/master"
)

SOURCE_FILES = {
    "metadata": ("Metadata", "_meta.xml"),
    "ocr_text": ("OCR_text", "_djvu.txt"),
}

DOCUMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def read_document_ids(manifest_path: Path) -> list[str]:
    """Read and validate document IDs from a newline-delimited manifest."""
    document_ids = [
        line.strip()
        for line in manifest_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

    invalid_ids = [
        document_id
        for document_id in document_ids
        if not DOCUMENT_ID_PATTERN.fullmatch(document_id)
    ]

    if invalid_ids:
        raise ValueError(f"Invalid document IDs: {invalid_ids}")

    if not document_ids:
        raise ValueError(f"No document IDs found in {manifest_path}")

    return document_ids


def download_file(url: str, destination: Path, overwrite: bool = False) -> bool:
    """Download one file atomically.

    Returns True when downloaded and False when an existing file was skipped.
    """
    if destination.exists() and not overwrite:
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".part")

    try:
        with urlopen(url, timeout=60) as response:
            with temporary_path.open("wb") as output_file:
                shutil.copyfileobj(response, output_file)

        temporary_path.replace(destination)
    except (HTTPError, URLError, TimeoutError):
        temporary_path.unlink(missing_ok=True)
        raise

    return True


def download_sample(
    manifest_path: Path,
    output_directory: Path,
    overwrite: bool = False,
) -> tuple[int, int]:
    """Download metadata and OCR text for every manifest document."""
    downloaded = 0
    skipped = 0

    for document_id in read_document_ids(manifest_path):
        for source_name, (source_directory, suffix) in SOURCE_FILES.items():
            filename = f"{document_id}{suffix}"
            url = f"{BASE_URL}/{source_directory}/{filename}"
            destination = output_directory / source_name / filename

            if download_file(url, destination, overwrite):
                downloaded += 1
            else:
                skipped += 1

    return downloaded, skipped


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download sample Newberry metadata and OCR text."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/sample/document_ids.txt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/sample"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def main() -> None:
    """Run the sample downloader."""
    args = parse_args()
    downloaded, skipped = download_sample(
        manifest_path=args.manifest,
        output_directory=args.output,
        overwrite=args.overwrite,
    )
    print(f"Downloaded: {downloaded}; skipped: {skipped}")


if __name__ == "__main__":
    main()