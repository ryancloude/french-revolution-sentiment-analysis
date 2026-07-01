# French Revolution Sentiment Analysis

This project analyzes how sentiment and rhetorical stance toward major figures of the French Revolution changed over time using digitized primary-source pamphlets from the Newberry French Revolution Collection.

The main research question is:

> How did sentiment toward major figures of the French Revolution change across the course of the Revolution?

This is a portfolio project focused on data engineering, PySpark, Databricks, Delta tables, NLP/text processing, and Power BI-ready analytics outputs.

## Current status

The current milestone builds a working ingestion and silver-layer preparation pipeline for a small sample of Newberry French Revolution Collection files.

Completed so far:

- Databricks Asset Bundle setup
- Unity Catalog schema and raw volume bootstrap
- local sample download process
- bronze metadata table
- bronze OCR text table
- silver parsed document metadata table
- silver cleaned OCR text table
- manually maintained figure lookup table
- dictionary-based entity mention extraction
- context window extraction around detected figure mentions
- data quality summary tables

Sentiment scoring and Power BI reporting are not implemented yet.

## Data source

The source data comes from the Newberry French Revolution Collection repository:

<https://github.com/NewberryDIS/frc-data>

The source repository contains digitized pamphlet data from the Newberry Library’s French Revolution Collection.

Important source file types:

### OCR text files

Plain text files generated from optical character recognition.

These are the main input for:

- OCR text cleaning
- figure mention detection
- context window extraction
- later sentiment or stance scoring

In the source repository, OCR text files follow this pattern:

```text
OCR_text/<document_id>_djvu.txt
```

### Metadata XML files

XML files containing bibliographic and archival metadata.

These are used for:

- document ID
- title
- publication year/date
- language
- Internet Archive identifier
- source URL
- joining metadata to OCR text

In the source repository, metadata files follow this pattern:

```text
Metadata/<document_id>_meta.xml
```

### OCR XML files

Structured OCR XML files may contain page-level, line-level, or word-level OCR information.

These are not used in the first version of the pipeline. They may be added later for page-level traceability and improved context windows.

In the source repository, OCR XML files follow this pattern:

```text
XML_for_OCR/<document_id>_djvu.xml
```

## Architecture

The project follows a bronze, silver, and gold data model.

Current architecture:

```text
Newberry GitHub data
    ↓
local raw sample files
    ↓
Databricks volume
    ↓
bronze Delta tables
    ↓
silver parsed and cleaned tables
    ↓
future gold dashboard-ready tables
    ↓
future Power BI dashboard
```

Current Databricks namespace:

```text
workspace.frc_sentiment
```

Raw files are stored in the Unity Catalog volume:

```text
/Volumes/workspace/frc_sentiment/raw/
```

## Current tables

### Bronze tables

Bronze tables preserve raw source data with minimal transformation.

```text
workspace.frc_sentiment.bronze_metadata
workspace.frc_sentiment.bronze_ocr_text
```

`bronze_metadata` contains one row per metadata XML file.

Key columns:

- `document_id`
- `file_path`
- `raw_metadata_xml`
- `ingested_at`

`bronze_ocr_text` contains one row per OCR text file.

Key columns:

- `document_id`
- `file_path`
- `raw_text`
- `source_file_type`
- `ingested_at`

### Silver tables

Silver tables are cleaned, parsed, or analysis-ready.

```text
workspace.frc_sentiment.silver_documents
workspace.frc_sentiment.silver_clean_text
workspace.frc_sentiment.silver_figures
workspace.frc_sentiment.silver_entity_mentions
workspace.frc_sentiment.silver_context_windows
workspace.frc_sentiment.silver_data_quality_summary
workspace.frc_sentiment.silver_figure_mention_summary
```

`silver_documents` parses metadata XML into document-level fields such as title, language, source URL, and publication year.

`silver_clean_text` normalizes OCR whitespace, creates lowercase matching text, calculates text length metrics, and assigns OCR quality flags.

`silver_figures` loads a manually maintained figure lookup CSV from:

```text
data/lookup/figures.csv
```

`silver_entity_mentions` uses dictionary-based matching to detect figure mentions in cleaned OCR text.

`silver_context_windows` extracts word windows around each detected mention for later sentiment or stance scoring.

`silver_data_quality_summary` stores reusable pipeline-level quality metrics.

`silver_figure_mention_summary` summarizes detected mentions by figure and matched variant.

## Figure lookup

The first version tracks these figures:

- Louis XVI
- Marie Antoinette
- Jacques Necker
- Mirabeau
- Lafayette
- Emmanuel-Joseph Sieyès
- Maximilien Robespierre
- Georges Danton
- Jean-Paul Marat
- Jacques Pierre Brissot
- Louis Antoine de Saint-Just
- Paul Barras
- Philippe Égalité

The lookup includes name variants and confidence levels.

Highly ambiguous political terms are avoided where they would create too many false positives. For example, plain `égalité` is not treated as a mention of Philippe Égalité because it usually refers to the political concept of equality.

## Reproducing the current pipeline

Authenticate to Databricks:

```powershell
databricks auth login --host <workspace-url> --profile <profile-name>
```

Validate and deploy the Databricks Asset Bundle:

```powershell
databricks bundle validate --profile <profile-name>
databricks bundle deploy --profile <profile-name>
```

Bootstrap the Databricks schema and raw volume:

```powershell
databricks bundle run bootstrap_infrastructure --profile <profile-name>
```

Load bronze tables:

```powershell
databricks bundle run bronze_ingestion --profile <profile-name>
```

Build silver metadata:

```powershell
databricks bundle run silver_metadata --profile <profile-name>
```

Build silver cleaned OCR text:

```powershell
databricks bundle run silver_clean_text --profile <profile-name>
```

Load figure lookup:

```powershell
databricks bundle run silver_figures --profile <profile-name>
```

Extract entity mentions:

```powershell
databricks bundle run silver_entity_mentions --profile <profile-name>
```

Build context windows:

```powershell
databricks bundle run silver_context_windows --profile f<profile-name>
```

Build data quality summaries:

```powershell
databricks bundle run silver_quality_summary --profile <profile-name>
```

## Known limitations

Current limitations:

- The pipeline currently uses a small sample of documents.
- OCR quality varies and may contain encoding artifacts.
- Metadata dates are currently parsed primarily at the year level.
- Entity extraction is dictionary-based and may miss spelling variants.
- Some title-based variants, such as `le roi` or `la reine`, are ambiguous.
- Context windows do not yet use page-level OCR XML.
- Sentiment or stance scoring has not been implemented yet.
- The current pipeline should not be interpreted as measuring public opinion directly. It analyzes rhetoric in surviving pamphlet texts.

## Planned next steps

Near-term next steps:

1. Improve publication date extraction beyond year-level metadata.
2. Add tests for metadata parsing, text cleaning, and entity matching.
3. Expand from the initial sample to a larger document set.
4. Build an explainable sentiment or stance scoring method for context windows.
5. Create gold tables for Power BI.
6. Build an interactive Power BI dashboard.
```