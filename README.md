# French Revolution Sentiment Analysis

This project analyzes how rhetorical stance toward major figures of the French Revolution changed over time using digitized primary-source pamphlets from the Newberry French Revolution Collection.

The main research question is:

> How did sentiment and rhetorical stance toward major figures of the French Revolution change across the course of the Revolution?

## Data source

The source data comes from the Newberry French Revolution Collection repository:

<https://github.com/NewberryDIS/frc-data>

The source repository contains digitized pamphlet data from the Newberry Library's French Revolution Collection.

Important source file types:

### OCR text files

Plain text files generated from optical character recognition.

These are the main input for:

- OCR text cleaning
- figure mention detection
- context window extraction
- stance scoring

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
- creator/author
- subject terms
- Internet Archive identifier
- source URL
- joining metadata to OCR text

In the source repository, metadata files follow this pattern:

```text
Metadata/<document_id>_meta.xml
```

### OCR XML files

Structured OCR XML files may contain page-level, line-level, or word-level OCR information.

These are not used in the current version of the pipeline. They may be added later for page-level traceability and improved context windows.

In the source repository, OCR XML files follow this pattern:

```text
XML_for_OCR/<document_id>_djvu.xml
```

## Architecture

The project follows a medallion architecture using bronze, silver, and gold layers.

```mermaid
flowchart LR
    source["Newberry FRC GitHub data"]
    raw["Raw files<br/>Unity Catalog volume"]
    bronze["Bronze tables<br/>raw OCR + metadata"]
    silver["Silver model<br/>facts, dimensions, bridges"]
    gold["Gold tables<br/>dashboard-ready aggregates"]
    powerbi["Power BI dashboard"]

    source --> raw
    raw --> bronze
    bronze --> silver
    silver --> gold
    gold --> powerbi
```

Databricks is used for the main pipeline execution. The project uses:

- Databricks Asset Bundles for job deployment
- Unity Catalog for table and volume organization
- Delta tables for bronze, silver, and gold data
- PySpark for distributed transformations
- Databricks SQL AI functions for publication date extraction and stance classification

The current Databricks namespace is parameterized in `databricks.yml`.

Example logical namespace:

```text
<catalog>.<schema>
```

Raw files are stored in a Unity Catalog volume:

```text
/Volumes/<catalog>/<schema>/raw/
```

## Data model

The silver layer uses a dimensional model with two main fact tables:

- `silver_fact_documents`
- `silver_fact_figure_mentions`

These facts are surrounded by document, date, text, figure, creator, subject, context, and stance dimensions.

See the full data model documentation here:

[Data model](data_model.md)

## Main tables

### Bronze tables

Bronze tables preserve raw source data with minimal transformation.

| Table | Grain | Purpose |
|---|---|---|
| `bronze_metadata` | One row per metadata XML file | Stores raw metadata XML. |
| `bronze_ocr_text` | One row per OCR text file | Stores raw OCR text. |

### Silver tables

Silver tables contain cleaned, parsed, normalized, and analysis-ready data.

| Table | Grain | Purpose |
|---|---|---|
| `silver_dim_documents` | One row per document | Document metadata such as title, language, source URL, and selected publication date. |
| `silver_fact_documents` | One row per document | Document-level measures and quality flags. |
| `silver_dim_dates` | One row per selected date or period | Shared date dimension for document and mention facts. |
| `silver_dim_document_text` | One row per document | Raw and cleaned OCR text. |
| `silver_dim_figures` | One row per tracked historical figure | Canonical figure records. |
| `silver_dim_figure_variants` | One row per figure-name variant | Dictionary entries used for matching figure mentions. |
| `silver_fact_figure_mentions` | One row per detected figure mention | Main mention-level fact table. |
| `silver_dim_mention_contexts` | One row per mention | Text window around each detected figure mention. |
| `silver_dim_stance_categories` | One row per stance category combination | Normalized stance label, intensity, confidence, relevance, and score. |
| `silver_stance_model_audit` | One row per model-scored mention | Raw AI response, evidence text, translation, and explanation. |
| `silver_dim_creators` | One row per creator/author name | Creator and author dimension. |
| `silver_bridge_document_creators` | One row per document-creator relationship | Handles documents with multiple creators. |
| `silver_dim_subjects` | One row per subject term | Subject metadata dimension. |
| `silver_bridge_document_subjects` | One row per document-subject relationship | Handles documents with multiple subjects. |
| `silver_publication_date_candidates` | One row per document/date candidate | Stores rule-based and AI-extracted date candidates. |

### Gold tables

Gold tables are dashboard-ready aggregates designed for Power BI.

| Table | Grain | Purpose |
|---|---|---|
| `gold_figure_mentions_by_period` | One row per figure per period | Mention volume and document coverage over time. |
| `gold_figure_stance_by_period` | One row per figure per period | Average stance and stance distribution over time. |
| `gold_figure_stance_by_document` | One row per figure per document | Document-level stance aggregation. |
| `gold_top_stance_contexts` | One row per selected passage | Representative positive and negative context windows. |
| `gold_figure_stance_by_creator_period` | One row per figure, creator, and period | Stance trends by creator or author. |
| `gold_figure_stance_by_subject_period` | One row per figure, subject, and period | Stance trends by subject metadata. |

## Figure lookup

The project currently tracks major figures of the French Revolution, including:

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

The lookup includes name variants and match-confidence levels.

## Stance scoring approach

The project scores stance at the context-window level, not the full-document level.

This matters because one pamphlet may mention multiple figures with different attitudes. A document-level sentiment score would lose that detail.

The stance pipeline works as follows:

1. detect a figure mention
2. extract surrounding context
3. send the context to Databricks `ai_query`
4. classify stance toward the specific figure
5. store normalized stance fields in the mention fact table
6. preserve raw model output and evidence in an audit table

The stance model produces fields such as:

- stance label
- stance intensity
- stance confidence
- target relevance
- deterministic stance score
- evidence text
- English translation
- explanation

The project treats stance scoring as an analytical signal, not as a perfect measurement of historical public opinion.