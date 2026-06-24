# Policy Summary Pipeline Reproducibility Guide

This guide explains how to rerun the current Dutch car-insurance policy summary pipeline from source PDFs to final HTML/PDF summaries.

The big-picture explanation for non-technical readers lives in `docs/big_picture_pipeline_overview.md`.

## Scope

The repository now keeps only the files needed for the front-to-back pipeline:

```text
PDFs -> extraction artifacts -> broad summaries -> filtered summaries -> final HTML/PDF package
```

Optional review, cost, tracing, presentation, and document-generation support scripts have been removed.

## Inputs

Raw policy PDFs:

```text
data/raw/car_policies/*.pdf
```

OpenAI API key, required for the two AI model stages:

```text
OPENAI_API_KEY=...
```

The key can be placed in `.env`; the pipeline loads `.env` automatically.

## Outputs

The normal output roots are:

| Purpose | Path |
|---|---|
| Extraction artifacts and broad grouped summaries | `outputs/current_full_gpt4omini/` |
| Concise filtered summaries | `outputs/filter_gpt41_context_tags_fixed/` |
| Final HTML and PDF package | `outputs/final_policy_summaries/` |

The final package contains:

```text
outputs/final_policy_summaries/index.html
outputs/final_policy_summaries/<doc_id>/policy_summary.html
outputs/final_policy_summaries/<doc_id>/<doc_id>.pdf
outputs/final_policy_summaries/<doc_id>/<doc_id>_annotated.pdf
outputs/final_policy_summaries/<doc_id>/filtered_summary_items.json
```

The `outputs/` folder is ignored by git and can be regenerated.

## Setup

Install dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

Required packages are limited to the front-to-back pipeline:

| Package | Used for |
|---|---|
| `pandas`, `pyarrow`, `numpy` | Table processing and parquet files. |
| `python-dotenv` | Loading `.env`. |
| `openai` | Calling the AI model for extraction and filtering. |
| `marker-pdf`, `accelerate` | Reading PDF text and layout. |
| `beautifulsoup4`, `lxml` | Reading table HTML from PDF extraction. |
| `pymupdf` | Creating annotated PDF copies. |

## Pipeline Flow

| Step | Main code | Main outputs |
|---|---|---|
| Read PDFs | `src/ingestion/marker_converter.py` | `marker_json_outputs/*.json` |
| Flatten layout and build chunks | `src/ingestion/chunk_builder.py` | `documents.csv`, `pages.csv`, `chunks.parquet` |
| Build source statements | `src/canonical/build_corpus.py` | `canonical_clauses.parquet`, `sections.parquet` |
| Build AI input snippets | `src/llm/context_builder.py` | `llm_context_windows.parquet` |
| Extract possible facts | `src/llm/extractor.py`, `src/llm/prompts.py` | `llm_extractions_raw.parquet` |
| Check quotes and table roles | `src/llm/validator.py` | `llm_extractions.parquet` |
| Build broad grouped summaries | `src/llm/summary_builder.py` | `<doc_id>/visual_summary_items.json`, `<doc_id>/extracted_summary.json` |
| Filter for customer summary | `src/pipeline/run_summary_filter.py`, `src/llm/summary_filter.py` | `<doc_id>/filtered_summary_items.json`, `<doc_id>/policy_summary.html` |
| Package final summaries | `src/rendering/policy_summary.py`, `src/pdf/annotate.py` | final HTML pages, original PDFs, annotated PDFs |

## Run Front To Back

Run these commands from the repository root.

### 1. Build Extraction Artifacts And Broad Summaries

```bash
python -m src.pipeline.run_policy_summary_pipeline --all --output-dir outputs/current_full_gpt4omini --model gpt-4o-mini --cache-dir outputs/current_full_gpt4omini/cache_context_tags --max-concurrent 20
```

Important outputs:

```text
outputs/current_full_gpt4omini/documents.csv
outputs/current_full_gpt4omini/pages.csv
outputs/current_full_gpt4omini/chunks.parquet
outputs/current_full_gpt4omini/canonical_clauses.parquet
outputs/current_full_gpt4omini/sections.parquet
outputs/current_full_gpt4omini/llm_context_windows.parquet
outputs/current_full_gpt4omini/llm_extractions.parquet
outputs/current_full_gpt4omini/<doc_id>/visual_summary_items.json
outputs/current_full_gpt4omini/<doc_id>/extracted_summary.json
```

### 2. Filter To The Concise Customer Summary

```bash
python -m src.pipeline.run_summary_filter --input-dir outputs/current_full_gpt4omini --output-dir outputs/filter_gpt41_context_tags_fixed --all --model gpt-4.1 --max-concurrent 8 --fail-on-error
```

Important outputs:

```text
outputs/filter_gpt41_context_tags_fixed/<doc_id>/visual_summary_items.json
outputs/filter_gpt41_context_tags_fixed/<doc_id>/filtered_summary_items.json
outputs/filter_gpt41_context_tags_fixed/<doc_id>/summary_filter_metadata.json
outputs/filter_gpt41_context_tags_fixed/<doc_id>/filtered_summary_report.html
outputs/filter_gpt41_context_tags_fixed/<doc_id>/policy_summary.html
outputs/filter_gpt41_context_tags_fixed/summary_filter_run_metadata.json
```

### 3. Package The Final HTML Summaries With PDFs

```bash
python -m src.rendering.policy_summary --input-dir outputs/filter_gpt41_context_tags_fixed --output-dir outputs/final_policy_summaries --all --pdf-root data/raw/car_policies --no-open
```

Open the final result from:

```text
outputs/final_policy_summaries/index.html
```

## One-Document Check

These commands verify the current code without running the full 40-document pipeline.

Check filter input preparation without calling the AI provider:

```bash
python -m src.pipeline.run_summary_filter --input-dir outputs/current_full_gpt4omini --doc-ids ALL26 --dry-run
```

Render one final summary package:

```bash
python -m src.rendering.policy_summary --input-dir outputs/filter_gpt41_context_tags_fixed --output-dir outputs/smoke_final_policy_summaries --doc-ids ALL26 --pdf-root data/raw/car_policies --no-open
```

Optional raw-PDF check without OpenAI calls:

```bash
python -m src.pipeline.run_policy_summary_pipeline --doc-ids ALL26 --output-dir outputs/smoke_all26_no_llm --skip-llm --skip-html --max-concurrent 1
```

## Resume Options

The main pipeline can restart from saved artifacts with `--from-stage`.

| Stage | What must already exist in `--output-dir` |
|---|---|
| `pdf` | Raw PDFs in `--input-dir`. |
| `marker` | `marker_json_outputs/*.json`. |
| `chunks` | `chunks.parquet`. |
| `canonical` | `canonical_clauses.parquet` and `sections.parquet`. |
| `llm` | `llm_extractions.parquet`. |
| `summary` | `llm_extractions.parquet`. |

Examples:

```bash
python -m src.pipeline.run_policy_summary_pipeline --doc-ids ALL26 --output-dir outputs/current_full_gpt4omini --from-stage canonical --model gpt-4o-mini
python -m src.pipeline.run_policy_summary_pipeline --doc-ids ALL26 --output-dir outputs/current_full_gpt4omini --from-stage llm --skip-html
```

Use a separate output folder for experiments. Earlier stages rewrite shared table files such as `chunks.parquet` and `canonical_clauses.parquet`.

## Core Python Files

These are the only Python code files needed by the current front-to-back pipeline:

```text
src/pipeline/run_policy_summary_pipeline.py
src/ingestion/marker_converter.py
src/ingestion/chunk_builder.py
src/canonical/build_corpus.py
src/extraction/canonical_corpus.py
src/extraction/canonical_clause_extractor.py
src/chunking/semantic_enrichment.py
src/llm/context_builder.py
src/llm/prompts.py
src/llm/extractor.py
src/llm/validator.py
src/llm/summary_builder.py
src/pipeline/run_summary_filter.py
src/llm/summary_filter.py
src/rendering/policy_summary.py
src/pdf/annotate.py
```

Package `__init__.py` files are kept only so module imports work.

## Current Reference Counts

The current preferred run contains:

| Item | Count |
|---|---:|
| Policy documents | 40 |
| PDF pages | 1,073 |
| Small source statements | 22,400 |
| AI extraction candidates | 16,625 |
| Candidates with direct source-quote match | 16,551 |
| Candidates with close source-quote match | 53 |
| Broad grouped summary items | 12,212 |
| Final displayed summary items | 1,933 |
| Final source quote records | 4,931 |

## Limits

- The tool supports human review; it does not replace legal or compliance approval.
- Some administrative or definition text can still enter candidate outputs before filtering.
- Similar policy rules can appear more than once when source policies repeat them.
- PDF text quality affects all later stages.
- Privacy and data-sharing rules must be checked before sending policy text to an external AI provider.
- HTML plus annotated PDFs are the active final output.

## Files To Keep In Sync

When the pipeline changes, update these files together:

```text
README.md
docs/big_picture_pipeline_overview.md
docs/policy_summary_pipeline_reproducibility.md
```
