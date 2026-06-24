# Insurance Policy Summary Pipeline

End-to-end pipeline for Dutch car-insurance policy PDFs. The pipeline starts from raw PDF files and produces source-linked visual HTML summaries with annotated PDFs.

For a company-facing overview, see `docs/big_picture_pipeline_overview.md`.
For exact reproduction commands, see `docs/policy_summary_pipeline_reproducibility.md`.

## Input

```text
data/raw/car_policies/*.pdf
```

Each PDF should be named with its document ID, for example `ALL26.pdf`.

## Main Outputs

Extraction and broad summaries:

```text
outputs/current_full_gpt4omini/
```

Concise filtered summaries:

```text
outputs/filter_gpt41_context_tags_fixed/
```

Final package:

```text
outputs/final_policy_summaries/index.html
outputs/final_policy_summaries/{doc_id}/policy_summary.html
outputs/final_policy_summaries/{doc_id}/{doc_id}.pdf
outputs/final_policy_summaries/{doc_id}/{doc_id}_annotated.pdf
outputs/final_policy_summaries/{doc_id}/filtered_summary_items.json
```

The current final visual output is HTML plus annotated PDFs.

## Repository Layout

```text
data/raw/car_policies/      Source PDFs used by the pipeline
docs/                       Big-picture overview and reproducibility guide
src/                        Core pipeline code
outputs/                    Generated runs and final deliverables, ignored by git
```

## Setup

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Set your OpenAI API key in `.env`:

```text
OPENAI_API_KEY=...
```

## Run Front To Back

Run these commands from the repository root.

1. Build extraction artifacts and broad per-document summaries:

```bash
python -m src.pipeline.run_policy_summary_pipeline --all --output-dir outputs/current_full_gpt4omini --model gpt-4o-mini --cache-dir outputs/current_full_gpt4omini/cache_context_tags --max-concurrent 20
```

2. Filter broad summaries into concise customer-facing summaries:

```bash
python -m src.pipeline.run_summary_filter --input-dir outputs/current_full_gpt4omini --output-dir outputs/filter_gpt41_context_tags_fixed --all --model gpt-4.1 --max-concurrent 8 --fail-on-error
```

3. Package final HTML summaries with original and annotated PDFs:

```bash
python -m src.rendering.policy_summary --input-dir outputs/filter_gpt41_context_tags_fixed --output-dir outputs/final_policy_summaries --all --pdf-root data/raw/car_policies --no-open
```

Open `outputs/final_policy_summaries/index.html` to view the final package.

## Run One Document

```bash
python -m src.pipeline.run_policy_summary_pipeline --doc-ids ALL26 --output-dir outputs/policy_summary_pipeline/ALL26 --model gpt-4o-mini --max-concurrent 5
python -m src.pipeline.run_summary_filter --input-dir outputs/policy_summary_pipeline/ALL26 --output-dir outputs/filter_ALL26 --doc-ids ALL26 --model gpt-4.1 --fail-on-error
python -m src.rendering.policy_summary --input-dir outputs/filter_ALL26 --output-dir outputs/final_ALL26 --doc-ids ALL26 --pdf-root data/raw/car_policies --no-open
```

## Resume From Saved Artifacts

Use these only when the needed files already exist in `--output-dir`:

```bash
python -m src.pipeline.run_policy_summary_pipeline --doc-ids ALL26 --output-dir outputs/current_full_gpt4omini --from-stage chunks
python -m src.pipeline.run_policy_summary_pipeline --doc-ids ALL26 --output-dir outputs/current_full_gpt4omini --from-stage canonical --model gpt-4o-mini
python -m src.pipeline.run_policy_summary_pipeline --doc-ids ALL26 --output-dir outputs/current_full_gpt4omini --from-stage llm --skip-html
```

Stage meanings:

- `pdf`: parse raw PDFs and run everything.
- `marker`: load existing `marker_json_outputs/*.json`, rebuild blocks/chunks, and continue.
- `chunks`: load `chunks.parquet`, rebuild source statements, and continue.
- `canonical`: load `canonical_clauses.parquet` and `sections.parquet`, run AI extraction, and continue.
- `llm`: load `llm_extractions.parquet` and regenerate grouped summaries.
- `summary`: same as `llm`, intended for summary-only reruns.

## Core Code Map

Only these Python files are part of the current front-to-back pipeline:

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

Package `__init__.py` files are present only so `python -m src...` imports work.

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

All final displayed items preserve source quotes, source pages, and source IDs so they can be checked against the policy.

## Known Limitations

- The tool supports human review; it does not replace legal or compliance approval.
- Some definitions or administrative clauses can still enter candidates before filtering.
- Similar clauses can appear more than once when policies repeat the same rule.
- PDF text quality depends on the source PDF extraction.
- The current final visual output is HTML and annotated PDFs.
- Privacy and data-sharing requirements must be checked before sending policy text to an external AI provider.
