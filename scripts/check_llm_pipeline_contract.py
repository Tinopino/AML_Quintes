"""Cheap non-LLM contract check for the LLM policy-summary pipeline.

Builds a tiny synthetic chunk table, runs canonical clause construction and
LLM context-window construction, and verifies that prompt-visible traceability
markers are preserved while removed legacy signal metadata stays absent.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.extraction.canonical_corpus import build_canonical_clause_table, build_section_table
from src.llm.context_builder import build_llm_context_windows


def _synthetic_chunks() -> pd.DataFrame:
    """Return a minimal chunk table covering paired columns and hints."""

    return pd.DataFrame(
        [
            {
                "doc_id": "synthetic-policy",
                "chunk_id": "chunk-paired-1",
                "page_start": 1,
                "page_end": 1,
                "bbox": [10, 20, 300, 160],
                "chunk_kind": "table_paired_chunk",
                "section_path": "Autoverzekering > Casco",
                "title_text": "1. Dit is verzekerd 2. Dit is niet verzekerd",
                "text_raw": "Verzekerd: schade door brand. Niet verzekerd: schade door opzet.",
                "table_rows_json": json.dumps(
                    [
                        {
                            "row_idx": 0,
                            "insured": "Schade door brand is verzekerd.",
                            "not_insured": "Schade door opzet is niet verzekerd.",
                        }
                    ],
                    ensure_ascii=False,
                ),
            },
            {
                "doc_id": "synthetic-policy",
                "chunk_id": "chunk-limit-1",
                "page_start": 2,
                "page_end": 2,
                "bbox": [10, 170, 300, 240],
                "chunk_kind": "paragraph_chunk",
                "section_path": "Autoverzekering > Maximale vergoedingen",
                "title_text": "Maximale vergoedingen",
                "text_raw": "Maximale vergoeding: Wij betalen maximaal € 1.000 per gebeurtenis.",
                "table_rows_json": "",
            },
            {
                "doc_id": "synthetic-policy",
                "chunk_id": "chunk-covered-list-1",
                "page_start": 3,
                "page_end": 3,
                "bbox": [10, 250, 300, 320],
                "chunk_kind": "list_chunk",
                "section_path": "Autoverzekering > Casco",
                "title_text": "Verzekerd zijn",
                "text_raw": "Schade door storm is verzekerd.",
                "table_rows_json": "",
            },
            {
                "doc_id": "synthetic-policy",
                "chunk_id": "chunk-value-matrix-1",
                "page_start": 4,
                "page_end": 4,
                "bbox": [10, 330, 300, 390],
                "chunk_kind": "table_matrix_chunk",
                "section_path": "Autoverzekering > Waardebepaling",
                "title_text": "Samenvatting waardebepaling",
                "text_raw": "Samenvatting waardebepaling",
                "table_rows_json": json.dumps(
                    [
                        {"cells": [{"text": "Leeftijd auto"}, {"text": "Eerste jaar"}, {"text": "Na eerste jaar"}]},
                        {"cells": [{"text": "0 tot 1 jaar"}, {"text": "nieuwwaarde"}, {"text": "dagwaarde"}]},
                        {"cells": [{"text": "1 tot 2 jaar"}, {"text": "aanschafwaarde"}, {"text": "dagwaarde"}]},
                    ],
                    ensure_ascii=False,
                ),
            },
        ]
    )


def main() -> None:
    chunks_df = _synthetic_chunks()
    canonical_clauses = build_canonical_clause_table(chunks_df)
    sections = build_section_table(canonical_clauses)
    contexts = build_llm_context_windows(sections, canonical_clauses)

    required_clause_columns = {
        "clause_id",
        "clause_text_raw",
        "section_path",
        "page_start",
        "page_end",
        "bbox",
        "column_role",
        "category_hint",
    }
    missing_clause_columns = required_clause_columns - set(canonical_clauses.columns)
    assert not missing_clause_columns, f"Missing canonical clause columns: {sorted(missing_clause_columns)}"

    assert "signal_hint" not in canonical_clauses.columns, "signal_hint should not be emitted"
    assert "module_hint" not in canonical_clauses.columns, "module_hint should not be emitted"
    assert "module_hint_confidence" not in canonical_clauses.columns, "module_hint_confidence should not be emitted"
    assert "module_scores_json" not in canonical_clauses.columns, "module_scores_json should not be emitted"
    assert "dominant_signals" not in sections.columns, "dominant_signals should not be emitted"
    assert "module_hint" not in sections.columns, "section module_hint should not be emitted"
    assert "module_hint_confidence" not in sections.columns, "section module_hint_confidence should not be emitted"
    assert "module_scores_json" not in sections.columns, "section module_scores_json should not be emitted"
    assert not contexts.empty, "Expected at least one LLM context window"
    assert "module_hint" not in contexts.columns, "context module_hint should not be emitted"
    assert "module_hint_confidence" not in contexts.columns, "context module_hint_confidence should not be emitted"
    assert "module_scores_json" not in contexts.columns, "context module_scores_json should not be emitted"

    all_context_text = "\n\n".join(contexts["context_text"].astype(str).tolist())
    assert "[clause_id=" in all_context_text, "Context text is missing clause markers"
    assert "[column=insured]" in all_context_text, "Context text is missing insured column marker"
    assert "[column=not_insured]" in all_context_text, "Context text is missing not_insured column marker"
    assert "[hint=covered]" in all_context_text, "Context text is missing covered hint marker"
    assert "[hint=not_covered]" in all_context_text, "Context text is missing not_covered hint marker"
    assert "[hint=condition]" in all_context_text, "Context text is missing condition hint marker"
    assert "[hint=limit]" in all_context_text, "Context text is missing limit hint marker"

    print("LLM pipeline contract smoke check passed")


if __name__ == "__main__":
    main()
