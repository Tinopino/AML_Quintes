"""Build canonical corpus tables from structural chunks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Any

import pandas as pd
from src.extraction.canonical_corpus import (
    build_canonical_clause_table,
    build_section_table,
    normalize_doc_ids,
)


def build_canonical_corpus(
    chunks_df: pd.DataFrame,
    output_dir: Path,
    *,
    doc_ids: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and save canonical clauses and sections.

    This wrapper intentionally preserves the original Marker section paths used
    by the earlier evaluated outputs from ``structure_chunks_enriched.parquet``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_df = add_passthrough_section_columns(chunks_df)
    selected_doc_ids = normalize_doc_ids(chunks_df, doc_ids=doc_ids)
    if not selected_doc_ids:
        raise ValueError("No matching doc_ids were found in the chunk dataset.")

    enriched_chunks_path = output_dir / "structure_chunks_enriched_with_sections.parquet"
    chunks_df.to_parquet(enriched_chunks_path, index=False)

    clauses_df = build_canonical_clause_table(chunks_df, doc_ids=selected_doc_ids)
    sections_df = build_section_table(clauses_df)
    clauses_df.to_parquet(output_dir / "canonical_clauses.parquet", index=False)
    sections_df.to_parquet(output_dir / "sections.parquet", index=False)
    write_json(output_dir / "canonical_metadata.json", {
        "enriched_chunks_path": str(enriched_chunks_path),
        "n_clauses": int(len(clauses_df)),
        "n_sections": int(len(sections_df)),
        "doc_ids": sorted(clauses_df["doc_id"].astype(str).unique().tolist()) if not clauses_df.empty else [],
    })
    return clauses_df, sections_df


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def add_passthrough_section_columns(chunks_df: pd.DataFrame) -> pd.DataFrame:
    """Add canonical section columns without inferring new section splits."""

    out = chunks_df.copy()
    if "chunk_id" not in out.columns:
        if "struct_chunk_id" not in out.columns:
            raise ValueError("chunks_df must contain chunk_id or struct_chunk_id")
        out["chunk_id"] = out["struct_chunk_id"]

    for column in ("section_path", "title_text"):
        if column not in out.columns:
            out[column] = ""

    out["section_path_original"] = out["section_path"].fillna("").astype(str)
    out["title_text_original"] = out["title_text"].fillna("").astype(str)
    out["section_path_enriched"] = out["section_path_original"]
    out["title_text_enriched"] = out["title_text_original"]
    out["heading_text_inferred"] = ""
    out["heading_level_inferred"] = 0
    out["heading_source"] = "original"
    out["section_enrichment_confidence"] = 1.0
    return out
