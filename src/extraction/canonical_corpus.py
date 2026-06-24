"""Build canonical clause and section tables for LLM extraction."""

from __future__ import annotations

import json
from typing import Any, Iterable

import pandas as pd

from src.extraction.canonical_clause_extractor import (
    build_clause_candidates_for_doc,
    build_item_id,
)


def build_canonical_clause_table(
    chunks_df: pd.DataFrame,
    *,
    doc_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build one row per atomic clause candidate with full provenance."""

    selected_doc_ids = list(normalize_doc_ids(chunks_df, doc_ids=doc_ids))
    rows: list[dict[str, Any]] = []

    for doc_id in selected_doc_ids:
        _, _, clause_candidates = build_clause_candidates_for_doc(chunks_df, doc_id)
        for candidate in clause_candidates:
            rows.append(
                {
                    "doc_id": str(candidate.get("doc_id") or doc_id),
                    "clause_id": build_item_id(
                        doc_id=str(candidate.get("doc_id") or doc_id),
                        chunk_id=str(candidate.get("chunk_id") or ""),
                        paired_row_idx=to_int_or_none(candidate.get("paired_row_idx")),
                        clause_index=to_int(candidate.get("clause_position_in_group") or candidate.get("clause_index")),
                    ),
                    "source_chunk_id": str(candidate.get("chunk_id") or ""),
                    "page_start": to_int(candidate.get("page_start")),
                    "page_end": to_int(candidate.get("page_end")),
                    "bbox": to_json_text(candidate.get("bbox")),
                    "section_id": str(candidate.get("section_id") or ""),
                    "section_path": str(candidate.get("section_path") or ""),
                    "title_text": str(candidate.get("title_text") or ""),
                    "section_path_original": str(candidate.get("section_path_original") or candidate.get("section_path") or ""),
                    "title_text_original": str(candidate.get("title_text_original") or candidate.get("title_text") or ""),
                    "section_path_enriched": str(candidate.get("section_path_enriched") or candidate.get("section_path") or ""),
                    "title_text_enriched": str(candidate.get("title_text_enriched") or candidate.get("title_text") or ""),
                    "heading_text_inferred": str(candidate.get("heading_text_inferred") or ""),
                    "heading_level_inferred": to_int(candidate.get("heading_level_inferred")),
                    "heading_source": str(candidate.get("heading_source") or ""),
                    "section_enrichment_confidence": round(float(candidate.get("section_enrichment_confidence") or 0.0), 4),
                    "column_role": normalize_column_role(candidate.get("column_role")),
                    "clause_text_raw": str(candidate.get("text") or ""),
                    "clause_text_norm": str(candidate.get("normalized_text") or ""),
                    "clause_source_type": str(candidate.get("splitter_type") or "unknown"),
                    "chunk_kind": str(candidate.get("chunk_kind") or ""),
                    "paired_row_idx": to_int_or_none(candidate.get("paired_row_idx")),
                    "sibling_group_id": str(candidate.get("sibling_group_id") or ""),
                    "clause_position_in_group": to_int(candidate.get("clause_position_in_group") or candidate.get("clause_index")),
                    "category_hint": str(candidate.get("category_hint") or ""),
                }
            )

    return pd.DataFrame.from_records(rows)


def build_section_table(canonical_clause_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate canonical clauses into reusable section units."""

    if canonical_clause_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for (doc_id, section_id), group in canonical_clause_df.groupby(["doc_id", "section_id"], sort=False):
        group = group.sort_values(["page_start", "clause_position_in_group", "source_chunk_id"], kind="stable")
        chunk_ids = unique_in_order(group["source_chunk_id"].astype(str).tolist())
        clause_ids = unique_in_order(group["clause_id"].astype(str).tolist())
        rows.append(
            {
                "doc_id": str(doc_id),
                "section_id": str(section_id),
                "section_path": first_non_empty(group["section_path"].tolist()),
                "title_text": first_non_empty(group["title_text"].tolist()),
                "section_path_original": first_existing_column_value(group, "section_path_original", "section_path"),
                "title_text_original": first_existing_column_value(group, "title_text_original", "title_text"),
                "section_path_enriched": first_existing_column_value(group, "section_path_enriched", "section_path"),
                "title_text_enriched": first_existing_column_value(group, "title_text_enriched", "title_text"),
                "heading_text_inferred": first_existing_column_value(group, "heading_text_inferred", "title_text"),
                "heading_source": first_existing_column_value(group, "heading_source", ""),
                "page_start": int(group["page_start"].min()),
                "page_end": int(group["page_end"].max()),
                "section_text_preview": build_text_preview(group["clause_text_raw"].tolist(), max_items=6),
                "chunk_ids": to_json_text(chunk_ids),
                "clause_ids": to_json_text(clause_ids),
                "clause_count": int(len(group)),
                "chunk_count": int(len(chunk_ids)),
            }
        )

    return pd.DataFrame.from_records(rows)


def normalize_doc_ids(chunks_df: pd.DataFrame, *, doc_ids: Iterable[str] | None) -> list[str]:
    """Normalize the optional document-id filter against the chunk dataset."""

    available_doc_ids = [str(value) for value in chunks_df["doc_id"].astype(str).unique().tolist()]
    if doc_ids is None:
        return sorted(available_doc_ids)

    normalized = [str(doc_id) for doc_id in doc_ids]
    return [doc_id for doc_id in normalized if doc_id in available_doc_ids]


def build_text_preview(values: list[str], *, max_items: int) -> str:
    """Build a short preview from top clause texts."""

    preview_items = [value.strip() for value in values if str(value).strip()][:max_items]
    return " ".join(preview_items)


def first_non_empty(values: list[Any]) -> str:
    """Return the first non-empty string in a sequence."""

    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def first_existing_column_value(group: pd.DataFrame, column: str, fallback_column: str) -> str:
    """Return the first non-empty value from a column, falling back to another column."""

    if column in group.columns:
        value = first_non_empty(group[column].tolist())
        if value:
            return value
    if fallback_column and fallback_column in group.columns:
        return first_non_empty(group[fallback_column].tolist())
    return ""


def unique_in_order(values: list[str]) -> list[str]:
    """Return unique values while preserving their first-seen order."""

    return list(dict.fromkeys(value for value in values if value))


def normalize_column_role(value: Any) -> str:
    """Normalize nullable column roles to the canonical values."""

    role = str(value or "").strip()
    return role if role in {"insured", "not_insured", "unknown"} else "unknown"


def to_json_text(value: Any) -> str:
    """Serialize JSON-friendly values for parquet storage."""

    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False)


def to_int(value: Any) -> int:
    """Convert numeric values to plain ints."""

    try:
        if value is None:
            return 0
        return int(value)
    except Exception:
        return 0


def to_int_or_none(value: Any) -> int | None:
    """Convert numeric values to ints while preserving missing values."""

    try:
        if value in {None, ""}:
            return None
        return int(value)
    except Exception:
        return None
