"""Build full section text with clause markers for LLM extraction.

This module reconstructs complete section text from atomic clauses,
adding clause ID markers that allow the LLM to cite specific sources.

The output format enables:
1. LLM to read full section context for better comprehension
2. LLM to cite specific clause IDs in its extractions
3. Direct mapping from LLM output back to source clauses (bbox, page, etc.)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class SectionTextConfig:
    """Configuration for section text building."""

    marker_template: str = "[clause_id={clause_id}]"
    clause_separator: str = "\n\n"
    max_words_per_window: int = 2500
    window_overlap_clauses: int = 2


def _build_one_section(
    section_clauses: pd.DataFrame,
    config: SectionTextConfig,
) -> dict[str, Any]:
    """Build full text for a single section from its clauses.

    Args:
        section_clauses: Pre-filtered DataFrame of clauses for one section.
        config: Marker / separator settings.

    Returns:
        Dict with section_text_full, section_text_with_markers,
        clause_ids, n_words, n_clauses.
    """
    if section_clauses.empty:
        return {
            "section_text_full": "",
            "section_text_with_markers": "",
            "clause_ids": [],
            "n_words": 0,
            "n_clauses": 0,
        }

    # Deterministic clause order
    if "clause_position_in_group" in section_clauses.columns:
        section_clauses = section_clauses.sort_values("clause_position_in_group")
    else:
        section_clauses = section_clauses.sort_values("clause_id")

    text_parts_plain: list[str] = []
    text_parts_marked: list[str] = []
    clause_ids: list[str] = []

    for _, row in section_clauses.iterrows():
        text = str(row["clause_text_raw"]).strip() if pd.notna(row["clause_text_raw"]) else ""
        if not text:
            continue
        clause_id = row["clause_id"]
        clause_ids.append(clause_id)
        text_parts_plain.append(text)
        marker = config.marker_template.format(clause_id=clause_id)
        # Add deterministic column_role tag when available
        col_role = str(row.get("column_role", "")).strip() if "column_role" in section_clauses.columns else ""
        if col_role in ("insured", "not_insured"):
            marker += f" [column={col_role}]"
        # Add category_hint tag when deterministic
        cat_hint = str(row.get("category_hint", "")).strip() if "category_hint" in section_clauses.columns else ""
        if cat_hint in ("not_covered", "covered", "condition", "limit"):
            marker += f" [hint={cat_hint}]"
        text_parts_marked.append(f"{marker}\n{text}")

    full = config.clause_separator.join(text_parts_plain)
    marked = config.clause_separator.join(text_parts_marked)

    return {
        "section_text_full": full,
        "section_text_with_markers": marked,
        "clause_ids": clause_ids,
        "n_words": len(full.split()),
        "n_clauses": len(clause_ids),
    }


def enrich_sections_with_full_text(
    sections_df: pd.DataFrame,
    clauses_df: pd.DataFrame,
    config: SectionTextConfig | None = None,
) -> pd.DataFrame:
    """Add full text columns to sections DataFrame.

    Joins on (doc_id, section_id) to avoid cross-document collisions.

    Returns the original sections with added columns:
        section_text_full, section_text_with_markers,
        clause_ids_ordered (JSON), n_words, n_clauses.
    """
    if config is None:
        config = SectionTextConfig()

    # Group clauses by (doc_id, section_id) for fast lookup
    grouped = clauses_df.groupby(["doc_id", "section_id"])

    rows: list[dict[str, Any]] = []
    for _, sec_row in sections_df.iterrows():
        key = (sec_row["doc_id"], sec_row["section_id"])
        if key in grouped.groups:
            sec_clauses = grouped.get_group(key)
        else:
            sec_clauses = pd.DataFrame()

        info = _build_one_section(sec_clauses, config)
        rows.append({
            "doc_id": sec_row["doc_id"],
            "section_id": sec_row["section_id"],
            "section_text_full": info["section_text_full"],
            "section_text_with_markers": info["section_text_with_markers"],
            "clause_ids_ordered": json.dumps(info["clause_ids"]),
            "n_words": info["n_words"],
            "n_clauses_full": info["n_clauses"],
        })

    enrichment = pd.DataFrame(rows)

    # Drop columns we are about to add (avoids _x / _y suffixes)
    new_cols = ["section_text_full", "section_text_with_markers",
                "clause_ids_ordered", "n_words", "n_clauses_full"]
    drop_existing = [c for c in new_cols if c in sections_df.columns]
    if drop_existing:
        sections_df = sections_df.drop(columns=drop_existing)

    enriched = sections_df.merge(enrichment, on=["doc_id", "section_id"], how="left")
    return enriched


# ── Context window splitting ─────────────────────────────────────────────

def split_section_into_windows(
    section_text_with_markers: str,
    clause_ids: list[str],
    config: SectionTextConfig,
) -> list[dict[str, Any]]:
    """Split a section into overlapping windows when it exceeds max_words."""
    n_words = len(section_text_with_markers.split())

    if n_words <= config.max_words_per_window:
        return [{
            "window_index": 0,
            "window_text": section_text_with_markers,
            "clause_ids": clause_ids,
            "is_continuation": False,
        }]

    # Split by clause marker boundaries
    raw_parts = section_text_with_markers.split("[clause_id=")
    clause_parts: list[str] = []
    for i, part in enumerate(raw_parts):
        if i == 0 and not part.strip():
            continue
        if part.strip():
            clause_parts.append("[clause_id=" + part.rstrip())

    # Fallback if marker parsing doesn't match clause count
    if len(clause_parts) != len(clause_ids):
        return [{
            "window_index": 0,
            "window_text": section_text_with_markers,
            "clause_ids": clause_ids,
            "is_continuation": False,
        }]

    windows: list[dict[str, Any]] = []
    start = 0
    idx = 0

    while start < len(clause_parts):
        end = start
        while end < len(clause_parts):
            text = "\n\n".join(clause_parts[start:end + 1])
            if len(text.split()) > config.max_words_per_window and end > start:
                break
            end += 1

        window_text = "\n\n".join(clause_parts[start:end])
        windows.append({
            "window_index": idx,
            "window_text": window_text,
            "clause_ids": clause_ids[start:end],
            "is_continuation": idx > 0,
        })

        start = max(start + 1, end - config.window_overlap_clauses)
        idx += 1
        if idx > 200:  # safety
            break

    return windows


def build_llm_context_windows(
    sections_df: pd.DataFrame,
    clauses_df: pd.DataFrame,
    config: SectionTextConfig | None = None,
) -> pd.DataFrame:
    """Build LLM context windows from sections.

    Returns DataFrame with: context_id, doc_id, section_id, section_path,
    window_index, context_text, clause_ids (JSON), n_words, is_continuation.
    """
    if config is None:
        config = SectionTextConfig()

    enriched = enrich_sections_with_full_text(sections_df, clauses_df, config)

    contexts: list[dict[str, Any]] = []

    for _, row in enriched.iterrows():
        text = row.get("section_text_with_markers", "")
        clause_ids_json = row.get("clause_ids_ordered", "[]")
        clause_ids = json.loads(clause_ids_json) if clause_ids_json else []

        if not text or not clause_ids:
            continue

        windows = split_section_into_windows(text, clause_ids, config)

        for w in windows:
            ctx_id = f"{row['doc_id']}__{row['section_id']}__w{w['window_index']}"
            contexts.append({
                "context_id": ctx_id,
                "doc_id": row["doc_id"],
                "section_id": row["section_id"],
                "section_path": row["section_path"],
                "window_index": w["window_index"],
                "context_text": w["window_text"],
                "clause_ids": json.dumps(w["clause_ids"]),
                "n_words": len(w["window_text"].split()),
                "is_continuation": w["is_continuation"],
            })

    return pd.DataFrame(contexts)
