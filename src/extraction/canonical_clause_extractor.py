"""Build traceable atomic clauses for the LLM policy-summary pipeline.

This module turns layout-aware chunks into clause-level records with stable
source provenance.  It intentionally keeps only lightweight, prompt-visible
hints such as table column role and broad category hints; final semantic item
typing is assigned by the LLM extraction stage.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from src.chunking.semantic_enrichment import infer_column_role, maybe_json_loads, normalize_space, safe_str


INLINE_BULLET_PATTERN = re.compile(r"\s*[•\u2022\u2023\u25cf\u25e6\u00b7\ufffd]\s*")

SKIP_PATTERNS = (
    "ons kantoor staat",
    "ingeschreven bij de kvk",
    "ingeschreven in het handelsregister",
    "zonder moeilijke woorden",
    "dit noemen wij",
    "zie paragraaf",
    "hieronder lees je",
    "hiermee bedoelen we",
)
LOW_VALUE_PATTERNS = (
    "dit is de waarde",
    "de experts moeten zich houden",
    "om dit te bepalen",
)


def build_clause_candidates_for_doc(
    chunks_df: pd.DataFrame,
    doc_id: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Build chunk and clause candidate views for one document.

    The empty section-view return value is kept for compatibility with older
    notebooks; the active pipeline builds sections from canonical clauses.
    """

    doc_df = chunks_df.loc[chunks_df["doc_id"].astype(str) == str(doc_id)].copy()
    if doc_df.empty:
        raise ValueError(f"No chunks found for doc_id={doc_id!r}.")

    doc_df = doc_df.sort_values(["page_start", "page_end", "chunk_id"], kind="stable")
    doc_df["section_id"] = doc_df.apply(build_section_id, axis=1)
    clause_candidates = build_clause_candidates(doc_df)
    return doc_df, {}, clause_candidates


def build_section_id(row: pd.Series) -> str:
    """Build a stable section id from the normalized section path or title."""

    section_path = normalize_space(row.get("section_path_enriched") or row.get("section_path"))
    if section_path:
        return section_path
    title = normalize_space(row.get("title_text_enriched") or row.get("title_text"))
    page = to_int(row.get("page_start"))
    if title:
        return f"p{page}:{title}"
    return f"p{page}:{safe_str(row.get('chunk_id') or row.get('struct_chunk_id'))}"


def preferred_section_path(row: pd.Series | dict[str, Any]) -> str:
    """Return enriched section path when present, otherwise the original path."""

    return safe_str(row.get("section_path_enriched") or row.get("section_path"))


def preferred_title_text(row: pd.Series | dict[str, Any]) -> str:
    """Return enriched title text when present, otherwise the original title."""

    return safe_str(row.get("title_text_enriched") or row.get("title_text"))


def build_clause_candidates(doc_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Create atomic clause candidates from layout-aware chunks."""

    candidates: list[dict[str, Any]] = []
    for row in doc_df.to_dict(orient="records"):
        chunk_id = safe_str(row.get("chunk_id") or row.get("struct_chunk_id"))
        fragments = split_chunk_into_clauses(row)
        for clause_index, fragment in enumerate(fragments):
            text = clean_clause_text(fragment.get("text", ""))
            if not text or should_skip_clause(text):
                continue
            paired_row_idx = to_int_or_none(fragment.get("paired_row_idx"))
            sibling_group_id = build_sibling_group_id(chunk_id, paired_row_idx)
            candidates.append(
                {
                    "doc_id": safe_str(row.get("doc_id")),
                    "chunk_id": chunk_id,
                    "section_id": safe_str(row.get("section_id")),
                    "page_start": to_int(row.get("page_start")),
                    "page_end": to_int(row.get("page_end")),
                    "bbox": row.get("bbox") if isinstance(row.get("bbox"), list) else maybe_json_loads(row.get("bbox")),
                    "section_path": preferred_section_path(row),
                    "title_text": preferred_title_text(row),
                    "section_path_original": safe_str(row.get("section_path_original") or row.get("section_path")),
                    "title_text_original": safe_str(row.get("title_text_original") or row.get("title_text")),
                    "section_path_enriched": safe_str(row.get("section_path_enriched") or row.get("section_path")),
                    "title_text_enriched": safe_str(row.get("title_text_enriched") or row.get("title_text")),
                    "heading_text_inferred": safe_str(row.get("heading_text_inferred")),
                    "heading_level_inferred": to_int(row.get("heading_level_inferred")),
                    "heading_source": safe_str(row.get("heading_source")),
                    "section_enrichment_confidence": float(row.get("section_enrichment_confidence") or 0.0),
                    "text": text,
                    "normalized_text": normalize_clause_text(text),
                    "column_role": fragment.get("column_role") or infer_column_role(row),
                    "paired_row_idx": paired_row_idx,
                    "sibling_group_id": sibling_group_id,
                    "clause_index": clause_index,
                    "clause_position_in_group": clause_index,
                    "category_hint": fragment.get("category_hint"),
                    "splitter_type": infer_splitter_type(row),
                    "chunk_kind": safe_str(row.get("chunk_kind")),
                }
            )
    return candidates


def infer_splitter_type(row: dict[str, Any]) -> str:
    """Expose which splitter produced a clause candidate."""

    kind = safe_str(row.get("chunk_kind"))
    if kind == "table_paired_chunk":
        return "table_paired"
    if kind == "table_matrix_chunk":
        return "table_matrix"
    if kind == "list_chunk":
        return "list"
    if kind == "paragraph_chunk":
        return "paragraph"
    return "single_text"


def split_chunk_into_clauses(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a chunk into atomic, traceable clause fragments."""

    kind = safe_str(row.get("chunk_kind"))
    if kind == "table_paired_chunk":
        return split_table_paired_chunk(row)
    if kind == "table_matrix_chunk":
        return split_table_matrix_chunk(row)
    if kind == "list_chunk":
        return split_list_chunk(row)
    return split_text_chunk(row)


def split_table_paired_chunk(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Split paired insured/not-insured tables into atomic clause fragments."""

    paired_rows = maybe_json_loads(row.get("table_rows_json"))
    if not isinstance(paired_rows, list):
        return split_text_chunk(row)

    fragments: list[dict[str, Any]] = []
    for row_obj in paired_rows:
        if not isinstance(row_obj, dict):
            continue

        paired_row_idx = to_int(row_obj.get("row_idx"))
        for column_role, key in (("insured", "insured"), ("not_insured", "not_insured")):
            value = clean_clause_text(row_obj.get(key, ""))
            if not value:
                continue
            category_hint = "not_covered" if column_role == "not_insured" else None
            for text in explode_clause_text(value):
                fragments.append(
                    {
                        "text": text,
                        "column_role": column_role,
                        "paired_row_idx": paired_row_idx,
                        "category_hint": category_hint,
                    }
                )
    return fragments


def split_table_matrix_chunk(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Split table matrices into clause fragments.

    The processed dataset already exposes table rows as JSON. For the current
    policy structure we only need a small amount of special handling for the two
    important matrix types on pages 8 and 9.
    """

    table_rows = maybe_json_loads(row.get("table_rows_json"))
    text = normalize_space(row.get("text_raw")).lower()
    if not isinstance(table_rows, list):
        return split_text_chunk(row)

    if "samenvatting waardebepaling" in text:
        return split_value_summary_matrix(table_rows)
    if "reparatie door:" in text or "vrije reparatiekeuze" in text:
        return split_repair_matrix(table_rows)

    return split_text_chunk(row)


def split_repair_matrix(table_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split the repair-network matrix into general rule clauses."""

    fragments: list[dict[str, Any]] = []
    row_texts = [flatten_matrix_row(row_obj) for row_obj in table_rows if flatten_matrix_row(row_obj)]
    for text in row_texts:
        lowered = text.lower()
        if "100% vergoeding" in lowered and "eigen bijdrage" not in lowered and "geen vergoeding" not in lowered:
            continue
        for part in explode_clause_text(text):
            hint = "not_covered" if "geen vergoeding" in part.lower() else ("condition" if "eigen bijdrage" in part.lower() else None)
            fragments.append({"text": part, "column_role": "unknown", "category_hint": hint})
    return fragments


def split_value_summary_matrix(table_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split the valuing-summary matrix into atomic limit/condition clauses."""

    fragments: list[dict[str, Any]] = []
    parsed_rows = [extract_cell_texts(row_obj) for row_obj in table_rows]
    if len(parsed_rows) < 3:
        return fragments

    for row_cells in parsed_rows[1:]:
        if len(row_cells) < 3:
            continue
        age_rule, first_year_rule, after_first_year_rule = row_cells[0], row_cells[1], row_cells[2]
        fragments.append({
            "text": f"{age_rule}. In het eerste jaar van de verzekering geldt {first_year_rule}.",
            "column_role": "unknown",
            "category_hint": "condition",
        })
        fragments.append({
            "text": f"{age_rule}. Na het eerste jaar geldt {after_first_year_rule}.",
            "column_role": "unknown",
            "category_hint": "limit",
        })
    return fragments


def split_list_chunk(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a list chunk into atomic clause fragments."""

    text = safe_str(row.get("text_raw"))
    column_role = infer_column_role(row)
    title = normalize_space(row.get("title_text")).lower()
    fragments: list[dict[str, Any]] = []
    for bullet in split_bullet_lines(text):
        category_hint = "not_covered" if column_role == "not_insured" or "nooit verzekerd" in title else None
        if title.startswith("verzekerd zijn"):
            category_hint = "covered"
        for part in explode_clause_text(bullet):
            fragments.append({"text": part, "column_role": column_role, "category_hint": category_hint})
    return fragments


def split_text_chunk(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Split paragraph and single-text chunks into atomic clause fragments."""

    text = clean_clause_text(row.get("text_raw", ""))
    column_role = infer_column_role(row)
    title = normalize_space(row.get("title_text")).lower()
    if not text:
        return []

    fragments: list[dict[str, Any]] = []
    for segment in split_marker_sections(text):
        hint = None
        lowered = segment.lower()
        if lowered.startswith("maximale vergoeding") or lowered.startswith("maximale vergoedingen"):
            hint = "limit"
        elif lowered.startswith("bijzonderheden"):
            hint = "condition"
        elif "dit is nooit verzekerd" in title:
            hint = "not_covered"
        for part in explode_clause_text(segment):
            fragments.append({"text": part, "column_role": column_role, "category_hint": hint})
    return fragments


def split_marker_sections(text: str) -> list[str]:
    """Split dense chunks around explicit semantic markers."""

    value = clean_clause_text(text)
    if not value:
        return []

    value = re.sub(r"\s+(?=(?:Maximale vergoeding:|Maximale vergoedingen:|Bijzonderheden:|Let op!))", "\n", value)
    return [segment.strip() for segment in value.split("\n") if segment.strip()]


def split_bullet_lines(text: str) -> list[str]:
    """Split newline bullets while keeping single-line chunks intact."""

    value = safe_str(text).replace("\r", "")
    if "\n" not in value:
        return [clean_clause_text(value)] if clean_clause_text(value) else []

    parts = []
    for raw_line in value.split("\n"):
        line = clean_clause_text(raw_line)
        if line:
            parts.append(line)
    return parts


def explode_clause_text(text: str) -> list[str]:
    """Explode one chunk fragment into atomic clauses."""

    value = clean_clause_text(text)
    if not value:
        return []

    parts = split_inline_bullets(value)
    atomic: list[str] = []
    for part in parts:
        atomic.extend(split_sentences(part))
    return [clean_clause_text(part) for part in atomic if clean_clause_text(part)]


def split_inline_bullets(text: str) -> list[str]:
    """Split inline bullet syntax, preserving useful prefixes."""

    value = clean_clause_text(text)
    if not value:
        return []
    if not INLINE_BULLET_PATTERN.search(value):
        return [value]

    first_match = INLINE_BULLET_PATTERN.search(value)
    if first_match is None:
        return [value]

    prefix = value[: first_match.start()].strip()
    items = [clean_clause_text(part) for part in INLINE_BULLET_PATTERN.split(value[first_match.start() :]) if clean_clause_text(part)]
    if not prefix:
        return items

    lowered_prefix = prefix.lower().rstrip(":")
    propagated = []
    for item in items:
        if lowered_prefix.endswith("door") or lowered_prefix.endswith("bij") or lowered_prefix.endswith("over"):
            propagated.append(clean_clause_text(f"{prefix} {item}"))
        else:
            propagated.append(clean_clause_text(f"{prefix} {item}"))
    return propagated


def split_sentences(text: str) -> list[str]:
    """Split sentences while keeping dependent follow-ups attached."""

    value = clean_clause_text(text)
    if not value:
        return []

    placeholders = {
        "max.": "max<dot>",
        "p.p.": "pp<dot>",
        "btw.": "btw<dot>",
        "bovag.": "bovag<dot>",
    }
    masked = value
    for token, placeholder in placeholders.items():
        masked = masked.replace(token, placeholder)
    masked = masked.replace("? Dan ", "?<keep>Dan ")

    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-Ý0-9])", masked) if segment.strip()]
    restored = []
    for sentence in sentences:
        for token, placeholder in placeholders.items():
            sentence = sentence.replace(placeholder, token)
        restored.append(sentence.replace("?<keep>", "? "))
    return merge_dependent_sentences(restored)


def merge_dependent_sentences(sentences: list[str]) -> list[str]:
    """Merge follow-up sentences that depend on the previous sentence."""

    merged: list[str] = []
    for sentence in sentences:
        current = clean_clause_text(sentence)
        if not current:
            continue
        lowered = current.lower()
        if merged and (
            merged[-1].endswith("?")
            or lowered.startswith(("bijvoorbeeld", "dit betekent", "om dit te bepalen", "maar ", "en ", "of ", "dan "))
        ):
            merged[-1] = clean_clause_text(f"{merged[-1]} {current}")
        else:
            merged.append(current)
    return merged


def should_skip_clause(text: str) -> bool:
    """Drop non-policy or low-value narrative clauses."""

    lowered = clean_clause_text(text).lower()
    if not lowered:
        return True
    if re.fullmatch(r"\d+[\.:]?", lowered) is not None:
        return True
    if any(pattern in lowered for pattern in SKIP_PATTERNS):
        return True
    if lowered in {"1. betaling bij schade", "4. de waarde van je auto", "maximale vergoeding:", "bijzonderheden:"}:
        return True
    if any(pattern in lowered for pattern in LOW_VALUE_PATTERNS):
        return True
    if lowered.startswith("met schade bedoelen we"):
        return True
    if "heeft haar kantoor in rotterdam" in lowered or "ingeschreven bij de kvk" in lowered:
        return True
    return False


def build_item_id(*, doc_id: str, chunk_id: str, paired_row_idx: int | None, clause_index: int) -> str:
    """Build a stable item id from source chunk coordinates."""

    row_part = f"r{paired_row_idx}" if paired_row_idx is not None else "rna"
    return f"{doc_id}:{chunk_id}:{row_part}:c{clause_index}"


def normalize_clause_text(text: str) -> str:
    """Normalize clause text for matching and deduplication."""

    value = clean_clause_text(text).lower()
    value = value.replace("�", "€")
    value = re.sub(r"\s+", " ", value)
    value = value.rstrip(".:;,")
    return value


def clean_clause_text(value: Any) -> str:
    """Normalize whitespace and trim list markers."""

    text = normalize_space(value)
    text = text.replace("�", "€")
    text = re.sub(r"^[\-•\u2022\u2023\u25cf\u25e6\ufffd\s]+", "", text)
    text = re.sub(r"^(Bijzonderheden:|1\. Dit is verzekerd 2\. Dit is niet verzekerd\s+)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def flatten_matrix_row(row_obj: dict[str, Any]) -> str:
    """Flatten one matrix row into readable text."""

    parts = extract_cell_texts(row_obj)
    return clean_clause_text(" ".join(parts))


def extract_cell_texts(row_obj: dict[str, Any]) -> list[str]:
    """Return non-empty text from row cells."""

    cells = row_obj.get("cells")
    if not isinstance(cells, list):
        return []
    values = [clean_clause_text(cell.get("text", "")) for cell in cells if isinstance(cell, dict)]
    return [value for value in values if value]


def build_sibling_group_id(chunk_id: str, paired_row_idx: int | None) -> str:
    """Build a stable sibling-group id for clause candidates from the same source unit."""

    if paired_row_idx is not None:
        return f"{chunk_id}:r{paired_row_idx}"
    return f"{chunk_id}:group"


def to_int_or_none(value: Any) -> int | None:
    """Convert numeric values to ints, preserving missing values as None."""

    try:
        if value is None:
            return None
        if isinstance(value, float) and pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def to_int(value: Any) -> int:
    """Convert numbers to ints, defaulting to 0 when unavailable."""

    try:
        if value is None:
            return 0
        if isinstance(value, float) and pd.isna(value):
            return 0
        return int(value)
    except Exception:
        return 0
