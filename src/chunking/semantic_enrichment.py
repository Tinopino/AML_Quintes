"""Small shared helpers for canonical clause construction.

Only prompt-visible table polarity support and generic JSON/string utilities are
kept here.  Legacy deterministic module inference and retrieval enrichment were
removed from the production LLM pipeline because final semantic typing belongs
to the LLM extraction stage.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd


GENERIC_PAIRED_TITLE_PATTERNS = (
    "1. dit is verzekerd 2. dit is niet verzekerd",
    "dit is verzekerd",
    "dit is niet verzekerd",
)


def infer_column_role(row: dict[str, Any]) -> str:
    """Infer whether a chunk belongs to the insured or not-insured table side.

    This is intentionally a narrow structural helper.  It only creates the
    prompt-visible ``[column=insured]`` / ``[column=not_insured]`` markers and
    supports validator column-role enforcement; it is not a customer-facing
    semantic classifier.
    """

    title = normalize_space(row.get("title_text")).lower()
    section = normalize_space(row.get("section_path")).lower()
    text = normalize_space(row.get("text_raw")).lower()
    bbox = row.get("bbox")

    if safe_str(row.get("chunk_kind")) == "table_paired_row":
        return safe_str(row.get("column_role")) or "unknown"

    if "dit is verzekerd" in title and "dit is niet verzekerd" not in title:
        return "insured"
    if "dit is niet verzekerd" in title and "dit is verzekerd" not in title:
        return "not_insured"
    if "dit is verzekerd" in section and "dit is niet verzekerd" not in section:
        return "insured"
    if "dit is niet verzekerd" in section and "dit is verzekerd" not in section:
        return "not_insured"

    is_generic_paired = any(pattern in title for pattern in GENERIC_PAIRED_TITLE_PATTERNS) or any(
        pattern in section for pattern in GENERIC_PAIRED_TITLE_PATTERNS
    )
    if is_generic_paired:
        center_x = bbox_center_x(bbox)
        if center_x is not None:
            return "insured" if center_x < 300 else "not_insured"

    if text.startswith("als ") or text.startswith("- als ") or text.startswith("• als "):
        center_x = bbox_center_x(bbox)
        if center_x is not None and center_x >= 300:
            return "not_insured"

    return "unknown"


def bbox_center_x(bbox: Any) -> float | None:
    """Return the horizontal center of a bbox if available."""

    bbox_obj = maybe_json_loads(bbox)
    if not isinstance(bbox_obj, list) or len(bbox_obj) != 4:
        return None
    try:
        return (float(bbox_obj[0]) + float(bbox_obj[2])) / 2.0
    except Exception:
        return None


def maybe_json_loads(value: Any) -> Any:
    """Load JSON-like values when possible."""

    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def normalize_space(value: Any) -> str:
    """Normalize whitespace for readable matching."""

    text = safe_str(value).replace("\u00ad", "")
    return re.sub(r"\s+", " ", text).strip()


def safe_str(value: Any) -> str:
    """Convert nullable values to plain strings."""

    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)
