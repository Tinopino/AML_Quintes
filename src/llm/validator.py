"""Validate LLM extraction results against source clauses.

Primary validation: Check that supporting_clause_ids exist and that
exact_quote can be found in the cited clauses.

Fallback: Fuzzy match quotes against all clauses in the section.
"""

from __future__ import annotations

import json
import logging
from difflib import SequenceMatcher
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace for matching."""
    return " ".join(text.lower().split())


def _fuzzy_ratio(a: str, b: str) -> float:
    """Quick ratio between two strings."""
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _substring_match_score(quote: str, source: str) -> float:
    """Check if quote is a substring of source (or vice versa).

    Returns 1.0 for exact substring, partial score for near-match.
    """
    nq = _normalize(quote)
    ns = _normalize(source)

    if nq in ns or ns in nq:
        return 1.0

    # Try finding the best matching window in source
    if len(nq) < 10:
        return 0.0

    # Sliding window
    best = 0.0
    window = len(nq)
    for i in range(max(1, len(ns) - window + 1)):
        chunk = ns[i:i + window]
        ratio = SequenceMatcher(None, nq, chunk).ratio()
        if ratio > best:
            best = ratio
    return best


def validate_extraction_item(
    exact_quote: str,
    supporting_clause_ids: list[str],
    section_clauses: pd.DataFrame,
    fuzzy_threshold: float = 0.80,
) -> dict[str, Any]:
    """Validate one extracted item against source clauses.

    Args:
        exact_quote: Quote from LLM.
        supporting_clause_ids: Clause IDs cited by LLM.
        section_clauses: All clauses for this section (must have clause_id, clause_text_raw).
        fuzzy_threshold: Minimum ratio for fuzzy match acceptance.

    Returns:
        Dict with: validation_status, quote_match_score, matched_clause_ids,
        page_start, page_end, bbox_union.
    """
    if section_clauses.empty or not exact_quote:
        return {
            "validation_status": "unverified",
            "quote_match_score": 0.0,
            "matched_clause_ids": [],
            "page_start": None,
            "page_end": None,
            "bbox_union": None,
        }

    clause_lookup = {
        row["clause_id"]: row
        for _, row in section_clauses.iterrows()
    }

    # Strategy 1: Check cited clause IDs
    cited_texts: list[str] = []
    valid_cited_ids: list[str] = []
    for cid in supporting_clause_ids:
        if cid in clause_lookup:
            valid_cited_ids.append(cid)
            cited_texts.append(str(clause_lookup[cid]["clause_text_raw"]))

    if valid_cited_ids:
        # Concatenate cited clause texts and compare with quote
        combined_cited = " ".join(cited_texts)
        score = max(
            _fuzzy_ratio(exact_quote, combined_cited),
            _substring_match_score(exact_quote, combined_cited),
        )

        if score >= 0.95:
            status = "verified"
        elif score >= fuzzy_threshold:
            status = "fuzzy_match"
        else:
            # Cited IDs exist but quote doesn't match well - try individual
            best_single = 0.0
            for t in cited_texts:
                s = max(_fuzzy_ratio(exact_quote, t), _substring_match_score(exact_quote, t))
                best_single = max(best_single, s)

            if best_single >= fuzzy_threshold:
                score = best_single
                status = "fuzzy_match"
            else:
                fallback_score, fallback_ids = _search_section_clauses(exact_quote, clause_lookup, fuzzy_threshold)
                if fallback_score >= 0.95:
                    return _build_result("verified", fallback_score, fallback_ids, clause_lookup)
                if fallback_score >= fuzzy_threshold:
                    return _build_result("fuzzy_match", fallback_score, fallback_ids, clause_lookup)
                status = "weak_match"

        return _build_result(status, score, valid_cited_ids, clause_lookup)

    # Strategy 2: Cited IDs don't exist - search all section clauses
    best_score, best_clause_ids = _search_section_clauses(exact_quote, clause_lookup, fuzzy_threshold)

    if best_score >= 0.95:
        status = "verified"
    elif best_score >= fuzzy_threshold:
        status = "fuzzy_match"
    else:
        status = "unverified"

    return _build_result(status, best_score, best_clause_ids, clause_lookup)


def _search_section_clauses(
    exact_quote: str,
    clause_lookup: dict[str, Any],
    fuzzy_threshold: float,
) -> tuple[float, list[str]]:
    """Search all clauses in a section for the best quote match."""
    best_score = 0.0
    best_clause_ids: list[str] = []

    for cid, row in clause_lookup.items():
        clause_text = str(row["clause_text_raw"])
        score = max(
            _fuzzy_ratio(exact_quote, clause_text),
            _substring_match_score(exact_quote, clause_text),
        )
        if score > best_score:
            best_score = score
            best_clause_ids = [cid]
        elif score == best_score and score > 0:
            best_clause_ids.append(cid)

    # Also try multi-clause matching (consecutive clauses)
    if best_score < fuzzy_threshold:
        sorted_ids = list(clause_lookup.keys())
        for i in range(len(sorted_ids) - 1):
            combined = " ".join(
                str(clause_lookup[sorted_ids[j]]["clause_text_raw"])
                for j in range(i, min(i + 3, len(sorted_ids)))
            )
            score = max(
                _fuzzy_ratio(exact_quote, combined),
                _substring_match_score(exact_quote, combined),
            )
            if score > best_score:
                best_score = score
                best_clause_ids = sorted_ids[i:min(i + 3, len(sorted_ids))]
    return best_score, best_clause_ids


def _build_result(
    status: str,
    score: float,
    clause_ids: list[str],
    clause_lookup: dict[str, Any],
) -> dict[str, Any]:
    """Build validation result dict with page/bbox info from matched clauses."""
    pages: list[int] = []
    bboxes: list[list[float]] = []

    for cid in clause_ids:
        if cid in clause_lookup:
            row = clause_lookup[cid]
            ps = row.get("page_start")
            pe = row.get("page_end")
            if pd.notna(ps):
                pages.append(int(ps))
            if pd.notna(pe):
                pages.append(int(pe))
            bbox = row.get("bbox")
            if bbox is not None and not (isinstance(bbox, float)):
                if isinstance(bbox, str):
                    try:
                        bbox = json.loads(bbox)
                    except (json.JSONDecodeError, TypeError):
                        bbox = None
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    bboxes.append(list(bbox))

    # Compute bbox union
    bbox_union = None
    if bboxes:
        bbox_union = [
            min(b[0] for b in bboxes),
            min(b[1] for b in bboxes),
            max(b[2] for b in bboxes),
            max(b[3] for b in bboxes),
        ]

    return {
        "validation_status": status,
        "quote_match_score": round(score, 4),
        "matched_clause_ids": clause_ids,
        "page_start": min(pages) if pages else None,
        "page_end": max(pages) if pages else None,
        "bbox_union": bbox_union,
    }


def validate_all_extractions(
    extractions_df: pd.DataFrame,
    clauses_df: pd.DataFrame,
    fuzzy_threshold: float = 0.80,
) -> pd.DataFrame:
    """Validate all extraction items against source clauses.

    Args:
        extractions_df: DataFrame from results_to_dataframe().
        clauses_df: Full canonical_clauses.parquet.
        fuzzy_threshold: Minimum ratio for fuzzy match.

    Returns:
        extractions_df with updated validation columns.
    """
    # Group clauses by (doc_id, section_id)
    clause_groups = clauses_df.groupby(["doc_id", "section_id"])

    validation_updates: list[dict[str, Any]] = []

    for idx, row in extractions_df.iterrows():
        doc_id = row["doc_id"]
        section_id = row["section_id"]
        exact_quote = row.get("exact_quote", "")
        supporting_ids = json.loads(row.get("supporting_clause_ids", "[]"))

        key = (doc_id, section_id)
        if key in clause_groups.groups:
            sec_clauses = clause_groups.get_group(key)
        else:
            sec_clauses = pd.DataFrame()

        result = validate_extraction_item(
            exact_quote=exact_quote,
            supporting_clause_ids=supporting_ids,
            section_clauses=sec_clauses,
            fuzzy_threshold=fuzzy_threshold,
        )

        validation_updates.append({
            "idx": idx,
            **result,
        })

    # Apply updates
    for update in validation_updates:
        idx = update.pop("idx")
        extractions_df.at[idx, "validation_status"] = update["validation_status"]
        extractions_df.at[idx, "quote_match_score"] = update["quote_match_score"]
        extractions_df.at[idx, "matched_clause_ids"] = json.dumps(update["matched_clause_ids"])
        extractions_df.at[idx, "page_start"] = update["page_start"]
        extractions_df.at[idx, "page_end"] = update["page_end"]
        extractions_df.at[idx, "bbox_union"] = json.dumps(update["bbox_union"]) if update["bbox_union"] else None

    # Log stats
    status_counts = extractions_df["validation_status"].value_counts()
    logger.info("Validation results: %s", status_counts.to_dict())

    return extractions_df


def enforce_column_role(
    extractions_df: pd.DataFrame,
    clauses_df: pd.DataFrame,
) -> pd.DataFrame:
    """Override item_type when it contradicts the deterministic column_role.

    Prefer quote-validated matched_clause_ids over LLM-supplied
    supporting_clause_ids, because the model can cite the wrong marker even when
    the exact quote validates to the correct clause.

    Only applies to clauses with column_role in {insured, not_insured}.
    Adds 'column_role_override' column (True where an override occurred).
    """
    if "column_role" not in clauses_df.columns:
        extractions_df["column_role_override"] = False
        return extractions_df

    # Build clause_id -> column_role lookup (only deterministic ones)
    role_map: dict[str, str] = {}
    for _, row in clauses_df.iterrows():
        cr = str(row.get("column_role", "")).strip()
        if cr in ("insured", "not_insured"):
            role_map[row["clause_id"]] = cr

    override_flags: list[bool] = []

    def parse_id_list(value: Any) -> list[str]:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        text = str(value).strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return [str(parsed)]

    for idx, row in extractions_df.iterrows():
        overridden = False
        item_type = row.get("item_type", "")
        matched_ids = parse_id_list(row.get("matched_clause_ids"))
        supporting_ids = parse_id_list(row.get("supporting_clause_ids"))
        source_ids = matched_ids or supporting_ids

        # Determine dominant column_role from quote-validated source clauses.
        roles = [role_map[cid] for cid in source_ids if cid in role_map]
        if not roles:
            override_flags.append(False)
            continue

        # Majority vote
        insured_count = roles.count("insured")
        not_insured_count = roles.count("not_insured")

        if insured_count > not_insured_count:
            expected_type = "covered"
            contradicts = item_type == "not_covered"
        elif not_insured_count > insured_count:
            expected_type = "not_covered"
            contradicts = item_type in ("covered", "condition", "limit")
        else:
            override_flags.append(False)
            continue

        if contradicts:
            logger.warning(
                "Column-role override: item at idx %s changed from '%s' to '%s' "
                "(clause_ids=%s)",
                idx, item_type, expected_type, source_ids,
            )
            extractions_df.at[idx, "item_type"] = expected_type
            overridden = True

        override_flags.append(overridden)

    extractions_df["column_role_override"] = override_flags
    n_overrides = sum(override_flags)
    if n_overrides:
        logger.info("Column-role enforcement: %d items overridden", n_overrides)
    return extractions_df
