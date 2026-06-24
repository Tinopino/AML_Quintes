"""Build the exact Marker-derived chunk dataset used by the LLM pipeline.

This module ports the reusable logic from ``preprocessingaml.py``.  The goal is
to reproduce ``structure_chunks_enriched.parquet`` rather than use a simplified
line-level PDF parser.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


DROP_BLOCK_TYPES = {
    "Page",
    "PageHeader",
    "PageFooter",
    "Picture",
}


def html_to_text(html: Optional[str]) -> str:
    """Convert Marker block HTML to readable text."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str) -> str:
    if not text:
        return ""
    value = text.replace("\u00ad", "")
    return re.sub(r"\s+", " ", value).strip()


def normalize_space(text: Any) -> str:
    if pd.isna(text):
        return ""
    value = str(text).replace("\u00ad", "")
    return re.sub(r"\s+", " ", value).strip()


def flatten_block_tree(
    block: Dict[str, Any],
    doc_id: str,
    page_no: int,
    parent_id: Optional[str] = None,
    depth: int = 0,
) -> List[Dict[str, Any]]:
    """Flatten one Marker block and all descendants into rows."""
    rows = []
    block_id = block.get("id", "")
    block_type = block.get("block_type", "")
    html = block.get("html", "")
    bbox = block.get("bbox", None)
    polygon = block.get("polygon", None)
    section_hierarchy = block.get("section_hierarchy", None)
    children = block.get("children", None)
    text_raw = normalize_text(html_to_text(html))

    rows.append({
        "doc_id": doc_id,
        "page_no": page_no,
        "block_id": block_id,
        "parent_id": parent_id,
        "depth": depth,
        "block_type": block_type,
        "html": html,
        "text_raw": text_raw,
        "text_len": len(text_raw),
        "bbox": json.dumps(bbox, ensure_ascii=False) if bbox is not None else None,
        "polygon": json.dumps(polygon, ensure_ascii=False) if polygon is not None else None,
        "section_hierarchy": json.dumps(section_hierarchy, ensure_ascii=False) if section_hierarchy is not None else None,
        "n_children": len(children) if isinstance(children, list) else 0,
    })

    if isinstance(children, list):
        for child in children:
            rows.extend(
                flatten_block_tree(
                    block=child,
                    doc_id=doc_id,
                    page_no=page_no,
                    parent_id=block_id,
                    depth=depth + 1,
                )
            )
    return rows


def get_pages_from_marker_json(obj: Any) -> List[Dict[str, Any]]:
    """Return page blocks from Marker JSON."""
    if isinstance(obj, dict) and "children" in obj and isinstance(obj["children"], list):
        return obj["children"]
    if isinstance(obj, list):
        return obj
    return []


def flatten_marker_json_directory(
    json_dir: Path,
    *,
    doc_ids: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Flatten Marker JSON files and build useful block/page/document tables."""
    wanted = {str(doc_id) for doc_id in doc_ids} if doc_ids else None
    json_paths = sorted(
        path for path in json_dir.glob("*.json")
        if wanted is None or path.stem in wanted
    )
    if not json_paths:
        raise FileNotFoundError(f"No matching Marker JSON files found in {json_dir}")

    all_rows = []
    run_log = []
    page_rows = []
    document_rows = []

    for path in json_paths:
        doc_id = path.stem
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            pages = get_pages_from_marker_json(obj)
            for page_idx, page in enumerate(pages, start=1):
                all_rows.extend(
                    flatten_block_tree(
                        block=page,
                        doc_id=doc_id,
                        page_no=page_idx,
                        parent_id=None,
                        depth=0,
                    )
                )
                page_rows.append({
                    "doc_id": doc_id,
                    "page_no": page_idx,
                })
            run_log.append({"doc_id": doc_id, "ok": True, "notes": "", "n_pages": len(pages)})
            document_rows.append({"doc_id": doc_id, "file_name": f"{doc_id}.pdf", "n_pages": len(pages)})
        except Exception as exc:
            run_log.append({"doc_id": doc_id, "ok": False, "notes": repr(exc), "n_pages": 0})

    marker_blocks_df = pd.DataFrame(all_rows)
    run_log_df = pd.DataFrame(run_log)
    documents_df = pd.DataFrame(document_rows)
    pages_df = pd.DataFrame(page_rows)

    if marker_blocks_df.empty:
        return documents_df, pages_df, marker_blocks_df, pd.DataFrame(), run_log_df

    header_map = (
        marker_blocks_df.loc[
            marker_blocks_df["block_type"] == "SectionHeader",
            ["doc_id", "block_id", "text_raw"],
        ]
        .drop_duplicates()
        .set_index(["doc_id", "block_id"])["text_raw"]
        .to_dict()
    )

    marker_blocks_df["section_path"] = marker_blocks_df.apply(
        lambda row: decode_section_path(row["doc_id"], row["section_hierarchy"], header_map),
        axis=1,
    )

    marker_blocks_useful_df = marker_blocks_df[
        (~marker_blocks_df["block_type"].isin(DROP_BLOCK_TYPES))
        & (marker_blocks_df["text_len"] > 0)
    ].copy()
    marker_blocks_useful_df = marker_blocks_useful_df.sort_values(
        ["doc_id", "page_no", "depth", "block_id"]
    ).reset_index(drop=True)

    return documents_df, pages_df, marker_blocks_df, marker_blocks_useful_df, run_log_df


def decode_section_path(doc_id: str, section_hierarchy_json: Optional[str], header_map: dict[tuple[str, str], str]) -> str:
    if not section_hierarchy_json:
        return ""
    try:
        hierarchy = json.loads(section_hierarchy_json)
        if not isinstance(hierarchy, dict):
            return ""
        parts = []
        for level in sorted(hierarchy.keys(), key=lambda value: int(value)):
            block_id = hierarchy[level]
            text = header_map.get((doc_id, block_id), "")
            if text:
                parts.append(text)
        return " > ".join(parts)
    except Exception:
        return ""


def parse_json_maybe(value: Any):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return None


def bbox_union(bboxes):
    bboxes = [bbox for bbox in bboxes if isinstance(bbox, list) and len(bbox) == 4]
    if not bboxes:
        return None
    return [
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    ]


def last_section_title(section_path):
    value = normalize_space(section_path)
    if not value:
        return ""
    return value.split(" > ")[-1].strip()


def textlike_merge_ok(curr_row, next_row, y_gap_max=35, x_tol=40):
    if curr_row["block_type"] != "Text" or next_row["block_type"] != "Text":
        return False
    if curr_row["section_path"] != next_row["section_path"]:
        return False
    if curr_row["page_no"] != next_row["page_no"]:
        return False
    b1 = curr_row["bbox_obj"]
    b2 = next_row["bbox_obj"]
    if b1 is None or b2 is None:
        return False
    y_gap = b2[1] - b1[3]
    x_diff = abs(b1[0] - b2[0])
    if y_gap < -5:
        return False
    return (y_gap <= y_gap_max) and (x_diff <= x_tol)


def list_merge_ok(curr_row, next_row, y_gap_max=28, x_tol=50):
    if curr_row["block_type"] != "ListItem" or next_row["block_type"] != "ListItem":
        return False
    if curr_row["section_path"] != next_row["section_path"]:
        return False
    if curr_row["page_no"] != next_row["page_no"]:
        return False
    b1 = curr_row["bbox_obj"]
    b2 = next_row["bbox_obj"]
    if b1 is None or b2 is None:
        return False
    y_gap = b2[1] - b1[3]
    x_diff = abs(b1[0] - b2[0])
    same_parent = normalize_space(curr_row["parent_id"]) == normalize_space(next_row["parent_id"])
    if y_gap < -5:
        return False
    return same_parent or ((y_gap <= y_gap_max) and (x_diff <= x_tol))


def parse_table_html_structure(html):
    """Parse Marker HTML table into structured rows."""
    html = normalize_space(html)
    if not html:
        return "", "", "", 0, ""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        return "", "", "", 0, html

    parsed_rows = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"], recursive=False)
        if not cells:
            cells = tr.find_all(["th", "td"])
        row_cells = []
        for cell in cells:
            row_cells.append({
                "tag": cell.name,
                "text": normalize_space(cell.get_text(" ", strip=True)),
                "colspan": int(cell.get("colspan", 1)),
                "rowspan": int(cell.get("rowspan", 1)),
            })
        if row_cells:
            parsed_rows.append(row_cells)

    if not parsed_rows:
        return "", "", "", 0, html

    all_texts = [
        normalize_space(cell["text"]).lower()
        for row in parsed_rows
        for cell in row
        if normalize_space(cell["text"])
    ]
    has_left = any(text == "1. dit is verzekerd" for text in all_texts)
    has_right = any(text == "2. dit is niet verzekerd" for text in all_texts)
    has_combined = any("1. dit is verzekerd 2. dit is niet verzekerd" in text for text in all_texts)

    if has_combined or (has_left and has_right):
        table_kind = "table_paired_chunk"
    elif any(cell["colspan"] > 1 or cell["rowspan"] > 1 for row in parsed_rows for cell in row) or len(parsed_rows) >= 3:
        table_kind = "table_matrix_chunk"
    else:
        table_kind = "table_other_chunk"

    if table_kind == "table_paired_chunk":
        headers = []
        body_rows = []
        for row in parsed_rows:
            texts = [normalize_space(cell["text"]) for cell in row]
            texts_low = [text.lower() for text in texts]
            if any("dit is verzekerd" in text or "dit is niet verzekerd" in text for text in texts_low):
                headers = texts
            else:
                body_rows.append(row)
        rows_out = []
        for row_idx, row in enumerate(body_rows):
            insured = normalize_space(row[0]["text"]) if len(row) >= 1 else ""
            not_insured = normalize_space(row[1]["text"]) if len(row) >= 2 else ""
            rows_out.append({"row_idx": row_idx, "insured": insured, "not_insured": not_insured})
        return table_kind, json.dumps(headers, ensure_ascii=False), json.dumps(rows_out, ensure_ascii=False), len(rows_out), str(table)

    if table_kind == "table_matrix_chunk":
        header_rows = []
        rows_out = []
        for row_idx, row in enumerate(parsed_rows):
            rows_out.append({"row_idx": row_idx, "cells": row})
            if any(cell["tag"] == "th" for cell in row):
                header_rows.append([normalize_space(cell["text"]) for cell in row])
        return table_kind, json.dumps(header_rows, ensure_ascii=False), json.dumps(rows_out, ensure_ascii=False), len(rows_out), str(table)

    rows_out = []
    header_rows = []
    for row_idx, row in enumerate(parsed_rows):
        rows_out.append({"row_idx": row_idx, "cells": row})
        if any(cell["tag"] == "th" for cell in row):
            header_rows.append([normalize_space(cell["text"]) for cell in row])
    return table_kind, json.dumps(header_rows, ensure_ascii=False), json.dumps(rows_out, ensure_ascii=False), len(rows_out), str(table)


def build_structure_chunks(marker_blocks_useful_df: pd.DataFrame) -> pd.DataFrame:
    """Build exact ``structure_chunks_enriched`` rows from useful Marker blocks."""
    blocks = marker_blocks_useful_df.copy()
    blocks["text_raw"] = blocks["text_raw"].map(normalize_space)
    blocks["section_path"] = blocks["section_path"].map(normalize_space)
    blocks["bbox_obj"] = blocks["bbox"].map(parse_json_maybe)
    blocks["x0"] = blocks["bbox_obj"].map(lambda b: b[0] if isinstance(b, list) and len(b) == 4 else np.nan)
    blocks["y0"] = blocks["bbox_obj"].map(lambda b: b[1] if isinstance(b, list) and len(b) == 4 else np.nan)
    blocks["x1"] = blocks["bbox_obj"].map(lambda b: b[2] if isinstance(b, list) and len(b) == 4 else np.nan)
    blocks["y1"] = blocks["bbox_obj"].map(lambda b: b[3] if isinstance(b, list) and len(b) == 4 else np.nan)
    blocks = blocks.sort_values(["doc_id", "page_no", "y0", "x0", "depth", "block_id"]).reset_index(drop=True)

    table_rows = blocks[blocks["block_type"] == "Table"].copy()
    table_cells = blocks[blocks["block_type"] == "TableCell"].copy()
    table_child_map = {}
    for (doc_id, parent_id), grp in table_cells.groupby(["doc_id", "parent_id"], dropna=False):
        table_child_map[(doc_id, parent_id)] = grp.sort_values(["page_no", "y0", "x0"]).copy()

    top_level_blocks = blocks[~blocks["block_type"].isin(["TableCell"])].copy()
    structure_chunks = []
    chunk_idx = 0

    for doc_id, doc_df in top_level_blocks.groupby("doc_id", sort=False):
        doc_df = doc_df.reset_index(drop=True)
        i = 0
        while i < len(doc_df):
            row = doc_df.iloc[i]
            btype = row["block_type"]
            if btype == "SectionHeader":
                i += 1
                continue

            if btype == "Table":
                child_df = table_child_map.get((doc_id, row["block_id"]), pd.DataFrame())
                source_block_ids = [row["block_id"]]
                source_block_types = [row["block_type"]]
                if not child_df.empty:
                    source_block_ids.extend(child_df["block_id"].tolist())
                    source_block_types.extend(child_df["block_type"].tolist())
                all_bboxes = [row["bbox_obj"]]
                if not child_df.empty:
                    all_bboxes.extend(child_df["bbox_obj"].tolist())
                table_kind, table_headers_json, table_rows_json, table_n_rows, table_html = parse_table_html_structure(row["html"])
                text_raw = normalize_space(row["text_raw"])
                title_text = last_section_title(row["section_path"])
                text_with_title = f"{title_text}\n{text_raw}" if title_text and text_raw else text_raw
                structure_chunks.append({
                    "doc_id": doc_id,
                    "struct_chunk_id": f"{doc_id}_sc{chunk_idx:06d}",
                    "page_start": int(row["page_no"]),
                    "page_end": int(row["page_no"]),
                    "section_path": row["section_path"],
                    "title_text": title_text,
                    "chunk_kind": table_kind if table_kind else "table_other_chunk",
                    "source_block_ids": json.dumps(source_block_ids, ensure_ascii=False),
                    "source_block_types": json.dumps(source_block_types, ensure_ascii=False),
                    "bbox": json.dumps(bbox_union(all_bboxes), ensure_ascii=False),
                    "n_blocks": len(source_block_ids),
                    "text_raw": text_raw,
                    "text_with_title": text_with_title,
                    "table_kind": table_kind,
                    "table_headers_json": table_headers_json,
                    "table_rows_json": table_rows_json,
                    "table_n_rows": table_n_rows,
                    "table_html": table_html,
                    "notes": "table kept as structural chunk; structure parsed from Table html; TableCell ids retained for provenance",
                })
                chunk_idx += 1
                i += 1
                continue

            if btype == "ListItem":
                chunk_rows = [row]
                j = i + 1
                while j < len(doc_df):
                    next_row = doc_df.iloc[j]
                    if next_row["block_type"] != "ListItem":
                        break
                    if not list_merge_ok(chunk_rows[-1], next_row):
                        break
                    chunk_rows.append(next_row)
                    j += 1
                title_text = last_section_title(row["section_path"])
                bullet_lines = [normalize_space(r["text_raw"]) for _, r in pd.DataFrame(chunk_rows).iterrows()]
                bullet_lines = [text for text in bullet_lines if text]
                text_raw = "\n".join(f"- {text}" for text in bullet_lines)
                text_with_title = f"{title_text}\n{text_raw}" if title_text else text_raw
                structure_chunks.append({
                    "doc_id": doc_id,
                    "struct_chunk_id": f"{doc_id}_sc{chunk_idx:06d}",
                    "page_start": int(min(r["page_no"] for r in chunk_rows)),
                    "page_end": int(max(r["page_no"] for r in chunk_rows)),
                    "section_path": row["section_path"],
                    "title_text": title_text,
                    "chunk_kind": "list_chunk",
                    "source_block_ids": json.dumps([r["block_id"] for r in chunk_rows], ensure_ascii=False),
                    "source_block_types": json.dumps([r["block_type"] for r in chunk_rows], ensure_ascii=False),
                    "bbox": json.dumps(bbox_union([r["bbox_obj"] for r in chunk_rows]), ensure_ascii=False),
                    "n_blocks": len(chunk_rows),
                    "text_raw": text_raw,
                    "text_with_title": text_with_title,
                    "table_kind": "",
                    "table_headers_json": "",
                    "table_rows_json": "",
                    "table_n_rows": 0,
                    "table_html": "",
                    "notes": "consecutive list items merged under same section_path",
                })
                chunk_idx += 1
                i = j
                continue

            if btype == "Text":
                text_val = normalize_space(row["text_raw"])
                if text_val in {"", ".", "-", "•"}:
                    i += 1
                    continue
                chunk_rows = [row]
                j = i + 1
                while j < len(doc_df):
                    next_row = doc_df.iloc[j]
                    if next_row["block_type"] != "Text":
                        break
                    if not textlike_merge_ok(chunk_rows[-1], next_row):
                        break
                    chunk_rows.append(next_row)
                    j += 1
                title_text = last_section_title(row["section_path"])
                paragraph_text = " ".join(normalize_space(r["text_raw"]) for _, r in pd.DataFrame(chunk_rows).iterrows())
                paragraph_text = normalize_space(paragraph_text)
                chunk_kind = "single_text_chunk" if len(chunk_rows) == 1 else "paragraph_chunk"
                text_with_title = f"{title_text}\n{paragraph_text}" if title_text else paragraph_text
                structure_chunks.append({
                    "doc_id": doc_id,
                    "struct_chunk_id": f"{doc_id}_sc{chunk_idx:06d}",
                    "page_start": int(min(r["page_no"] for r in chunk_rows)),
                    "page_end": int(max(r["page_no"] for r in chunk_rows)),
                    "section_path": row["section_path"],
                    "title_text": title_text,
                    "chunk_kind": chunk_kind,
                    "source_block_ids": json.dumps([r["block_id"] for r in chunk_rows], ensure_ascii=False),
                    "source_block_types": json.dumps([r["block_type"] for r in chunk_rows], ensure_ascii=False),
                    "bbox": json.dumps(bbox_union([r["bbox_obj"] for r in chunk_rows]), ensure_ascii=False),
                    "n_blocks": len(chunk_rows),
                    "text_raw": paragraph_text,
                    "text_with_title": text_with_title,
                    "table_kind": "",
                    "table_headers_json": "",
                    "table_rows_json": "",
                    "table_n_rows": 0,
                    "table_html": "",
                    "notes": "nearby text blocks merged within same section_path",
                })
                chunk_idx += 1
                i = j
                continue
            i += 1

    return pd.DataFrame(structure_chunks)
