"""Annotate a PDF with highlight rectangles and labels for summary items.

Uses PyMuPDF (fitz) to add colored highlight annotations with pop-up labels
so that clicking a card in the HTML summary opens the PDF to the exact page
with only the relevant quote visually highlighted and labeled.
"""

from __future__ import annotations

import fitz  # PyMuPDF
from pathlib import Path
from typing import Any


# Color mapping per item_type (RGB floats for PyMuPDF)
HIGHLIGHT_COLORS: dict[str, tuple[float, float, float]] = {
    "covered":           (0.6, 0.9, 0.6),   # green
    "not_covered":       (1.0, 0.5, 0.5),   # coral/red
    "notification_duty": (1.0, 0.85, 0.4),  # amber
    "obligation":        (1.0, 0.85, 0.4),  # amber
    "claim_obligation":  (1.0, 0.85, 0.4),  # amber
    "limit":             (0.5, 0.7, 1.0),   # blue
    "deadline":          (1.0, 0.75, 0.3),  # orange-amber
    "condition":         (0.5, 0.9, 0.9),   # cyan
}

DEFAULT_COLOR = (1.0, 1.0, 0.4)  # yellow fallback


def _search_quote_on_page(page: fitz.Page, quote_text: str) -> list[fitz.Quad]:
    """Search for quote text on a page, return quads for highlighting.

    Tries the full quote first, then progressively shorter prefixes to
    handle minor text extraction differences.
    """
    if not quote_text or not quote_text.strip():
        return []

    # Normalize whitespace
    normalized = " ".join(quote_text.split())

    # Try full text first
    quads = page.search_for(normalized, quads=True)
    if quads:
        return quads

    # Try first 150 chars (handles slight trailing differences)
    if len(normalized) > 150:
        short = normalized[:150].rsplit(" ", 1)[0]
        quads = page.search_for(short, quads=True)
        if quads:
            return quads

    # Try first 80 chars
    if len(normalized) > 80:
        short = normalized[:80].rsplit(" ", 1)[0]
        quads = page.search_for(short, quads=True)
        if quads:
            return quads

    # Try first sentence or 50 chars
    short = normalized[:50].rsplit(" ", 1)[0]
    quads = page.search_for(short, quads=True)
    return quads


def annotate_pdf(
    pdf_path: Path,
    items: list[dict[str, Any]],
    output_path: Path | None = None,
) -> Path:
    """Create an annotated copy of a PDF with highlights and labels for each item.

    Parameters
    ----------
    pdf_path : Path
        Path to the source PDF.
    items : list[dict]
        Summary items, each with keys:
        - exact_quote: str
        - source_pages: list[int] (1-indexed page numbers)
        - item_type: str (for color selection)
        - headline: str (used as the annotation label)
        - item_id: str | int (optional, for labeling)
    output_path : Path | None
        Where to save the annotated PDF. Defaults to <name>_annotated.pdf.

    Returns
    -------
    Path
        The path to the annotated PDF.
    """
    if output_path is None:
        output_path = pdf_path.with_stem(pdf_path.stem + "_annotated")

    doc = fitz.open(pdf_path)

    for idx, item in enumerate(items):
        quote_text = item.get("exact_quote", "")
        pages = item.get("source_pages", [])
        item_type = item.get("item_type", "covered")
        headline = item.get("headline", f"Item {idx + 1}")
        item_id = item.get("item_id", str(idx + 1))
        color = HIGHLIGHT_COLORS.get(item_type, DEFAULT_COLOR)

        if not pages or not quote_text:
            continue

        for page_num in pages:
            # source_pages are 1-indexed, PyMuPDF is 0-indexed
            if page_num < 1 or page_num > len(doc):
                continue
            page = doc[page_num - 1]
            quads = _search_quote_on_page(page, quote_text)

            if not quads:
                continue

            # Add highlight annotation
            annot = page.add_highlight_annot(quads)
            annot.set_colors(stroke=color)
            annot.set_opacity(0.4)

            # Build a label: [item_id] headline
            label = f"[{item_id}] {headline}"
            annot.set_info(title=label, content=quote_text[:300], creationDate="", modDate="")
            annot.update()

            # Add a small text label near the first quad for visual identification
            first_quad = quads[0]
            label_point = fitz.Point(first_quad.ul.x, first_quad.ul.y - 2)

            # Draw a small label box above the highlight
            short_label = f"[{item_id}]"
            fontsize = 7
            text_length = fitz.get_text_length(short_label, fontsize=fontsize)
            label_rect = fitz.Rect(
                label_point.x,
                label_point.y - fontsize - 3,
                label_point.x + text_length + 6,
                label_point.y + 1,
            )

            # Background rectangle for readability
            shape = page.new_shape()
            shape.draw_rect(label_rect)
            shape.finish(color=color, fill=color, width=0.3)
            shape.insert_text(
                fitz.Point(label_rect.x0 + 3, label_rect.y1 - 3),
                short_label,
                fontsize=fontsize,
                color=(0, 0, 0),
            )
            shape.commit()

            # Only annotate on the first matching page
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    doc.close()
    return output_path


def annotate_from_summary(
    pdf_path: Path,
    summary_data: dict[str, Any],
    output_path: Path | None = None,
) -> Path:
    """Annotate a PDF from a full summary JSON structure.

    Walks the summary and extracts all items with their quotes and pages,
    assigning sequential IDs for labeling.
    """
    items = _collect_items_from_summary(summary_data)
    return annotate_pdf(pdf_path, items, output_path)


def _collect_items_from_summary(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Recursively collect all items from the summary structure."""
    summary = data.get("summary", data)
    items = []
    counter = [0]

    def walk(obj):
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict) and "exact_quote" in item:
                    counter[0] += 1
                    items.append({
                        "exact_quote": item.get("exact_quote", ""),
                        "source_pages": item.get("source_pages", []),
                        "item_type": item.get("item_type", "covered"),
                        "headline": item.get("headline", ""),
                        "item_id": str(counter[0]),
                    })
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v)

    walk(summary)
    return items
