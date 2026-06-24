"""Select the most important extracted items for the visual summary.

Produces a module-grouped, taxonomy-aware output matching the target format:
  1.1 Wat is verzekerd (per module)
  1.2 Uitsluitingen (common + module-specific)
  1.3 Meldplichten (notification duties)
  1.4 Verplichtingen bij schadeafhandeling (claim obligations)
  + Limits and deadlines
"""

from __future__ import annotations

import json
import logging
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ── Module display order and labels ──────────────────────────────────────

MODULE_ORDER = [
    "wa", "beperkt_casco", "all_risk",
    "pechhulp", "rechtsbijstand", "inzittenden", "general",
]

MODULE_LABELS = {
    "wa": "WA – Wettelijke Aansprakelijkheid",
    "beperkt_casco": "Beperkt Casco",
    "all_risk": "Casco Allrisk",
    "pechhulp": "Pechhulp / Hulpverlening",
    "rechtsbijstand": "Rechtsbijstand / Juridische hulp",
    "inzittenden": "Ongevallen Inzittenden",
    "general": "Algemeen",
}

# Item types that are customer-relevant
CUSTOMER_RELEVANT_TYPES = {
    "covered", "not_covered", "condition", "limit", "deadline",
    "obligation", "notification_duty", "claim_obligation",
}

# Section path priority keywords
SECTION_PRIORITY_KEYWORDS = {
    "wat is verzekerd": 1.0,
    "dit is verzekerd": 1.0,
    "verzekerd": 0.9,
    "wat is niet verzekerd": 0.9,
    "dit is niet verzekerd": 0.9,
    "niet verzekerd": 0.85,
    "nooit verzekerd": 0.85,
    "eigen risico": 0.7,
    "maximale vergoeding": 0.7,
    "meldplicht": 0.65,
    "schade melden": 0.6,
    "verplichtingen": 0.6,
    "spelregels": 0.4,
    "klachtenregeling": 0.2,
    "begrippen": 0.1,
}

TYPE_PRIORITY = {
    "covered": 1.0,
    "not_covered": 0.9,
    "notification_duty": 0.85,
    "claim_obligation": 0.8,
    "limit": 0.8,
    "deadline": 0.8,
    "condition": 0.6,
    "obligation": 0.7,
}


MODULE_ALIASES = {
    "assistance": "pechhulp",
    "replacement_vehicle": "pechhulp",
    "hulpverlening": "pechhulp",
    "pech": "pechhulp",
    "casco": "all_risk",
    "volledig_casco": "all_risk",
    "allrisk": "all_risk",
    "own_damage": "all_risk",
    "limited_casco": "beperkt_casco",
    "fire": "beperkt_casco",
    "theft": "beperkt_casco",
    "storm_weather": "beperkt_casco",
    "glass": "beperkt_casco",
    "accessories": "all_risk",
    "wettelijke_aansprakelijkheid": "wa",
    "aansprakelijkheid": "wa",
    "liability": "wa",
    "juridisch": "rechtsbijstand",
    "legal": "rechtsbijstand",
    "ongevallen_inzittenden": "inzittenden",
    "passengers": "inzittenden",
}


# ── Scoring ──────────────────────────────────────────────────────────────

def _section_score(section_path: str) -> float:
    path_lower = section_path.lower()
    best = 0.3
    for keyword, score in SECTION_PRIORITY_KEYWORDS.items():
        if keyword in path_lower:
            best = max(best, score)
    return best


def _specificity_score(row: dict) -> float:
    score = 0.0
    money = _parse_json_field(row.get("money_amounts", "[]"))
    deadlines = _parse_json_field(row.get("deadlines", "[]"))
    if money:
        score += 0.5
    if deadlines:
        score += 0.5
    return min(score, 1.0)


def _parse_json_field(val) -> list:
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return []
    return []


def score_item(row: dict) -> float:
    """Composite score for ranking."""
    importance = float(row.get("importance", 3)) / 5.0
    item_type = row.get("item_type", "covered")

    score = (
        0.35 * importance
        + 0.25 * _section_score(row.get("section_path", ""))
        + 0.20 * TYPE_PRIORITY.get(item_type, 0.3)
        + 0.10 * _specificity_score(row)
        + 0.10 * 0.7  # base visual suitability
    )

    if row.get("validation_status") == "unverified":
        score *= 0.7

    return round(score, 4)


# ── Selection ────────────────────────────────────────────────────────────

def select_all_relevant_items(
    extractions_df: pd.DataFrame,
    doc_id: str,
) -> pd.DataFrame:
    """Select all customer-relevant items for a document, scored and ranked.

    Unlike the old select_top_items, this does NOT cap per type.
    Instead it selects everything relevant and lets the output builder
    organize by module and type.
    """
    doc_items = extractions_df[
        (extractions_df["doc_id"] == doc_id)
        & (extractions_df["item_type"].isin(CUSTOMER_RELEVANT_TYPES))
    ].copy()

    if doc_items.empty:
        return doc_items

    # Normalize module names
    doc_items["module"] = doc_items["module"].map(
        lambda m: MODULE_ALIASES.get(m, m)
    )

    doc_items["composite_score"] = doc_items.apply(
        lambda r: score_item(r.to_dict()), axis=1,
    )

    return doc_items.sort_values("composite_score", ascending=False)


# ── Module-grouped output builder ────────────────────────────────────────

def _make_entry(row: pd.Series) -> dict[str, Any]:
    """Build one summary entry from a DataFrame row."""
    matched_ids = _parse_json_field(row.get("matched_clause_ids", "[]"))
    headline = row.get("headline") or ""

    # Fallback: if headline is empty or null, use quote but don't truncate
    if not headline:
        headline = str(row.get("exact_quote", ""))

    pages: list[int] = []
    ps = row.get("page_start")
    pe = row.get("page_end")
    if pd.notna(ps):
        pages = list(range(int(ps), int(pe or ps) + 1))

    entry = {
        "headline": headline,
        "exact_quote": str(row.get("exact_quote", "")),
        "module": row.get("module", "general"),
        "item_type": row.get("item_type", ""),
        "theme": row.get("theme", None),
        "icon_hint": row.get("icon_hint", None),
        "importance": int(row.get("importance", 3)),
        "source_pages": pages,
        "chunk_ids": matched_ids,
        "composite_score": row.get("composite_score", 0),
        "exclusion_scope": [MODULE_ALIASES.get(m, m) for m in _parse_json_field(row.get("exclusion_scope", "[]"))],
    }
    # Propagate table-structure flag for downstream filter
    if row.get("column_role_override"):
        entry["from_table"] = True
    return entry


def build_module_grouped_summary(
    selected_df: pd.DataFrame,
    doc_id: str,
    model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """Build the final summary JSON grouped by module, matching the target format.

    Structure:
        coverage_by_module: {module: [items]}
        exclusions:
            common: [items applying to multiple modules]
            by_module: {module: [items]}
        notification_duties: [items]
        claim_obligations: [items]
        limits_by_module: {module: [items]}
        deadlines: [items]
        conditions: [items]
    """
    if selected_df.empty:
        return _empty_summary(doc_id, model)

    # ── Coverage by module ───────────────────────────────────────────
    covered = selected_df[selected_df["item_type"] == "covered"]
    coverage_by_module: dict[str, list] = {}

    for module in MODULE_ORDER:
        mod_items = covered[covered["module"] == module]
        if not mod_items.empty:
            entries = [_make_entry(row) for _, row in mod_items.iterrows()]
            coverage_by_module[module] = entries

    # ── Exclusions ───────────────────────────────────────────────────
    not_covered = selected_df[selected_df["item_type"] == "not_covered"]

    # Split into common vs module-specific
    common_exclusions: list[dict] = []
    exclusions_by_module: dict[str, list] = {}

    for _, row in not_covered.iterrows():
        entry = _make_entry(row)
        scope = entry.get("exclusion_scope", [])
        module = row.get("module", "general")

        # If scope has multiple modules or is "general", it's common
        if module == "general" or len(scope) > 1:
            common_exclusions.append(entry)
        else:
            mod = scope[0] if scope else module
            exclusions_by_module.setdefault(mod, []).append(entry)

    # ── Notification duties ──────────────────────────────────────────
    notif_types = {"notification_duty", "obligation"}
    notif_items = selected_df[selected_df["item_type"].isin(notif_types)]
    notification_duties = [_make_entry(row) for _, row in notif_items.iterrows()]

    # ── Claim obligations ────────────────────────────────────────────
    claim_items = selected_df[selected_df["item_type"] == "claim_obligation"]
    claim_obligations = [_make_entry(row) for _, row in claim_items.iterrows()]

    # ── Limits by module ─────────────────────────────────────────────
    limits = selected_df[selected_df["item_type"] == "limit"]
    limits_by_module: dict[str, list] = {}

    for module in MODULE_ORDER:
        mod_items = limits[limits["module"] == module]
        if not mod_items.empty:
            entries = [_make_entry(row) for _, row in mod_items.iterrows()]
            limits_by_module[module] = entries

    # ── Deadlines ────────────────────────────────────────────────────
    deadline_items = selected_df[selected_df["item_type"] == "deadline"]
    deadlines = [_make_entry(row) for _, row in deadline_items.iterrows()]

    # ── Conditions ───────────────────────────────────────────────────
    condition_items = selected_df[selected_df["item_type"] == "condition"]
    conditions = [_make_entry(row) for _, row in condition_items.iterrows()]

    # ── Assemble ─────────────────────────────────────────────────────
    summary = {
        "coverage_by_module": coverage_by_module,
        "exclusions": {
            "common": common_exclusions,
            "by_module": exclusions_by_module,
        },
        "notification_duties": notification_duties,
        "claim_obligations": claim_obligations,
        "limits_by_module": limits_by_module,
        "deadlines": deadlines,
        "conditions": conditions,
    }

    # Count items
    n_covered = sum(len(v) for v in coverage_by_module.values())
    n_excl = len(common_exclusions) + sum(len(v) for v in exclusions_by_module.values())
    n_limits = sum(len(v) for v in limits_by_module.values())

    return {
        "doc_id": doc_id,
        "insurance_type": "car",
        "model": model,
        "summary": summary,
        "module_labels": {
            m: MODULE_LABELS.get(m, m)
            for m in coverage_by_module.keys()
        },
        "item_counts": {
            "covered": n_covered,
            "exclusions_common": len(common_exclusions),
            "exclusions_module_specific": sum(len(v) for v in exclusions_by_module.values()),
            "notification_duties": len(notification_duties),
            "claim_obligations": len(claim_obligations),
            "limits": n_limits,
            "deadlines": len(deadlines),
            "conditions": len(conditions),
        },
        "total_items": (
            n_covered + n_excl + len(notification_duties) + len(claim_obligations)
            + n_limits + len(deadlines) + len(conditions)
        ),
    }


def _empty_summary(doc_id: str, model: str) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "insurance_type": "car",
        "model": model,
        "summary": {
            "coverage_by_module": {},
            "exclusions": {"common": [], "by_module": {}},
            "notification_duties": [],
            "claim_obligations": [],
            "limits_by_module": {},
            "deadlines": [],
            "conditions": [],
        },
        "module_labels": {},
        "item_counts": {},
        "total_items": 0,
    }


# ── Generate all summaries ───────────────────────────────────────────────

def generate_all_summaries(
    extractions_df: pd.DataFrame,
    output_dir: Path,
    model: str = "gpt-4o-mini",
) -> dict[str, dict]:
    """Generate visual summary JSON for all documents."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict] = {}

    for doc_id in sorted(extractions_df["doc_id"].unique()):
        selected = select_all_relevant_items(extractions_df, doc_id)
        summary = build_module_grouped_summary(selected, doc_id, model)
        summaries[doc_id] = summary

        doc_dir = output_dir / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)
        summary_path = doc_dir / "visual_summary_items.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(
            "%s: %d items (%s)",
            doc_id,
            summary["total_items"],
            ", ".join(f"{k}={v}" for k, v in summary["item_counts"].items()),
        )

    return summaries


# ── HTML Report Generator ────────────────────────────────────────────────

def generate_html_report(
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    """Generate an HTML report for high-recall or filtered summary JSON."""
    doc_id = str(summary["doc_id"])
    s = summary.get("summary", {})
    labels = summary.get("module_labels", {})
    is_filtered = bool(summary.get("filter_prompt_version"))
    report_title = "Gefilterde samenvatting" if is_filtered else "Samenvatting"
    report_label = "klantgerichte selectie" if is_filtered else "hoog-recall extractie-overzicht"

    html_parts: list[str] = [
        "<!DOCTYPE html>",
        "<html lang='nl'><head><meta charset='utf-8'>",
        f"<title>{_html_text(report_title)} {_html_text(doc_id)}</title>",
        "<style>",
        "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; "
        "max-width: 900px; margin: 2em auto; padding: 0 1em; color: #1a1a1a; line-height: 1.6; }",
        "h1 { color: #003366; border-bottom: 3px solid #003366; padding-bottom: 0.3em; }",
        "h2 { color: #004488; margin-top: 1.5em; border-bottom: 2px solid #e0e0e0; padding-bottom: 0.2em; }",
        "h3 { color: #005599; margin-top: 1.2em; }",
        "ul { padding-left: 1.5em; }",
        "li { margin-bottom: 0.4em; }",
        ".badge { display: inline-block; background: #e8f2ff; color: #003366; padding: 0.2em 0.6em; "
        "border-radius: 999px; font-size: 0.85em; font-weight: 600; margin-bottom: 0.6em; }",
        ".module-header { background: #f0f6ff; padding: 0.4em 0.8em; border-radius: 4px; "
        "font-weight: bold; margin-top: 1em; }",
        ".covered { color: #0a6e0a; }",
        ".not-covered { color: #cc0000; }",
        ".limit { color: #cc6600; }",
        ".deadline { color: #6600cc; }",
        ".duty { color: #006699; }",
        ".quote { font-size: 0.85em; color: #666; font-style: italic; margin-left: 1.5em; }",
        ".page-ref { font-size: 0.8em; color: #999; }",
        ".stats { background: #f8f8f8; padding: 1em; border-radius: 6px; margin: 1em 0; }",
        ".stats div { margin: 0.2em 0; }",
        ".counts { color: #555; font-size: 0.92em; }",
        "</style></head><body>",
        f"<h1>{_html_text(report_title)}: {_html_text(doc_id)}</h1>",
        f"<div class='badge'>{_html_text(report_label)}</div>",
    ]

    _append_stats(html_parts, summary)

    # ── 1.1 Coverage by module ───────────────────────────────────────
    html_parts.append("<h2>1.1 Wat is verzekerd?</h2>")

    coverage = s.get("coverage_by_module", {})
    if coverage:
        for module in _ordered_module_keys(coverage):
            items = coverage.get(module, [])
            if not items:
                continue
            _append_module_header(html_parts, labels, module)
            _append_item_list(html_parts, items, "covered")
    else:
        html_parts.append("<p><em>Geen dekkingen gevonden.</em></p>")

    # ── 1.2 Exclusions ───────────────────────────────────────────────
    html_parts.append("<h2>1.2 Uitsluitingen</h2>")

    excl = s.get("exclusions", {})
    common = excl.get("common", [])
    by_mod = excl.get("by_module", {})

    if common:
        html_parts.append("<h3>Veelvoorkomende uitsluitingen (algemeen)</h3>")
        _append_item_list(html_parts, common, "not-covered")

    for module in _ordered_module_keys(by_mod):
        items = by_mod.get(module, [])
        if not items:
            continue
        label = _module_label(labels, module)
        html_parts.append(f"<h3>{_html_text(label)} – specifieke uitsluitingen</h3>")
        _append_item_list(html_parts, items, "not-covered")

    if not common and not any(by_mod.values()):
        html_parts.append("<p><em>Geen uitsluitingen gevonden.</em></p>")

    # ── 1.3 Notification duties ──────────────────────────────────────
    html_parts.append("<h2>1.3 Meldplichten</h2>")

    duties = s.get("notification_duties", [])
    if duties:
        _append_item_list(html_parts, duties, "duty")
    else:
        html_parts.append("<p><em>Geen meldplichten gevonden.</em></p>")

    # ── 1.4 Claim obligations ────────────────────────────────────────
    html_parts.append("<h2>1.4 Verplichtingen bij schadeafhandeling</h2>")

    claims = s.get("claim_obligations", [])
    if claims:
        _append_item_list(html_parts, claims, "duty")
    else:
        html_parts.append("<p><em>Geen verplichtingen gevonden.</em></p>")

    # ── Limits ───────────────────────────────────────────────────────
    html_parts.append("<h2>1.5 Limieten &amp; maximale vergoedingen</h2>")

    limits = s.get("limits_by_module", {})
    if limits:
        for module in _ordered_module_keys(limits):
            items = limits.get(module, [])
            if not items:
                continue
            _append_module_header(html_parts, labels, module)
            _append_item_list(html_parts, items, "limit")
    else:
        html_parts.append("<p><em>Geen limieten gevonden.</em></p>")

    # ── Deadlines ────────────────────────────────────────────────────
    html_parts.append("<h2>1.6 Termijnen</h2>")

    deadlines = s.get("deadlines", [])
    if deadlines:
        _append_item_list(html_parts, deadlines, "deadline")
    else:
        html_parts.append("<p><em>Geen termijnen gevonden.</em></p>")

    # ── Conditions ───────────────────────────────────────────────────
    html_parts.append("<h2>1.7 Voorwaarden</h2>")

    conditions = s.get("conditions", [])
    if conditions:
        _append_item_list(html_parts, conditions, "")
    else:
        html_parts.append("<p><em>Geen voorwaarden gevonden.</em></p>")

    html_parts.append("</body></html>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))


def _append_stats(html_parts: list[str], summary: dict[str, Any]) -> None:
    """Append compact metadata and item counts to a report."""

    counts = summary.get("item_counts", {})
    html_parts.append("<div class='stats'>")
    html_parts.append(f"<div><strong>Totaal items:</strong> {_html_text(summary.get('total_items', 0))}</div>")

    if summary.get("filter_prompt_version"):
        input_total = summary.get("input_total_items")
        selected_total = summary.get("selected_source_items")
        filter_model = summary.get("filter_model")
        filter_version = summary.get("filter_prompt_version")
        source_model = summary.get("source_model")
        if input_total is not None:
            html_parts.append(f"<div><strong>Input items:</strong> {_html_text(input_total)}</div>")
        if selected_total is not None:
            html_parts.append(f"<div><strong>Geselecteerde bron-items:</strong> {_html_text(selected_total)}</div>")
        if filter_model:
            html_parts.append(f"<div><strong>Filtermodel:</strong> {_html_text(filter_model)}</div>")
        if filter_version:
            html_parts.append(f"<div><strong>Filterversie:</strong> {_html_text(filter_version)}</div>")
        if source_model:
            html_parts.append(f"<div><strong>Extractiemodel:</strong> {_html_text(source_model)}</div>")
    elif summary.get("model"):
        html_parts.append(f"<div><strong>Extractiemodel:</strong> {_html_text(summary.get('model'))}</div>")

    if counts:
        count_text = " | ".join(f"{_html_text(k)}: {_html_text(v)}" for k, v in counts.items())
        html_parts.append(f"<div class='counts'>{count_text}</div>")
    html_parts.append("</div>")


def _append_module_header(html_parts: list[str], labels: dict[str, Any], module: str) -> None:
    label = _module_label(labels, module)
    html_parts.append(f"<div class='module-header'>{_html_text(label)}</div>")


def _append_item_list(html_parts: list[str], items: list[dict[str, Any]], css_class: str) -> None:
    html_parts.append("<ul>")
    for item in items:
        class_attr = f" class='{css_class}'" if css_class else ""
        headline = _html_text(item.get("headline", ""))
        html_parts.append(f"<li{class_attr}>{headline}{_page_ref(item)}</li>")
    html_parts.append("</ul>")


def _module_label(labels: dict[str, Any], module: str) -> str:
    return str(labels.get(module, MODULE_LABELS.get(module, module)))


def _ordered_module_keys(module_items: dict[str, Any]) -> list[str]:
    ordered = [module for module in MODULE_ORDER if module in module_items]
    remaining = sorted(module for module in module_items if module not in MODULE_ORDER)
    return ordered + remaining


def _page_ref(item: dict[str, Any]) -> str:
    pages = item.get("source_pages", [])
    if not pages:
        return ""
    page_text = ", ".join(str(page) for page in pages)
    return f" <span class='page-ref'>(p. {_html_text(page_text)})</span>"


def _html_text(value: Any) -> str:
    return escape("" if value is None else str(value), quote=True)
