"""Render filtered policy summaries to customer-facing HTML pages."""

import argparse
import json
import shutil
import sys
import webbrowser
from pathlib import Path
from typing import Any
from urllib.parse import quote

from src.pdf.annotate import annotate_from_summary

# ---------------------------------------------------------------------------
# Theme maps
# ---------------------------------------------------------------------------

MODULE_NAMES = {
    "wa": "WA — Aansprakelijkheid",
    "inzittenden": "Inzittenden",
    "rechtsbijstand": "Rechtsbijstand",
    "pechhulp": "Pechhulp",
    "beperkt_casco": "Beperkt casco",
    "all_risk": "All risk",
    "general": "Algemeen",
    "common": "Algemeen (alle modules)",
}

MODULE_ICONS = {
    "wa": "shield-check",
    "inzittenden": "users",
    "rechtsbijstand": "gavel",
    "pechhulp": "tool",
    "beperkt_casco": "lock",
    "all_risk": "shield",
    "general": "file-text",
    "common": "list",
}

THEME_ICONS = {
    "liability": "scale",
    "passengers": "users",
    "legal": "gavel",
    "breakdown": "tool",
    "theft": "lock-open",
    "assistance": "lifebuoy",
    "glass": "diamond",
    "fire": "flame",
    "valuation": "coins",
    "behavior": "alert-triangle",
    "events": "flag",
    "usage_restrictions": "ban",
    "driver_restrictions": "id-badge",
    "change_reporting": "bell-ringing",
    "damage_reporting": "clipboard-list",
    "claim_handling": "clipboard-list",
    "deadline": "calendar-time",
    "replacement_vehicle": "car",
    "animal_damage": "paw",
    "other_coverage": "circle-check",
    "other_exclusion": "circle-x",
}

SECTION_DEFS = [
    {"key": "coverage_by_module",  "label": "Dekking",      "icon": "shield-check",    "default_type": "covered"},
    {"key": "exclusions",          "label": "Uitsluitingen", "icon": "shield-x",        "default_type": "not_covered"},
    {"key": "notification_duties", "label": "Meldplichten",  "icon": "bell",            "default_type": "notification_duty"},
    {"key": "claim_obligations",   "label": "Schadeplichten","icon": "clipboard-list",  "default_type": "claim_obligation"},
    {"key": "limits_by_module",    "label": "Limieten",      "icon": "adjustments",     "default_type": "limit"},
    {"key": "deadlines",           "label": "Termijnen",     "icon": "calendar-time",    "default_type": "deadline"},
    {"key": "conditions",          "label": "Voorwaarden",   "icon": "file-certificate","default_type": "condition"},
]

ITEM_TYPE_META = {
    "covered":           {"label": "Gedekt",       "icon": "circle-check",    "color": "green"},
    "not_covered":       {"label": "Niet gedekt",  "icon": "circle-x",        "color": "coral"},
    "notification_duty": {"label": "Meldplicht",   "icon": "bell-ringing",    "color": "amber"},
    "obligation":        {"label": "Verplichting", "icon": "clipboard-list",  "color": "amber"},
    "claim_obligation":  {"label": "Schadeplicht", "icon": "clipboard-list",  "color": "amber"},
    "limit":             {"label": "Limiet",        "icon": "info-circle",     "color": "blue"},
    "deadline":          {"label": "Termijn",       "icon": "calendar-time",   "color": "amber"},
    "condition":         {"label": "Voorwaarde",   "icon": "file-info",       "color": "cyan"},
}

INSURANCE_TYPE_ICONS = {
    "car":   "car",
    "home":  "home",
    "life":  "heart",
    "travel":"plane",
    "health":"stethoscope",
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def flatten_section(section_data, default_type):
    """Return {module_key: [items]} from any section shape."""
    if not section_data:
        return {}
    if isinstance(section_data, list):
        return {"_flat": section_data}
    groups = {}
    for k, v in section_data.items():
        if k == "by_module" and isinstance(v, dict):
            for mod, items in v.items():
                groups.setdefault(mod, []).extend(items)
        elif isinstance(v, list):
            groups[k] = v
    return groups


def count_items(summary):
    counts = {
        "covered": 0,
        "not_covered": 0,
        "notification_duty": 0,
        "obligation": 0,
        "claim_obligation": 0,
        "limit": 0,
        "deadline": 0,
        "condition": 0,
    }
    def walk(obj):
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict) and "item_type" in item:
                    t = item["item_type"]
                    counts[t] = counts.get(t, 0) + 1
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v)
    walk(summary)
    return counts


# ---------------------------------------------------------------------------
# HTML builders
# ---------------------------------------------------------------------------

def esc(s):
    """Minimal HTML escaping."""
    return (str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def icon(name, extra_class="", size=16):
    return f'<i class="ti ti-{esc(name)} {extra_class}" style="font-size:{size}px" aria-hidden="true"></i>'


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate display text without changing the source quote used for search links."""

    value = str(text or "")
    return value if len(value) <= max_chars else value[:max_chars] + "..."


def build_pdf_target(pdf_url: str | None, page: int | None, quote_text: str | None = None) -> str:
    """Build a PDF link that opens the page. Highlighting is done via annotations."""

    if not pdf_url:
        return "#"
    if page:
        return f"{pdf_url}#page={page}"
    return pdf_url


def short_clause_id(clause_id: Any) -> str:
    """Return a compact clause label for display."""

    text = str(clause_id or "")
    parts = text.split(":")
    if len(parts) >= 2:
        chunk = parts[1]
        if "_" in chunk:
            chunk = chunk.split("_")[-1]
        return f"{chunk}:{parts[-1]}"
    return text


def render_item(item, default_type, pdf_url=None):
    t = item.get("item_type") or default_type or "covered"
    meta = ITEM_TYPE_META.get(t, ITEM_TYPE_META["covered"])
    theme = item.get("theme", "")
    item_icon = THEME_ICONS.get(theme, meta["icon"])
    item_id = item.get("_item_id", "")

    raw_quote = str(item.get("exact_quote", "") or "")
    quote = esc(truncate_text(raw_quote, 130))
    headline = esc(item.get("headline", ""))
    source_ids = item.get("chunk_ids") or item.get("source_clause_ids") or []
    source_ids = source_ids if isinstance(source_ids, list) else [source_ids]
    source_chips = "".join(
        f'<span class="clause-chip" title="{esc(source_id)}">{icon("quote", "", 10)} {esc(short_clause_id(source_id))}</span>'
        for source_id in source_ids[:4]
    )
    if len(source_ids) > 4:
        source_chips += f'<span class="clause-chip">+{len(source_ids) - 4}</span>'

    # Extract source page numbers
    pages = item.get("source_pages", [])
    page_text = ""
    target_link = "#"

    if pages:
        first_page = pages[0]
        page_text = f"p. {', '.join(map(str, pages))}"
        # If a PDF file path/URL is supplied, target the specific page metadata hash parameter
        if pdf_url:
            target_link = build_pdf_target(pdf_url, first_page, raw_quote)

    # Wrap the entire card component inside an anchor tag link if a PDF target exists
    link_title = "Open bronpagina in PDF — markering [{0}] in het document".format(item_id) if item_id else "Open bronpagina in PDF"
    link_start = f'<a href="{esc(target_link)}" target="_blank" class="card-anchor-link" title="{esc(link_title)}">' if pdf_url else ''
    link_end = '</a>' if pdf_url else ''

    id_badge = f'<span class="item-id-badge">[{esc(str(item_id))}]</span>' if item_id else ''

    return f"""
      {link_start}
      <div class="item-card type-{esc(t)}">
        <div class="item-icon-wrap type-bg-{esc(t)}">
          <i class="ti ti-{esc(item_icon)}" style="font-size:18px" aria-hidden="true"></i>
        </div>
        <div class="item-body">
          <div class="item-headline">{id_badge} {headline}</div>
          {f'<div class="item-quote">{quote}</div>' if quote else ''}
          {f'<div class="source-row">{source_chips}</div>' if source_chips else ''}
          <div class="item-footer">
            <span class="item-badge type-badge-{esc(t)}">{icon(meta['icon'], '', 11)} {esc(meta['label'])}</span>
            {f'<span class="item-page-badge">{icon("file-search", "", 11)} {esc(page_text)}</span>' if page_text else ''}
          </div>
        </div>
      </div>
      {link_end}"""


def render_module_group(module_key, items, default_type, pdf_url=None):
    mod_name = MODULE_NAMES.get(module_key, module_key.replace("_", " ").title())
    mod_icon = MODULE_ICONS.get(module_key, "folder")
    if not mod_name:
        return "".join(render_item(i, default_type, pdf_url) for i in items)
    items_html = "".join(render_item(i, default_type, pdf_url) for i in items)
    return f"""
    <div class="module-section">
      <div class="module-label">
        {icon(mod_icon, '', 13)}
        {esc(mod_name)}
      </div>
      <div class="items-grid">{items_html}</div>
    </div>"""


def render_section_panel(section_key, section_data, default_type, pdf_url=None):
    groups = flatten_section(section_data, default_type)
    if not groups:
        return '<p class="empty">Geen items gevonden.</p>'
    parts = []
    for module_key, items in groups.items():
        if items:
            parts.append(render_module_group(module_key, items, default_type, pdf_url))
    return "".join(parts)


def _assign_item_ids(summary: dict) -> None:
    """Walk summary and assign sequential _item_id to each item dict."""
    counter = [0]

    def walk(obj):
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict) and "item_type" in item:
                    counter[0] += 1
                    item["_item_id"] = str(counter[0])
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v)

    walk(summary)


def build_html(data, pdf_url=None):
    summary = data.get("summary", {})
    doc_id = esc(data.get("doc_id", "—"))
    ins_type = data.get("insurance_type", "")
    ins_icon = INSURANCE_TYPE_ICONS.get(ins_type, "file-description")
    filter_ver = esc(data.get("filter_prompt_version", ""))

    counts = count_items(summary)

    # Meta chips
    chips = []
    if doc_id != "—":
        chips.append(f'{icon("file-text", "", 13)} {doc_id}')
    if ins_type:
        chips.append(f'{icon(ins_icon, "", 13)} {esc(ins_type)}')
    if filter_ver:
        chips.append(f'{icon("tag", "", 13)} {filter_ver}')
    if pdf_url:
        chips.append(f'{icon("link", "", 13)} PDF Gekoppeld: {esc(Path(pdf_url).name)}')
    chips_html = "".join(f'<span class="meta-chip">{c}</span>' for c in chips)

    # Stat cards
    stat_defs = [
        ("covered",           "Gedekt",        "shield-check",  "green"),
        ("not_covered",       "Uitsluitingen", "shield-x",      "coral"),
        ("notification_duty", "Meldplichten",  "bell",          "amber"),
        ("obligation",        "Verplichtingen","clipboard-list","amber"),
        ("claim_obligation",  "Schadeplichten","clipboard-list","amber"),
        ("limit",             "Limieten",      "adjustments",   "blue"),
        ("deadline",          "Termijnen",     "calendar-time", "amber"),
        ("condition",         "Voorwaarden",   "file-certificate","cyan"),
    ]
    stats_html = ""
    for key, label, ico, color in stat_defs:
        n = counts.get(key, 0)
        if n:
            stats_html += f"""
        <div class="stat-card">
          <div class="stat-icon c-{color}">{icon(ico, '', 20)}</div>
          <div class="stat-num">{n}</div>
          <div class="stat-label">{label}</div>
        </div>"""

    # Tabs + panels
    # Assign sequential item IDs for cross-referencing with PDF annotations
    _assign_item_ids(summary)
    available = [(d, summary[d["key"]]) for d in SECTION_DEFS if d["key"] in summary]
    tabs_html = ""
    panels_html = ""
    for i, (d, section_data) in enumerate(available):
        active = "active" if i == 0 else ""
        tabs_html += f"""
        <button class="tab-btn {active}" data-panel="panel-{esc(d['key'])}">
          {icon(d['icon'], '', 15)}
          {esc(d['label'])}
        </button>"""
        visible = "" if i == 0 else 'style="display:none"'
        panel_html = render_section_panel(d["key"], section_data, d["default_type"], pdf_url)
        panels_html += f'<div id="panel-{esc(d["key"])}" class="panel" {visible}>{panel_html}</div>'

    return f"""<!DOCTYPE html>
<html lang="nl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Polisoverzicht — {doc_id}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.31.0/dist/tabler-icons.min.css">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500&family=DM+Mono:wght@400&display=swap" rel="stylesheet">
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

:root {{
  --font: 'DM Sans', sans-serif;
  --bg: #F8F7F4;
  --surface: #FFFFFF;
  --border: rgba(0,0,0,0.08);
  --border-med: rgba(0,0,0,0.14);
  --text-primary: #1A1917;
  --text-secondary: #6B6860;
  --text-tertiary: #A09D97;
  --radius-md: 10px;
  --radius-lg: 14px;

  --c-green-bg: #E1F5EE; --c-green-mid: #1D9E75; --c-green-text: #0F6E56;
  --c-coral-bg: #FAECE7; --c-coral-mid: #D85A30; --c-coral-text: #993C1D;
  --c-amber-bg: #FAEEDA; --c-amber-mid: #BA7517; --c-amber-text: #854F0B;
  --c-blue-bg:  #E6F1FB; --c-blue-mid:  #378ADD; --c-blue-text:  #185FA5;
  --c-cyan-bg:  #E6F7FF; --c-cyan-mid:  #13C2C2; --c-cyan-text:  #006D75;
  --c-gray-bg:  #F1EFE8; --c-gray-mid:  #888780; --c-gray-text:  --text-secondary;
}}

@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #18181A; --surface: #23232A; --border: rgba(255,255,255,0.08);
    --border-med: rgba(255,255,255,0.14);
    --text-primary: #F0EFE9; --text-secondary: #A09D97; --text-tertiary: #6B6860;
    --c-green-bg: #043C2A; --c-green-text: #9FE1CB;
    --c-coral-bg: #3D1A0C; --c-coral-text: #F5C4B3;
    --c-amber-bg: #3A2000; --c-amber-text: #FAC775;
    --c-blue-bg:  #042C53; --c-blue-text:  #B5D4F4;
    --c-cyan-bg:  #003A40; --c-cyan-text:  #A3E8EC;
    --c-gray-bg:  #2C2C2A; --c-gray-text:  #D3D1C7;
  }}
}}

body {{
  font-family: var(--font);
  background: var(--bg);
  color: var(--text-primary);
  min-height: 100vh;
  padding: 0 0 4rem;
}}

header {{
  background: var(--surface);
  border-bottom: 0.5px solid var(--border-med);
  padding: 1.25rem 2rem;
  display: flex;
  align-items: center;
  gap: 12px;
}}
.header-icon {{
  width: 40px; height: 40px;
  background: var(--c-blue-bg);
  border-radius: var(--radius-md);
  display: flex; align-items: center; justify-content: center;
  color: var(--c-blue-text);
  font-size: 20px;
  flex-shrink: 0;
}}
.header-title {{ font-size: 17px; font-weight: 500; color: var(--text-primary); }}
.header-sub {{ font-size: 13px; color: var(--text-secondary); margin-top: 1px; }}

.container {{ max-width: 1040px; margin: 0 auto; padding: 2rem 1.5rem 0; }}

.meta-bar {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 1.5rem; }}
.meta-chip {{
  font-size: 12px; color: var(--text-secondary);
  background: var(--surface); border: 0.5px solid var(--border);
  border-radius: 99px; padding: 4px 12px;
  display: flex; align-items: center; gap: 5px;
}}

.stat-row {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 2rem; }}
.stat-card {{
  background: var(--surface); border: 0.5px solid var(--border);
  border-radius: var(--radius-lg); padding: 14px 18px;
  display: flex; flex-direction: column; gap: 4px; min-width: 100px;
}}
.stat-icon {{ font-size: 20px; }}
.c-green {{ color: var(--c-green-mid); }}
.c-coral {{ color: var(--c-coral-mid); }}
.c-amber {{ color: var(--c-amber-mid); }}
.c-blue  {{ color: var(--c-blue-mid);  }}
.c-cyan  {{ color: var(--c-cyan-mid);  }}
.stat-num {{ font-size: 26px; font-weight: 500; color: var(--text-primary); line-height: 1; }}
.stat-label {{ font-size: 12px; color: var(--text-secondary); }}

.section-tabs {{ display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 1.5rem; }}
.tab-btn {{
  font-family: var(--font); font-size: 13px; font-weight: 500;
  padding: 7px 16px; border-radius: 99px;
  border: 0.5px solid var(--border-med);
  background: transparent; color: var(--text-secondary);
  cursor: pointer; display: flex; align-items: center; gap: 6px;
  transition: all 0.12s;
}}
.tab-btn:hover {{ background: var(--c-gray-bg); color: var(--text-primary); }}
.tab-btn.active {{
  background: var(--text-primary); color: var(--bg);
  border-color: var(--text-primary);
}}

.panel {{ animation: fadeIn 0.15s ease; }}
@keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(4px); }} to {{ opacity: 1; transform: none; }} }}

.module-section {{ margin-bottom: 1.75rem; }}
.module-label {{
  font-size: 11px; font-weight: 500; letter-spacing: 0.07em;
  text-transform: uppercase; color: var(--text-tertiary);
  margin-bottom: 10px; padding: 0 2px;
  display: flex; align-items: center; gap: 6px;
}}

.items-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 8px;
}}

/* Clickable Anchor Wrapper styling */
.card-anchor-link {{
  text-decoration: none;
  color: inherit;
  display: block;
}}

.item-card {{
  background: var(--surface);
  border: 0.5px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 12px 14px;
  display: flex; gap: 12px; align-items: flex-start;
  transition: border-color 0.12s, box-shadow 0.12s, transform 0.12s;
  height: 100%;
}}
.card-anchor-link:hover .item-card {{
  border-color: var(--border-med);
  box-shadow: 0 4px 12px rgba(0,0,0,0.06);
  transform: translateY(-2px);
}}

.item-icon-wrap {{
  width: 36px; height: 36px; border-radius: var(--radius-md);
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
}}
.type-bg-covered           {{ background: var(--c-green-bg); color: var(--c-green-text); }}
.type-bg-not_covered       {{ background: var(--c-coral-bg); color: var(--c-coral-text); }}
.type-bg-notification_duty {{ background: var(--c-amber-bg); color: var(--c-amber-text); }}
.type-bg-obligation        {{ background: var(--c-amber-bg); color: var(--c-amber-text); }}
.type-bg-claim_obligation  {{ background: var(--c-amber-bg); color: var(--c-amber-text); }}
.type-bg-limit             {{ background: var(--c-blue-bg);  color: var(--c-blue-text);  }}
.type-bg-deadline          {{ background: var(--c-amber-bg); color: var(--c-amber-text); }}
.type-bg-condition         {{ background: var(--c-cyan-bg);  color: var(--c-cyan-text);  }}

.item-body {{ flex: 1; min-width: 0; display: flex; flex-direction: column; height: 100%; }}
.item-headline {{ font-size: 13px; font-weight: 500; color: var(--text-primary); line-height: 1.4; }}
.item-quote {{
  font-size: 12px; color: var(--text-secondary); line-height: 1.5;
  margin-top: 4px;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
  overflow: hidden;
}}
.source-row {{
  display: flex; gap: 4px; flex-wrap: wrap; margin-top: 7px;
}}
.clause-chip {{
  font-size: 10px; font-family: 'DM Mono', monospace; color: var(--text-tertiary);
  background: var(--c-gray-bg); border-radius: 99px; padding: 2px 6px;
  display: inline-flex; align-items: center; gap: 3px;
}}
.item-footer {{ display: flex; align-items: center; justify-content: space-between; margin-top: auto; padding-top: 8px; }}
.item-badge {{
  font-size: 11px; font-weight: 500; padding: 2px 8px;
  border-radius: 99px; display: flex; align-items: center; gap: 4px;
}}
.item-id-badge {{
  display: inline-block; font-size: 10px; font-weight: 700;
  background: var(--text-primary); color: var(--surface);
  padding: 1px 5px; border-radius: 4px; margin-right: 4px;
  font-family: 'DM Mono', monospace; vertical-align: middle;
}}
.type-badge-covered           {{ background: var(--c-green-bg); color: var(--c-green-text); }}
.type-badge-not_covered       {{ background: var(--c-coral-bg); color: var(--c-coral-text); }}
.type-badge-notification_duty {{ background: var(--c-amber-bg); color: var(--c-amber-text); }}
.type-badge-obligation        {{ background: var(--c-amber-bg); color: var(--c-amber-text); }}
.type-badge-claim_obligation  {{ background: var(--c-amber-bg); color: var(--c-amber-text); }}
.type-badge-limit             {{ background: var(--c-blue-bg);  color: var(--c-blue-text);  }}
.type-badge-deadline          {{ background: var(--c-amber-bg); color: var(--c-amber-text); }}
.type-badge-condition         {{ background: var(--c-cyan-bg);  color: var(--c-cyan-text);  }}

.item-page-badge {{
  font-size: 11px; color: var(--text-tertiary);
  display: flex; align-items: center; gap: 3px;
}}

.empty {{ font-size: 14px; color: var(--text-secondary); padding: 2rem 0; }}
</style>
</head>
<body>

<header>
  <div class="header-icon">{icon(ins_icon, '', 20)}</div>
  <div>
    <div class="header-title">Polisoverzicht — {doc_id}</div>
    <div class="header-sub">Visuele samenvatting van uw verzekeringsvoorwaarden (Klik op een kaart om de bronpagina te openen)</div>
  </div>
</header>

<div class="container">
  <div class="meta-bar">{chips_html}</div>
  <div class="stat-row">{stats_html}</div>
  <div class="section-tabs" id="tabs">{tabs_html}</div>
  <div id="panels">{panels_html}</div>
</div>

<script>
document.getElementById('tabs').addEventListener('click', function(e) {{
  const btn = e.target.closest('.tab-btn');
  if (!btn) return;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  const panelId = btn.dataset.panel;
  document.querySelectorAll('.panel').forEach(p => p.style.display = p.id === panelId ? '' : 'none');
}});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def read_summary_json(path: Path) -> dict[str, Any]:
    """Read a summary JSON object from disk."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def render_summary_file(
    json_path: Path,
    *,
    output_path: Path | None = None,
    pdf_url: str | None = None,
    pdf_path: Path | None = None,
    open_browser: bool = False,
) -> Path:
    """Render one saved summary JSON file to an interactive HTML page.

    If pdf_path is provided, an annotated PDF is generated with highlights
    and labels matching the item IDs shown in the HTML cards.
    """

    if not json_path.exists():
        raise FileNotFoundError(f"Summary file not found: {json_path}")

    data = read_summary_json(json_path)
    out_path = output_path or json_path.with_suffix(".html")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate annotated PDF if source PDF is available
    effective_pdf_url = pdf_url
    if pdf_path and pdf_path.exists():
        annotated_path = out_path.parent / (pdf_path.stem + "_annotated.pdf")
        try:
            annotate_from_summary(pdf_path, data, output_path=annotated_path)
            # Point the HTML to the annotated PDF instead of the original
            effective_pdf_url = annotated_path.name
        except Exception as exc:
            print(f"Warning: PDF annotation failed ({exc}), falling back to original PDF link.")

    out_path.write_text(build_html(data, pdf_url=effective_pdf_url), encoding="utf-8")

    if open_browser:
        webbrowser.open(out_path.resolve().as_uri())
    return out_path


def discover_summary_files(
    input_dir: Path,
    *,
    summary_filename: str,
    doc_ids: list[str] | None,
) -> list[Path]:
    """Find per-document summary JSON files under an output root."""

    wanted = {str(doc_id) for doc_id in doc_ids} if doc_ids else None
    paths = []
    for path in sorted(input_dir.glob(f"*/{summary_filename}")):
        if wanted is None or path.parent.name in wanted:
            paths.append(path)
    return paths


def render_summary_directory(
    input_dir: Path,
    *,
    output_dir: Path | None = None,
    doc_ids: list[str] | None = None,
    summary_filename: str = "filtered_summary_items.json",
    output_filename: str = "policy_summary.html",
    pdf_root: Path | None = None,
) -> list[Path]:
    """Render all selected per-document summary files under an output root."""

    summary_paths = discover_summary_files(
        input_dir,
        summary_filename=summary_filename,
        doc_ids=doc_ids,
    )
    if not summary_paths:
        raise FileNotFoundError(f"No {summary_filename} files found under {input_dir}")

    output_paths: list[Path] = []
    for summary_path in summary_paths:
        doc_id = summary_path.parent.name
        doc_output_dir = output_dir / doc_id if output_dir else summary_path.parent
        doc_output_dir.mkdir(parents=True, exist_ok=True)

        if output_dir:
            summary_copy = doc_output_dir / summary_filename
            if summary_copy.resolve() != summary_path.resolve():
                shutil.copy2(summary_path, summary_copy)

        pdf_url = None
        source_pdf_path = None
        if pdf_root:
            pdf_path = pdf_root / f"{doc_id}.pdf"
            if pdf_path.exists():
                source_pdf_path = pdf_path
                if output_dir:
                    pdf_copy = doc_output_dir / pdf_path.name
                    if pdf_copy.resolve() != pdf_path.resolve():
                        shutil.copy2(pdf_path, pdf_copy)
                    pdf_url = pdf_copy.name
                    source_pdf_path = pdf_copy
                else:
                    pdf_url = pdf_path.resolve().as_uri()
        output_paths.append(
            render_summary_file(
                summary_path,
                output_path=doc_output_dir / output_filename,
                pdf_url=pdf_url,
                pdf_path=source_pdf_path,
                open_browser=False,
            )
        )
    if output_dir:
        write_index_page(output_dir, output_paths)
    return output_paths


def write_index_page(output_dir: Path, output_paths: list[Path]) -> Path:
    """Write a small index page for a packaged summary output directory."""

    links = []
    for path in sorted(output_paths):
        doc_id = path.parent.name
        rel_summary = f"{doc_id}/{path.name}"
        rel_pdf = f"{doc_id}/{doc_id}.pdf"
        pdf_link = f' · <a href="{esc(rel_pdf)}" target="_blank">PDF</a>' if (output_dir / rel_pdf).exists() else ""
        links.append(f'<li><a href="{esc(rel_summary)}" target="_blank">{esc(doc_id)}</a>{pdf_link}</li>')

    html = f"""<!DOCTYPE html>
<html lang="nl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Finale Polisoverzichten</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 2rem; background: #f8f7f4; color: #1a1917; }}
h1 {{ font-size: 24px; }}
p {{ color: #5f5d57; }}
ul {{ columns: 2; line-height: 1.9; padding-left: 1.2rem; }}
a {{ color: #185fa5; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<h1>Finale Polisoverzichten</h1>
<p>Klik op een polisoverzicht en daarna op een kaart/bronclausule om de gekoppelde PDF-pagina te openen.</p>
<ul>
{''.join(links)}
</ul>
</body>
</html>"""
    index_path = output_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")
    return index_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a visual insurance policy summary.")
    parser.add_argument("json_file", nargs="?", type=Path, help="Path to filtered_summary_items.json")
    parser.add_argument("--input-dir", type=Path, default=None, help="Output root containing one folder per document")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output root for batch rendering")
    parser.add_argument("--doc-ids", nargs="+", default=None, help="Document IDs to render in batch mode")
    parser.add_argument("--all", action="store_true", dest="process_all", help="Render all documents in --input-dir")
    parser.add_argument("--summary-filename", default="filtered_summary_items.json", help="Summary JSON filename in each document folder")
    parser.add_argument("--output-filename", default="policy_summary.html", help="Output HTML filename for batch rendering")
    parser.add_argument("--pdf", "-p", default=None, help="Filename or web URL of the original policy PDF file")
    parser.add_argument("--pdf-root", type=Path, default=None, help="Folder containing PDFs named <doc_id>.pdf for batch links")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output HTML file path for single-file mode")
    parser.add_argument("--no-open", action="store_true", help="Don't open in browser automatically")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        if args.input_dir:
            if args.json_file:
                raise ValueError("Use either a positional JSON file or --input-dir, not both")
            if args.output:
                raise ValueError("--output is only valid for single-file mode; use --output-dir for batch mode")
            if args.pdf:
                raise ValueError("--pdf is only valid for single-file mode; use --pdf-root for batch mode")
            if not args.doc_ids and not args.process_all:
                raise ValueError("Specify --doc-ids or --all with --input-dir")
            output_paths = render_summary_directory(
                args.input_dir,
                output_dir=args.output_dir,
                doc_ids=None if args.process_all else args.doc_ids,
                summary_filename=args.summary_filename,
                output_filename=args.output_filename,
                pdf_root=args.pdf_root,
            )
            print(f"Generated {len(output_paths)} policy summary HTML files under {args.output_dir or args.input_dir}")
            return

        if not args.json_file:
            raise ValueError("Specify a JSON file or use --input-dir")

        out_path = render_summary_file(
            args.json_file,
            output_path=args.output,
            pdf_url=args.pdf,
            pdf_path=Path(args.pdf) if args.pdf and Path(args.pdf).exists() else None,
            open_browser=not args.no_open,
        )
        print(f"Summary written to: {out_path}")
        if not args.no_open:
            print("Opened in browser.")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
