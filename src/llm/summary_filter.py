"""LLM-based filtering for high-recall policy summary items.

This stage takes the broad ``visual_summary_items.json`` output and asks an LLM
to select and merge the items that belong in a concise customer-facing summary.
The LLM only returns source item ids and display metadata.  We reconstruct the
filtered summary from the original items so provenance is preserved.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


FILTER_PROMPT_VERSION = "filter-v1.3"

DEFAULT_FILTER_DESCRIPTION = """\
Create a concise customer-facing visual summary for a Dutch car insurance policy.
The goal: a client should quickly see what they sign up for when getting in their car.

PRIORITY LOGIC (what a client MUST see):
- Priority 1 (ALWAYS keep): Coverage and exclusion status for common everyday
  damage types: diefstal, inbraak, brand, explosie, vandalisme, storm, hagel,
  ruitschade, parkeerschade, aanrijding met dier, slippen, transport, pech.
  Also: driver responsibilities the client controls (alarm, sleutels, beveiliging,
  onderhoud, rijbewijs, alcohol, drugs, toestemming). Also: core WA coverage
  (schade aan anderen). For each common damage type, keep BOTH what IS and what
  IS NOT covered.
- Priority 2 (keep when space allows): Less common scenarios, specific conditions
  that limit coverage, replacement transport, accessories, valuation rules.
- Priority 3 (only if space remains): Post-damage process details, exact
  reimbursement mechanics, administrative obligations, legal definitions,
  internal claim procedures.

Remove duplicates, but keep both the coverage AND exclusion side of the same
damage type (these are NOT duplicates). Do not silently drop complete source
modules when a useful coverage, exclusion, or limit item exists for that module.
The output should usually contain 35 to 80 grouped items.
"""

DEFAULT_TARGETS = {
    "coverage_per_module": 6,
    "common_exclusions": 14,
    "module_specific_exclusions_per_module": 10,
    "notification_duties": 5,
    "claim_obligations": 4,
    "limits": 10,
    "deadlines": 5,
    "conditions": 8,
    "total_min": 35,
    "total_max": 80,
}

BUCKET_LABELS = {
    "coverage_by_module": "coverage",
    "exclusions.common": "common exclusions",
    "exclusions.by_module": "module-specific exclusions",
    "notification_duties": "notification duties",
    "claim_obligations": "claim obligations",
    "limits_by_module": "limits",
    "deadlines": "deadlines",
    "conditions": "conditions",
}

MODULE_ORDER = {
    "wa": 0,
    "beperkt_casco": 1,
    "all_risk": 2,
    "pechhulp": 3,
    "rechtsbijstand": 4,
    "inzittenden": 5,
    "general": 6,
}

MODULE_ALIASES = {
    "other_coverage": "general",
}

BALANCE_MODULE_BUCKETS = (
    "coverage_by_module",
    "exclusions.by_module",
    "limits_by_module",
)


@dataclass
class FilterConfig:
    """Configuration for one filtering run."""

    model: str = "gpt-4o-mini"
    description: str = DEFAULT_FILTER_DESCRIPTION
    targets: dict[str, int] | None = None
    max_retries: int = 3
    temperature: float = 0.1
    max_candidates: int = 450

    def resolved_targets(self) -> dict[str, int]:
        values = dict(DEFAULT_TARGETS)
        if self.targets:
            values.update(self.targets)
        return values


@dataclass
class FilterResult:
    """Result for one filtered document."""

    doc_id: str
    filtered_summary: dict[str, Any]
    raw_llm_response: dict[str, Any]
    metadata: dict[str, Any]


def flatten_summary_items(summary_doc: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Flatten grouped summary JSON into id-addressable compact records."""

    records: list[dict[str, Any]] = []
    item_map: dict[str, dict[str, Any]] = {}
    summary = summary_doc.get("summary", {})

    def add_item(item_id: str, bucket: str, item: dict[str, Any], *, module: str | None = None) -> None:
        full_item = dict(item)
        full_item["source_item_id"] = item_id
        full_item["source_bucket"] = bucket
        if module:
            full_item["module"] = normalize_module(module)
        item_map[item_id] = full_item
        records.append(compact_item_record(item_id, bucket, full_item))

    for module, items in summary.get("coverage_by_module", {}).items():
        for idx, item in enumerate(items):
            add_item(f"coverage_by_module.{module}.{idx:03d}", "coverage_by_module", item, module=module)

    exclusions = summary.get("exclusions", {})
    for idx, item in enumerate(exclusions.get("common", [])):
        add_item(f"exclusions.common.{idx:03d}", "exclusions.common", item)

    for module, items in exclusions.get("by_module", {}).items():
        for idx, item in enumerate(items):
            add_item(f"exclusions.by_module.{module}.{idx:03d}", "exclusions.by_module", item, module=module)

    for bucket in ("notification_duties", "claim_obligations", "deadlines", "conditions"):
        for idx, item in enumerate(summary.get(bucket, [])):
            add_item(f"{bucket}.{idx:03d}", bucket, item)

    for module, items in summary.get("limits_by_module", {}).items():
        for idx, item in enumerate(items):
            add_item(f"limits_by_module.{module}.{idx:03d}", "limits_by_module", item, module=module)

    return records, item_map


def compact_item_record(item_id: str, bucket: str, item: dict[str, Any]) -> dict[str, Any]:
    """Project one item to the compact payload sent to the LLM."""

    rec = {
        "item_id": item_id,
        "bucket": bucket,
        "bucket_label": BUCKET_LABELS.get(bucket, bucket),
        "module": normalize_module(item.get("module")),
        "item_type": item.get("item_type") or "",
        "theme": item.get("theme"),
        "headline": item.get("headline") or "",
        "quote_snippet": truncate_text(item.get("exact_quote") or "", max_chars=180),
        "importance": item.get("importance", 3),
        "source_pages": item.get("source_pages", []),
        "composite_score": item.get("composite_score", 0),
        "exclusion_scope": item.get("exclusion_scope", []),
    }
    # Flag items that came from structured insurer tables (column_role override)
    if item.get("column_role_override") or item.get("from_table"):
        rec["from_table"] = True
    return rec


def build_filter_messages(
    *,
    doc_id: str,
    records: list[dict[str, Any]],
    config: FilterConfig,
) -> list[dict[str, str]]:
    """Build OpenAI chat messages for summary filtering."""

    targets = config.resolved_targets()
    system = """\
You are an expert in Dutch car insurance and customer-facing policy summaries.
You filter a high-recall extraction pool into a concise visual-summary content
set. You must not invent facts. You may only select existing item_id values.

Return strict JSON only. No markdown.
"""
    user_payload = {
        "task": "filter_policy_summary_items",
        "doc_id": doc_id,
        "filter_description": config.description,
        "targets": targets,
        "item_key": {
            "id": "source item id to select",
            "b": "bucket",
            "m": "module",
            "t": "item type",
            "th": "theme",
            "h": "headline",
            "q": "quote snippet",
            "imp": "importance 1-5",
            "score": "existing ranking score",
            "scope": "exclusion scope when present",
        },
        "rules": [
            "Select the most customer-relevant items for a concise visual summary.",
            "Merge near-duplicate or conceptually related items by listing multiple source_item_ids.",
            "If at least total_min candidate items exist, selected_groups must contain at least total_min groups.",
            "Do not return a tiny abstract. Cover each important bucket that exists in the input.",
            "ALWAYS keep exclusions for common damage types: diefstal, inbraak, brand, explosie, "
            "vandalisme, storm, hagel, ruitschade, parkeerschade, aanrijding met dier, slippen, "
            "transport, pech. These are the items clients most need to know about.",
            "ALWAYS keep driver-responsibility exclusions: alarm, sleutels, beveiliging, onderhoud, "
            "rijbewijs, alcohol, drugs, toestemming. Clients can control these.",
            "For each common damage type, keep BOTH the coverage item AND the exclusion item if both exist.",
            "Keep broad/core coverage over narrow examples.",
            "Items from structured insurer tables (from_table=true) are the insurer's own "
            "key highlights — strongly prefer keeping these.",
            "Keep duties, claim obligations, deadlines, and limits only when they have direct "
            "practical impact on the client (e.g., reporting deadlines, major deductibles).",
            "Remove administrative, legal-definition, internal-process, and low-value edge-case items.",
            "Deprioritize post-damage process details, exact reimbursement mechanics, and items "
            "that only matter after a claim has been filed.",
            "Use concise Dutch display_headline values, max 12 words.",
            "Every source_item_ids value must be from the supplied item id values.",
            "If a module has useful coverage, module-specific exclusions, or limits, "
            "keep at least one representative item for that module.",
        ],
        "output_schema": {
            "doc_id": doc_id,
            "selected_groups_count": "must be between targets.total_min and targets.total_max when enough candidates exist",
            "selected_groups": [
                {
                    "display_headline": "short Dutch customer-facing headline",
                    "bucket": "one of the input bucket values",
                    "module": "wa|beperkt_casco|all_risk|pechhulp|rechtsbijstand|inzittenden|general",
                    "item_type": "covered|not_covered|condition|limit|deadline|notification_duty|claim_obligation|obligation",
                    "theme": "theme or null",
                    "source_item_ids": ["item_id", "item_id"],
                    "visual_priority": 1,
                    "reason": "brief reason for keeping",
                }
            ],
            "rejected_item_ids": ["item_id"],
        },
        "items": [prompt_item_record(record) for record in records],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def prompt_item_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert an internal compact record to an even smaller prompt record."""

    item = {
        "id": record["item_id"],
        "b": record.get("bucket"),
        "m": record.get("module"),
        "t": record.get("item_type"),
        "h": truncate_text(record.get("headline") or "", max_chars=90),
        "q": truncate_text(record.get("quote_snippet") or "", max_chars=120),
        "imp": record.get("importance", 3),
        "score": record.get("composite_score", 0),
    }
    if record.get("theme"):
        item["th"] = record.get("theme")
    if record.get("exclusion_scope"):
        item["scope"] = record.get("exclusion_scope")
    if record.get("from_table"):
        item["from_table"] = True
    return item


async def filter_summary_with_llm(
    summary_doc: dict[str, Any],
    *,
    config: FilterConfig | None = None,
    client: Any | None = None,
    cache_dir: Path | None = None,
) -> FilterResult:
    """Filter one summary document with an LLM and reconstruct provenance."""

    config = config or FilterConfig()
    doc_id = str(summary_doc.get("doc_id") or "")
    if not doc_id:
        raise ValueError("summary_doc must contain doc_id")

    records, item_map = flatten_summary_items(summary_doc)
    if not records:
        return build_empty_filter_result(summary_doc, config)
    candidate_records = select_candidate_records(records, config)

    cache_key = build_cache_key(doc_id, candidate_records, config)
    cached = load_cached_response(cache_dir, cache_key) if cache_dir else None
    if cached is not None:
        raw_response = cached
        usage = cached.get("_usage", {}) if isinstance(cached, dict) else {}
        error = None
    else:
        if client is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI()
        raw_response, usage, error = await call_filter_llm(
            client=client,
            doc_id=doc_id,
            records=candidate_records,
            config=config,
        )
        if raw_response is None:
            raw_response = local_fallback_response(doc_id, records, config, error=error)
        raw_response["_usage"] = usage
        if cache_dir:
            save_cached_response(cache_dir, cache_key, raw_response)

    filtered_summary, metadata = reconstruct_filtered_summary(
        summary_doc=summary_doc,
        records=records,
        candidate_records=candidate_records,
        item_map=item_map,
        raw_response=raw_response,
        config=config,
        error=error,
    )
    return FilterResult(
        doc_id=doc_id,
        filtered_summary=filtered_summary,
        raw_llm_response=raw_response,
        metadata=metadata,
    )


async def call_filter_llm(
    *,
    client: Any,
    doc_id: str,
    records: list[dict[str, Any]],
    config: FilterConfig,
) -> tuple[dict[str, Any] | None, dict[str, int], str | None]:
    """Call the LLM with retries and parse strict JSON."""

    messages = build_filter_messages(doc_id=doc_id, records=records, config=config)
    usage: dict[str, int] = {}
    for attempt in range(config.max_retries):
        try:
            response = await client.chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=config.temperature,
                response_format={"type": "json_object"},
                timeout=180,
            )
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            content = response.choices[0].message.content or "{}"
            parsed = parse_json_response(content)
            too_few_message = validate_group_count(parsed, records, config)
            if too_few_message and attempt < config.max_retries - 1:
                logger.warning("Filter response for %s was too small: %s", doc_id, too_few_message)
                messages = messages + [
                    {"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)},
                    {"role": "user", "content": too_few_message},
                ]
                await asyncio.sleep(1)
                continue
            return parsed, usage, None
        except Exception as exc:  # pragma: no cover - depends on API behavior
            logger.warning("Filter LLM error for %s attempt %d: %s", doc_id, attempt + 1, exc)
            if attempt < config.max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                return None, usage, str(exc)
    return None, usage, "Max retries exceeded"


def validate_group_count(parsed: dict[str, Any], records: list[dict[str, Any]], config: FilterConfig) -> str | None:
    """Return a repair prompt when the LLM selected too few groups."""

    targets = config.resolved_targets()
    min_groups = min(int(targets["total_min"]), len(records))
    max_groups = int(targets["total_max"])
    groups = parsed.get("selected_groups") or parsed.get("selected_items") or []
    n_groups = len(groups) if isinstance(groups, list) else 0
    if len(records) >= min_groups and n_groups < min_groups:
        return (
            f"Your previous response selected only {n_groups} groups, which is invalid. "
            f"Return a replacement JSON object with between {min_groups} and {max_groups} selected_groups. "
            "Keep it concise, but include the key coverage, exclusions, notification duties, "
            "claim obligations, limits, deadlines, and major conditions that exist in the input. "
            "Use only item ids from the supplied input."
        )
    return None


def reconstruct_filtered_summary(
    *,
    summary_doc: dict[str, Any],
    records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    item_map: dict[str, dict[str, Any]],
    raw_response: dict[str, Any],
    config: FilterConfig,
    error: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate selected ids and rebuild grouped filtered summary."""

    doc_id = str(summary_doc.get("doc_id") or raw_response.get("doc_id") or "")
    record_ids = {record["item_id"] for record in records}
    selected_groups = raw_response.get("selected_groups") or raw_response.get("selected_items") or []
    if not isinstance(selected_groups, list):
        selected_groups = []

    filtered_groups: list[dict[str, Any]] = []
    invalid_ids: list[str] = []
    selected_source_ids: set[str] = set()
    supplemented_count = 0
    balanced_count = 0
    truncated_count = 0

    for group in selected_groups:
        if not isinstance(group, dict):
            continue
        source_ids = group.get("source_item_ids") or group.get("item_ids") or []
        if isinstance(source_ids, str):
            source_ids = [source_ids]
        valid_ids = [str(item_id) for item_id in source_ids if str(item_id) in item_map and str(item_id) not in selected_source_ids]
        invalid_ids.extend(str(item_id) for item_id in source_ids if str(item_id) not in item_map)
        if not valid_ids:
            continue
        selected_source_ids.update(valid_ids)
        filtered_groups.append(build_filtered_entry(group, valid_ids, item_map))

    if not filtered_groups:
        fallback = local_fallback_response(doc_id, records, config, error="No valid selected groups")
        return reconstruct_filtered_summary(
            summary_doc=summary_doc,
            records=records,
            candidate_records=candidate_records,
            item_map=item_map,
            raw_response=fallback,
            config=config,
            error=error or "No valid selected groups",
        )

    targets = config.resolved_targets()
    min_groups = min(int(targets["total_min"]), len(records))
    if len(filtered_groups) < min_groups:
        supplemented_count = supplement_minimum_groups(
            filtered_groups=filtered_groups,
            selected_source_ids=selected_source_ids,
            records=records,
            item_map=item_map,
            min_groups=min_groups,
        )

    max_groups = min(int(targets["total_max"]), len(records))
    if len(filtered_groups) < max_groups:
        balanced_count = supplement_missing_module_groups(
            filtered_groups=filtered_groups,
            selected_source_ids=selected_source_ids,
            records=records,
            item_map=item_map,
            max_groups=max_groups,
        )

    if len(filtered_groups) > max_groups:
        truncated_count = len(filtered_groups) - max_groups
        filtered_groups = sorted(filtered_groups, key=filtered_sort_key)[:max_groups]
        selected_source_ids = selected_ids_from_groups(filtered_groups)

    grouped_summary = group_filtered_entries(filtered_groups)
    item_counts = count_filtered_items(grouped_summary)
    rejected_ids = sorted(record_ids - selected_source_ids)

    filtered_doc = {
        "doc_id": doc_id,
        "insurance_type": summary_doc.get("insurance_type", "car"),
        "source_model": summary_doc.get("model"),
        "filter_model": config.model,
        "filter_prompt_version": FILTER_PROMPT_VERSION,
        "filter_description": config.description,
        "summary": grouped_summary,
        "module_labels": summary_doc.get("module_labels", {}),
        "item_counts": item_counts,
        "total_items": sum(item_counts.values()),
        "input_total_items": len(records),
        "selected_source_items": len(selected_source_ids),
        "balanced_items": balanced_count,
        "truncated_items": truncated_count,
        "rejected_item_ids": rejected_ids,
    }

    metadata = {
        "doc_id": doc_id,
        "filter_prompt_version": FILTER_PROMPT_VERSION,
        "filter_model": config.model,
        "input_items": len(records),
        "candidate_items": len(candidate_records),
        "filtered_items": int(filtered_doc["total_items"]),
        "selected_source_items": len(selected_source_ids),
        "supplemented_items": supplemented_count,
        "balanced_items": balanced_count,
        "truncated_items": truncated_count,
        "reduction_ratio": round(1 - (filtered_doc["total_items"] / len(records)), 4) if records else 0.0,
        "invalid_selected_ids": sorted(set(invalid_ids)),
        "usage": raw_response.get("_usage", {}),
        "llm_error": error,
        "targets": config.resolved_targets(),
        "item_counts": item_counts,
    }
    return filtered_doc, metadata


def select_candidate_records(records: list[dict[str, Any]], config: FilterConfig) -> list[dict[str, Any]]:
    """Prefilter oversized documents to a balanced candidate set for the LLM."""

    if len(records) <= config.max_candidates:
        return records

    targets = config.resolved_targets()
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add_candidates(candidates: list[dict[str, Any]], limit: int) -> None:
        for record in sorted(candidates, key=record_rank_key)[:limit]:
            if record["item_id"] not in selected_ids:
                selected.append(record)
                selected_ids.add(record["item_id"])

    modules = sorted({record.get("module") or "general" for record in records})
    for module in modules:
        add_candidates(
            [record for record in records if record["bucket"] == "coverage_by_module" and record.get("module") == module],
            targets["coverage_per_module"] * 3,
        )
        add_candidates(
            [record for record in records if record["bucket"] == "exclusions.by_module" and record.get("module") == module],
            targets["module_specific_exclusions_per_module"] * 3,
        )

    bucket_limits = {
        "exclusions.common": targets["common_exclusions"] * 3,
        "notification_duties": targets["notification_duties"] * 3,
        "claim_obligations": targets["claim_obligations"] * 3,
        "limits_by_module": targets["limits"] * 3,
        "deadlines": targets["deadlines"] * 3,
        "conditions": targets["conditions"] * 3,
    }
    for bucket, limit in bucket_limits.items():
        add_candidates([record for record in records if record["bucket"] == bucket], limit)

    remaining_slots = max(config.max_candidates - len(selected), 0)
    if remaining_slots:
        add_candidates([record for record in records if record["item_id"] not in selected_ids], remaining_slots)

    return selected[: config.max_candidates]


def supplement_minimum_groups(
    *,
    filtered_groups: list[dict[str, Any]],
    selected_source_ids: set[str],
    records: list[dict[str, Any]],
    item_map: dict[str, dict[str, Any]],
    min_groups: int,
) -> int:
    """Add deterministic ranked items when the LLM returns too few groups."""

    added = 0
    for record in sorted(records, key=record_rank_key):
        if len(filtered_groups) >= min_groups:
            break
        item_id = record["item_id"]
        if item_id in selected_source_ids:
            continue
        group = group_from_record(record, reason="local_minimum_supplement")
        filtered_groups.append(build_filtered_entry(group, [item_id], item_map))
        selected_source_ids.add(item_id)
        added += 1
    return added


def supplement_missing_module_groups(
    *,
    filtered_groups: list[dict[str, Any]],
    selected_source_ids: set[str],
    records: list[dict[str, Any]],
    item_map: dict[str, dict[str, Any]],
    max_groups: int,
) -> int:
    """Add one ranked representative for missing core source modules.

    The LLM remains responsible for the main filtering decision. This post-pass
    only prevents a concise summary from dropping an entire module when the
    high-recall input has customer-facing coverage, exclusion, or limit evidence.
    """

    added = 0
    for bucket in BALANCE_MODULE_BUCKETS:
        available_modules = sorted(
            {
                normalize_module(record.get("module"))
                for record in records
                if record.get("bucket") == bucket
            },
            key=module_sort_key,
        )
        selected_modules = {
            normalize_module(entry.get("module"))
            for entry in filtered_groups
            if source_bucket_from_ids(entry.get("source_item_ids", [])) == bucket
        }
        for module in available_modules:
            if len(filtered_groups) >= max_groups:
                return added
            if module in selected_modules:
                continue
            candidate = first_ranked_unselected(
                records,
                bucket=bucket,
                module=module,
                selected_source_ids=selected_source_ids,
            )
            if candidate is None:
                continue
            item_id = candidate["item_id"]
            group = group_from_record(candidate, reason="local_module_balance_supplement")
            filtered_groups.append(build_filtered_entry(group, [item_id], item_map))
            selected_source_ids.add(item_id)
            selected_modules.add(module)
            added += 1
    return added


def first_ranked_unselected(
    records: list[dict[str, Any]],
    *,
    bucket: str,
    module: str,
    selected_source_ids: set[str],
) -> dict[str, Any] | None:
    """Return the best unselected record for a source bucket/module pair."""

    candidates = [
        record
        for record in records
        if record.get("bucket") == bucket
        and normalize_module(record.get("module")) == module
        and record["item_id"] not in selected_source_ids
    ]
    if not candidates:
        return None
    return sorted(candidates, key=record_rank_key)[0]


def build_filtered_entry(
    group: dict[str, Any],
    valid_ids: list[str],
    item_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build one merged filtered entry from original source items."""

    source_items = [item_map[item_id] for item_id in valid_ids]
    base = source_items[0]
    source_modules = unique_in_order(normalize_module(item.get("module")) for item in source_items)
    module = (
        source_modules[0]
        if len(source_modules) == 1
        else str(group.get("module") or base.get("module") or "general")
    )
    pages = sorted({int(page) for item in source_items for page in item.get("source_pages", []) if str(page).isdigit()})
    chunk_ids = unique_in_order(
        str(chunk_id)
        for item in source_items
        for chunk_id in item.get("chunk_ids", [])
        if chunk_id
    )
    quotes = unique_in_order(str(item.get("exact_quote") or "") for item in source_items if item.get("exact_quote"))
    exclusion_scope = unique_in_order(
        str(scope)
        for item in source_items
        for scope in item.get("exclusion_scope", [])
        if scope
    )
    return {
        "headline": str(group.get("display_headline") or base.get("headline") or ""),
        "exact_quote": quotes[0] if quotes else "",
        "supporting_quotes": quotes,
        "module": module,
        "item_type": str(group.get("item_type") or base.get("item_type") or ""),
        "theme": group.get("theme", base.get("theme")),
        "icon_hint": base.get("icon_hint"),
        "importance": max(int_or_default(item.get("importance"), 3) for item in source_items),
        "visual_priority": int_or_default(group.get("visual_priority"), 3),
        "source_pages": pages,
        "chunk_ids": chunk_ids,
        "source_item_ids": valid_ids,
        "merged_count": len(valid_ids),
        "composite_score": max(float_or_default(item.get("composite_score"), 0.0) for item in source_items),
        "exclusion_scope": exclusion_scope,
        "filter_reason": str(group.get("reason") or ""),
    }


def group_filtered_entries(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Place filtered entries back into the visual summary bucket structure."""

    summary: dict[str, Any] = {
        "coverage_by_module": {},
        "exclusions": {"common": [], "by_module": {}},
        "notification_duties": [],
        "claim_obligations": [],
        "limits_by_module": {},
        "deadlines": [],
        "conditions": [],
    }

    for entry in sorted(entries, key=filtered_sort_key):
        item_type = entry.get("item_type")
        module = entry.get("module") or "general"
        source_bucket = source_bucket_from_ids(entry.get("source_item_ids", []))
        if item_type == "covered" or source_bucket == "coverage_by_module":
            summary["coverage_by_module"].setdefault(module, []).append(entry)
        elif item_type == "not_covered" or source_bucket.startswith("exclusions"):
            if source_bucket == "exclusions.common" or module == "general" or len(entry.get("exclusion_scope", [])) > 1:
                summary["exclusions"]["common"].append(entry)
            else:
                summary["exclusions"]["by_module"].setdefault(module, []).append(entry)
        elif item_type in {"notification_duty", "obligation"} or source_bucket == "notification_duties":
            summary["notification_duties"].append(entry)
        elif item_type == "claim_obligation" or source_bucket == "claim_obligations":
            summary["claim_obligations"].append(entry)
        elif item_type == "limit" or source_bucket == "limits_by_module":
            summary["limits_by_module"].setdefault(module, []).append(entry)
        elif item_type == "deadline" or source_bucket == "deadlines":
            summary["deadlines"].append(entry)
        elif item_type == "condition" or source_bucket == "conditions":
            summary["conditions"].append(entry)
        else:
            summary["conditions"].append(entry)

    return summary


def count_filtered_items(summary: dict[str, Any]) -> dict[str, int]:
    """Count filtered items in the standard bucket layout."""

    return {
        "covered": sum(len(items) for items in summary.get("coverage_by_module", {}).values()),
        "exclusions_common": len(summary.get("exclusions", {}).get("common", [])),
        "exclusions_module_specific": sum(len(items) for items in summary.get("exclusions", {}).get("by_module", {}).values()),
        "notification_duties": len(summary.get("notification_duties", [])),
        "claim_obligations": len(summary.get("claim_obligations", [])),
        "limits": sum(len(items) for items in summary.get("limits_by_module", {}).values()),
        "deadlines": len(summary.get("deadlines", [])),
        "conditions": len(summary.get("conditions", [])),
    }


def selected_ids_from_groups(groups: list[dict[str, Any]]) -> set[str]:
    """Collect selected source item ids after local group pruning."""

    return {
        str(item_id)
        for group in groups
        for item_id in group.get("source_item_ids", [])
        if item_id
    }


def local_fallback_response(doc_id: str, records: list[dict[str, Any]], config: FilterConfig, *, error: str | None) -> dict[str, Any]:
    """Deterministic fallback if the LLM call fails or returns no valid ids."""

    targets = config.resolved_targets()
    selected: list[dict[str, Any]] = []

    def choose(bucket: str, limit: int, *, module: str | None = None) -> None:
        candidates = [record for record in records if record["bucket"] == bucket]
        if module:
            candidates = [record for record in candidates if record.get("module") == module]
        candidates = sorted(candidates, key=record_rank_key)[:limit]
        for record in candidates:
            selected.append(group_from_record(record, reason="fallback_ranked_selection"))

    modules = sorted({record.get("module") or "general" for record in records})
    for module in modules:
        choose("coverage_by_module", targets["coverage_per_module"], module=module)
        choose("exclusions.by_module", targets["module_specific_exclusions_per_module"], module=module)
    choose("exclusions.common", targets["common_exclusions"])
    choose("notification_duties", targets["notification_duties"])
    choose("claim_obligations", targets["claim_obligations"])
    choose("limits_by_module", targets["limits"])
    choose("deadlines", targets["deadlines"])
    choose("conditions", targets["conditions"])

    max_total = targets["total_max"]
    selected = sorted(selected, key=lambda group: int_or_default(group.get("visual_priority"), 3))[:max_total]
    selected_ids = {item_id for group in selected for item_id in group.get("source_item_ids", [])}
    return {
        "doc_id": doc_id,
        "selected_groups": selected,
        "rejected_item_ids": [record["item_id"] for record in records if record["item_id"] not in selected_ids],
        "fallback_error": error,
    }


def group_from_record(record: dict[str, Any], *, reason: str) -> dict[str, Any]:
    """Create one selected group from one compact record."""

    return {
        "display_headline": record.get("headline") or record.get("quote_snippet") or "",
        "bucket": record.get("bucket"),
        "module": record.get("module"),
        "item_type": record.get("item_type"),
        "theme": record.get("theme"),
        "source_item_ids": [record["item_id"]],
        "visual_priority": visual_priority(record),
        "reason": reason,
    }


def record_rank_key(record: dict[str, Any]) -> tuple[int, float, str]:
    """Sort compact records by customer relevance."""

    return (
        visual_priority(record),
        -float_or_default(record.get("composite_score"), 0.0),
        str(record.get("headline") or ""),
    )


def normalize_module(value: Any) -> str:
    """Normalize missing module labels to the generic module."""

    module = str(value or "general")
    return MODULE_ALIASES.get(module, module)


def module_sort_key(module: str) -> tuple[int, str]:
    """Sort known insurance modules before unknown labels."""

    return (MODULE_ORDER.get(module, len(MODULE_ORDER)), module)


def visual_priority(record: dict[str, Any]) -> int:
    """Convert extraction importance/type into a visual priority rank."""

    item_type = str(record.get("item_type") or "")
    importance = int_or_default(record.get("importance"), 3)
    if item_type in {"covered", "not_covered"} and importance >= 5:
        return 1
    if item_type in {"limit", "deadline", "notification_duty", "claim_obligation"} and importance >= 4:
        return 2
    if importance >= 4:
        return 3
    return 4


def build_empty_filter_result(summary_doc: dict[str, Any], config: FilterConfig) -> FilterResult:
    """Return an empty filter result for empty source summaries."""

    doc_id = str(summary_doc.get("doc_id") or "")
    filtered_summary = {
        "doc_id": doc_id,
        "insurance_type": summary_doc.get("insurance_type", "car"),
        "source_model": summary_doc.get("model"),
        "filter_model": config.model,
        "filter_prompt_version": FILTER_PROMPT_VERSION,
        "summary": group_filtered_entries([]),
        "module_labels": summary_doc.get("module_labels", {}),
        "item_counts": count_filtered_items(group_filtered_entries([])),
        "total_items": 0,
        "input_total_items": 0,
        "selected_source_items": 0,
        "balanced_items": 0,
        "truncated_items": 0,
        "rejected_item_ids": [],
    }
    metadata = {
        "doc_id": doc_id,
        "filter_prompt_version": FILTER_PROMPT_VERSION,
        "filter_model": config.model,
        "input_items": 0,
        "candidate_items": 0,
        "filtered_items": 0,
        "selected_source_items": 0,
        "supplemented_items": 0,
        "balanced_items": 0,
        "truncated_items": 0,
        "reduction_ratio": 0.0,
        "invalid_selected_ids": [],
        "usage": {},
        "llm_error": None,
        "targets": config.resolved_targets(),
        "item_counts": filtered_summary["item_counts"],
    }
    return FilterResult(doc_id=doc_id, filtered_summary=filtered_summary, raw_llm_response={}, metadata=metadata)


def build_cache_key(doc_id: str, records: list[dict[str, Any]], config: FilterConfig) -> str:
    """Build a stable cache key for a filter request."""

    payload = {
        "version": FILTER_PROMPT_VERSION,
        "doc_id": doc_id,
        "model": config.model,
        "description": config.description,
        "targets": config.resolved_targets(),
        "records": records,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return digest[:24]


def load_cached_response(cache_dir: Path | None, cache_key: str) -> dict[str, Any] | None:
    """Load a cached raw LLM response."""

    if cache_dir is None:
        return None
    path = cache_dir / f"{cache_key}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_cached_response(cache_dir: Path, cache_key: str, response: dict[str, Any]) -> None:
    """Save a raw LLM response to cache."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{cache_key}.json").write_text(
        json.dumps(response, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_json_response(raw: str) -> dict[str, Any]:
    """Parse JSON from an LLM response."""

    text = raw.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Filter response must be a JSON object")
    return parsed


def source_bucket_from_ids(source_item_ids: list[str]) -> str:
    """Infer the source bucket from the first source item id."""

    if not source_item_ids:
        return ""
    item_id = str(source_item_ids[0])
    if item_id.startswith("coverage_by_module."):
        return "coverage_by_module"
    if item_id.startswith("exclusions.common."):
        return "exclusions.common"
    if item_id.startswith("exclusions.by_module."):
        return "exclusions.by_module"
    if item_id.startswith("limits_by_module."):
        return "limits_by_module"
    return item_id.rsplit(".", 1)[0]


def filtered_sort_key(entry: dict[str, Any]) -> tuple[int, int, float, str]:
    """Sort filtered entries by visual priority and source score."""

    return (
        int_or_default(entry.get("visual_priority"), 3),
        -int_or_default(entry.get("importance"), 3),
        -float_or_default(entry.get("composite_score"), 0.0),
        str(entry.get("headline") or ""),
    )


def truncate_text(value: str, *, max_chars: int) -> str:
    """Truncate text for compact prompts."""

    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def unique_in_order(values) -> list[str]:
    """Return unique non-empty values in first-seen order."""

    return list(dict.fromkeys(str(value) for value in values if str(value or "").strip()))


def int_or_default(value: Any, default: int) -> int:
    """Convert to int with fallback."""

    try:
        return int(value)
    except Exception:
        return default


def float_or_default(value: Any, default: float) -> float:
    """Convert to float with fallback."""

    try:
        return float(value)
    except Exception:
        return default


def dump_filter_result(result: FilterResult, doc_dir: Path) -> None:
    """Write filtered summary, metadata, and raw response for one document."""

    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / "filtered_summary_items.json").write_text(
        json.dumps(result.filtered_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (doc_dir / "summary_filter_metadata.json").write_text(
        json.dumps(result.metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (doc_dir / "summary_filter_llm_response.json").write_text(
        json.dumps(result.raw_llm_response, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def config_to_json(config: FilterConfig) -> dict[str, Any]:
    """Serialize filter config."""

    data = asdict(config)
    data["targets"] = config.resolved_targets()
    data["prompt_version"] = FILTER_PROMPT_VERSION
    return data
