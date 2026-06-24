"""LLM-based hierarchical extraction for insurance policy sections.

Two-pass architecture:
  Pass 1 (section-level): Extract all candidate items with clause citations.
  Pass 2 (document-level): Select and rank the most important items.

Uses GPT-4o-mini by default.  Results are cached to avoid repeat API calls.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.llm.prompts import (
    PROMPT_VERSION,
    build_extraction_messages,
)

logger = logging.getLogger(__name__)

# ── Data classes ─────────────────────────────────────────────────────────

@dataclass
class ExtractedItem:
    """One item extracted by the LLM from a section."""

    item_type: str  # covered / not_covered / condition / limit / deadline / obligation / notification_duty / claim_obligation / definition / admin
    module: str
    customer_facing_headline: str | None
    exact_quote: str
    supporting_clause_ids: list[str]
    importance: int
    money_amounts: list[str] = field(default_factory=list)
    deadlines: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    theme: str | None = None
    exclusion_scope: list[str] = field(default_factory=list)

    # Populated during validation
    validation_status: str = "pending"  # pending / verified / fuzzy_match / unverified
    quote_match_score: float = 0.0
    matched_clause_ids: list[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    """Result of extracting items from one context window."""

    context_id: str
    doc_id: str
    section_id: str
    section_path: str
    items: list[ExtractedItem]
    model: str
    prompt_version: str
    usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None


# ── Cache ────────────────────────────────────────────────────────────────

def _cache_key(context_text: str, section_path: str, model: str) -> str:
    """Deterministic hash for caching LLM results."""
    payload = f"{PROMPT_VERSION}|{model}|{section_path}|{context_text}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_cache(cache_path: Path) -> dict[str, Any]:
    """Load cache from a JSON file."""
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict[str, Any], cache_path: Path) -> None:
    """Save cache to a JSON file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)


# ── OpenAI helpers ───────────────────────────────────────────────────────

def _parse_llm_response(raw: str) -> list[dict]:
    """Parse JSON from LLM response, handling markdown code fences."""
    text = raw.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    parsed = json.loads(text)

    if isinstance(parsed, dict) and "items" in parsed:
        return parsed["items"]
    if isinstance(parsed, list):
        return parsed
    return []


def _make_extracted_items(raw_items: list[dict]) -> list[ExtractedItem]:
    """Convert raw dicts from LLM into ExtractedItem instances."""
    valid_item_types = {
        "covered", "not_covered", "condition", "limit", "deadline",
        "notification_duty", "claim_obligation", "obligation", "definition", "admin",
    }
    items: list[ExtractedItem] = []
    for raw in raw_items:
        try:
            item_type = raw.get("item_type", "covered")
            if item_type not in valid_item_types:
                logger.warning("Unknown item_type '%s', defaulting to 'covered'", item_type)
                item_type = "covered"
            items.append(ExtractedItem(
                item_type=item_type,
                module=raw.get("module", "general"),
                customer_facing_headline=raw.get("customer_facing_headline"),
                exact_quote=raw.get("exact_quote", ""),
                supporting_clause_ids=raw.get("supporting_clause_ids", []),
                importance=int(raw.get("importance", 3)),
                money_amounts=raw.get("money_amounts", []),
                deadlines=raw.get("deadlines", []),
                conditions=raw.get("conditions", []),
                theme=raw.get("theme"),
                exclusion_scope=raw.get("exclusion_scope", []),
            ))
        except (ValueError, TypeError) as exc:
            logger.warning("Skipping malformed item: %s – %s", raw, exc)
    return items


# ── Async extraction (one context window) ────────────────────────────────

async def _extract_one_async(
    context_text: str,
    section_path: str,
    is_continuation: bool,
    model: str,
    temperature: float,
    max_retries: int,
    client,
) -> tuple[list[ExtractedItem], dict[str, int], str | None]:
    """Async version of extract_one_context."""
    messages = build_extraction_messages(context_text, section_path, is_continuation)

    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
                timeout=120,
            )

            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            raw_content = response.choices[0].message.content
            raw_items = _parse_llm_response(raw_content)
            items = _make_extracted_items(raw_items)

            return items, usage, None

        except json.JSONDecodeError as exc:
            logger.warning("JSON parse error (attempt %d): %s", attempt + 1, exc)
            if attempt == max_retries - 1:
                return [], {}, f"JSON parse error: {exc}"

        except Exception as exc:
            logger.warning("API error (attempt %d): %s", attempt + 1, exc)
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                return [], {}, f"API error: {exc}"

    return [], {}, "Max retries exceeded"


# ── Batch extraction (all windows for a corpus) ─────────────────────────

def extract_all_contexts(
    context_windows_df: pd.DataFrame,
    model: str = "gpt-4o-mini",
    cache_dir: Path | None = None,
    max_concurrent: int = 10,
    doc_ids: list[str] | None = None,
    progress_callback=None,
) -> list[ExtractionResult]:
    """Extract items from all context windows using concurrent API calls.

    Args:
        context_windows_df: DataFrame from build_llm_context_windows().
        model: OpenAI model name.
        cache_dir: Directory for caching results.
        max_concurrent: Maximum concurrent API calls (default 10).
        doc_ids: Optional filter to specific documents.
        progress_callback: Optional callable(done, total) for progress.

    Returns:
        List of ExtractionResult objects.
    """
    if doc_ids:
        context_windows_df = context_windows_df[
            context_windows_df["doc_id"].isin(doc_ids)
        ]

    # Load cache
    cache: dict[str, Any] = {}
    cache_path = (cache_dir or Path("data/cache")) / "llm_extraction_cache.json"
    if cache_dir:
        cache = load_cache(cache_path)
        logger.info("Loaded %d cached results", len(cache))

    # Separate cached vs uncached rows
    rows_list = list(context_windows_df.iterrows())
    total = len(rows_list)

    cached_results: dict[int, tuple] = {}  # index -> (items, usage, error)
    uncached_indices: list[int] = []

    for i, (_, row) in enumerate(rows_list):
        key = _cache_key(row["context_text"], row["section_path"], model)
        if key in cache:
            cached = cache[key]
            items = _make_extracted_items(cached.get("items", []))
            cached_results[i] = (items, cached.get("usage", {}), cached.get("error"))
        else:
            uncached_indices.append(i)

    logger.info(
        "%d/%d windows cached, %d to extract",
        len(cached_results), total, len(uncached_indices),
    )

    # Run async extraction for uncached windows
    if uncached_indices:
        api_results = _run_async_extraction(
            rows_list, uncached_indices, model, cache, cache_path,
            cache_dir, max_concurrent, progress_callback,
        )
    else:
        api_results = {}

    # Assemble results in order
    results: list[ExtractionResult] = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for i, (_, row) in enumerate(rows_list):
        if i in cached_results:
            items, usage, error = cached_results[i]
        elif i in api_results:
            items, usage, error = api_results[i]
        else:
            items, usage, error = [], {}, "Missing result"

        for k in total_usage:
            total_usage[k] += usage.get(k, 0)

        results.append(ExtractionResult(
            context_id=row["context_id"],
            doc_id=row["doc_id"],
            section_id=row["section_id"],
            section_path=row["section_path"],
            items=items,
            model=model,
            prompt_version=PROMPT_VERSION,
            usage=usage,
            error=error,
        ))

    # Final cache save
    if cache_dir:
        save_cache(cache, cache_path)

    logger.info(
        "Extraction complete: %d windows, %d items, %d errors, "
        "tokens: %d prompt + %d completion = %d total",
        total,
        sum(len(r.items) for r in results),
        sum(1 for r in results if r.error),
        total_usage["prompt_tokens"],
        total_usage["completion_tokens"],
        total_usage["total_tokens"],
    )

    return results


def _run_async_extraction(
    rows_list, uncached_indices, model, cache, cache_path,
    cache_dir, max_concurrent, progress_callback,
) -> dict[int, tuple]:
    """Run async extraction for uncached windows."""
    from openai import AsyncOpenAI

    async def _run():
        client = AsyncOpenAI()
        semaphore = asyncio.Semaphore(max_concurrent)
        api_results: dict[int, tuple] = {}
        done_count = [0]
        total_uncached = len(uncached_indices)

        async def process_one(idx: int):
            _, row = rows_list[idx]
            context_text = row["context_text"]
            section_path = row["section_path"]
            is_continuation = row.get("is_continuation", False)

            async with semaphore:
                items, usage, error = await _extract_one_async(
                    context_text=context_text,
                    section_path=section_path,
                    is_continuation=is_continuation,
                    model=model,
                    temperature=0.1,
                    max_retries=3,
                    client=client,
                )

            api_results[idx] = (items, usage, error)

            # Update cache
            key = _cache_key(context_text, section_path, model)
            cache[key] = {
                "context_id": row["context_id"],
                "items": [asdict(it) for it in items],
                "usage": usage,
                "error": error,
            }

            done_count[0] += 1
            if progress_callback:
                progress_callback(done_count[0], total_uncached)
            elif done_count[0] % 50 == 0 or done_count[0] == total_uncached:
                logger.info(
                    "  %d/%d uncached windows extracted",
                    done_count[0], total_uncached,
                )
            # Periodic cache save, also when progress is logged by the caller.
            if cache_dir and (done_count[0] % 50 == 0 or done_count[0] == total_uncached):
                save_cache(cache, cache_path)

        tasks = [process_one(idx) for idx in uncached_indices]
        await asyncio.gather(*tasks)

        return api_results

    # Run the async loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside an existing event loop (e.g., Jupyter)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = pool.submit(asyncio.run, _run()).result()
            return result
    except RuntimeError:
        pass

    return asyncio.run(_run())


# ── Results to DataFrame ─────────────────────────────────────────────────

def results_to_dataframe(results: list[ExtractionResult]) -> pd.DataFrame:
    """Flatten extraction results into a DataFrame.

    Returns DataFrame with one row per extracted item.
    """
    rows: list[dict[str, Any]] = []
    item_counter = 0

    for result in results:
        for item in result.items:
            item_counter += 1
            rows.append({
                "extraction_id": f"ext_{item_counter:06d}",
                "context_id": result.context_id,
                "doc_id": result.doc_id,
                "section_id": result.section_id,
                "section_path": result.section_path,
                "item_type": item.item_type,
                "module": item.module,
                "headline": item.customer_facing_headline,
                "exact_quote": item.exact_quote,
                "supporting_clause_ids": json.dumps(item.supporting_clause_ids),
                "importance": item.importance,
                "money_amounts": json.dumps(item.money_amounts),
                "deadlines": json.dumps(item.deadlines),
                "conditions": json.dumps(item.conditions),
                "theme": item.theme,
                "exclusion_scope": json.dumps(item.exclusion_scope),
                "validation_status": item.validation_status,
                "quote_match_score": item.quote_match_score,
                "matched_clause_ids": json.dumps(item.matched_clause_ids),
                "llm_model": result.model,
                "prompt_version": result.prompt_version,
            })

    return pd.DataFrame(rows)
