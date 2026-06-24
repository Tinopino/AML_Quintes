"""Run LLM-based filtering over generated policy summaries.

Example:
    python -m src.pipeline.run_summary_filter --input-dir outputs/current_full_gpt4omini --doc-ids ALL26 ASR96
    python -m src.pipeline.run_summary_filter --input-dir outputs/current_full_gpt4omini --all --max-concurrent 4
    python -m src.pipeline.run_summary_filter --input-dir outputs/current_full_gpt4omini --output-dir outputs/filter_gpt4o_subset --doc-ids ALL26 --model gpt-4o --fail-on-error
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.llm.summary_builder import generate_html_report
from src.llm.summary_filter import (
    DEFAULT_FILTER_DESCRIPTION,
    DEFAULT_TARGETS,
    FilterConfig,
    config_to_json,
    dump_filter_result,
    filter_summary_with_llm,
    flatten_summary_items,
    prompt_item_record,
    select_candidate_records,
)
from src.rendering.policy_summary import render_summary_file

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def run_filter(
    *,
    input_dir: Path,
    output_dir: Path | None,
    doc_ids: list[str] | None,
    model: str,
    max_concurrent: int,
    cache_dir: Path | None,
    description: str,
    targets: dict[str, int],
    max_candidates: int,
    dry_run: bool,
    skip_html: bool,
    fail_on_error: bool,
) -> None:
    """Run summary filtering for selected document outputs."""

    summary_paths = discover_summary_paths(input_dir, doc_ids=doc_ids)
    if not summary_paths:
        raise FileNotFoundError(f"No visual_summary_items.json files found in {input_dir}")

    config = FilterConfig(model=model, description=description, targets=targets, max_candidates=max_candidates)
    output_root = output_dir or input_dir
    cache_dir = cache_dir or (output_root / "filter_cache")
    started = time.time()

    if dry_run:
        summarize_dry_run(summary_paths, config)
        return

    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []

    async def process_one(path: Path):
        async with semaphore:
            summary_doc = read_json(path)
            doc_id = str(summary_doc.get("doc_id") or path.parent.name)
            logger.info("Filtering %s", doc_id)
            result = await filter_summary_with_llm(
                summary_doc,
                config=config,
                client=client,
                cache_dir=cache_dir,
            )
            if fail_on_error and result.metadata.get("llm_error"):
                raise RuntimeError(f"Filter LLM failed for {doc_id}: {result.metadata['llm_error']}")
            doc_output_dir = output_root / doc_id if output_dir else path.parent
            if output_dir:
                doc_output_dir.mkdir(parents=True, exist_ok=True)
                write_json(doc_output_dir / "visual_summary_items.json", summary_doc)
            dump_filter_result(result, doc_output_dir)
            if not skip_html:
                filtered_path = doc_output_dir / "filtered_summary_items.json"
                generate_html_report(result.filtered_summary, doc_output_dir / "filtered_summary_report.html")
                render_summary_file(
                    filtered_path,
                    output_path=doc_output_dir / "policy_summary.html",
                    open_browser=False,
                )
            logger.info(
                "%s: %d -> %d groups (%d source items, %.1f%% reduction)",
                doc_id,
                result.metadata["input_items"],
                result.metadata["filtered_items"],
                result.metadata["selected_source_items"],
                result.metadata["reduction_ratio"] * 100,
            )
            return result.metadata

    for done, task in enumerate(asyncio.as_completed([process_one(path) for path in summary_paths]), start=1):
        metadata = await task
        results.append(metadata)
        logger.info("Progress: %d/%d", done, len(summary_paths))

    write_run_metadata(output_root, results, config, started, source_input_dir=input_dir)


def discover_summary_paths(input_dir: Path, *, doc_ids: list[str] | None) -> list[Path]:
    """Find per-document visual summary files."""

    wanted = {str(doc_id) for doc_id in doc_ids} if doc_ids else None
    paths = []
    for path in sorted(input_dir.glob("*/visual_summary_items.json")):
        if wanted is None or path.parent.name in wanted:
            paths.append(path)
    return paths


def summarize_dry_run(summary_paths: list[Path], config: FilterConfig) -> None:
    """Log prompt-size estimates without calling the API."""

    total_items = 0
    total_candidates = 0
    total_chars = 0
    largest: list[tuple[str, int, int]] = []
    for path in summary_paths:
        summary_doc = read_json(path)
        records, _ = flatten_summary_items(summary_doc)
        candidates = select_candidate_records(records, config)
        payload = json.dumps([prompt_item_record(record) for record in candidates], ensure_ascii=False)
        total_items += len(records)
        total_candidates += len(candidates)
        total_chars += len(payload)
        largest.append((path.parent.name, len(records), len(candidates), len(payload)))

    largest = sorted(largest, key=lambda item: item[2], reverse=True)[:8]
    rough_tokens = int(total_chars / 4)
    logger.info("Dry run documents: %d", len(summary_paths))
    logger.info("Input items: %d", total_items)
    logger.info("Candidate items sent to LLM: %d", total_candidates)
    logger.info("Compact payload chars: %d", total_chars)
    logger.info("Very rough input-token estimate: %d", rough_tokens)
    logger.info("Largest docs (doc, input_items, candidate_items, chars): %s", largest)
    logger.info("Filter config: %s", config_to_json(config))


def write_run_metadata(
    output_root: Path,
    results: list[dict[str, Any]],
    config: FilterConfig,
    started: float,
    *,
    source_input_dir: Path,
) -> None:
    """Write aggregate metadata for a filter run."""

    total_prompt = sum(int(result.get("usage", {}).get("prompt_tokens", 0)) for result in results)
    total_completion = sum(int(result.get("usage", {}).get("completion_tokens", 0)) for result in results)
    payload = {
        "filter_config": config_to_json(config),
        "source_input_dir": str(source_input_dir),
        "n_docs": len(results),
        "input_items": sum(int(result.get("input_items", 0)) for result in results),
        "candidate_items": sum(int(result.get("candidate_items", 0)) for result in results),
        "filtered_items": sum(int(result.get("filtered_items", 0)) for result in results),
        "selected_source_items": sum(int(result.get("selected_source_items", 0)) for result in results),
        "supplemented_items": sum(int(result.get("supplemented_items", 0)) for result in results),
        "balanced_items": sum(int(result.get("balanced_items", 0)) for result in results),
        "truncated_items": sum(int(result.get("truncated_items", 0)) for result in results),
        "avg_reduction_ratio": round(
            sum(float(result.get("reduction_ratio", 0.0)) for result in results) / len(results), 4
        ) if results else 0.0,
        "invalid_selected_id_count": sum(len(result.get("invalid_selected_ids", [])) for result in results),
        "usage": {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
        },
        "elapsed_seconds": round(time.time() - started, 1),
        "docs": results,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary_filter_run_metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    """Read JSON from disk."""

    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    """Write JSON to disk."""

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_targets(args: argparse.Namespace) -> dict[str, int]:
    """Build target dictionary from CLI arguments."""

    targets = dict(DEFAULT_TARGETS)
    if args.total_min is not None:
        targets["total_min"] = args.total_min
    if args.total_max is not None:
        targets["total_max"] = args.total_max
    return targets


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Filter visual summary items with an LLM")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--doc-ids", nargs="+", default=None)
    parser.add_argument("--all", action="store_true", dest="process_all")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--description", default=None)
    parser.add_argument("--description-file", type=Path, default=None)
    parser.add_argument("--total-min", type=int, default=None)
    parser.add_argument("--total-max", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=450)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-html", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()
    if not args.doc_ids and not args.process_all:
        parser.error("Specify --doc-ids or --all")
    return args


def resolve_description(args: argparse.Namespace) -> str:
    """Resolve filter description from flags or defaults."""

    if args.description_file:
        return args.description_file.read_text(encoding="utf-8").strip()
    if args.description:
        return args.description.strip()
    return DEFAULT_FILTER_DESCRIPTION


def main() -> None:
    args = parse_args()
    asyncio.run(
        run_filter(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            doc_ids=None if args.process_all else args.doc_ids,
            model=args.model,
            max_concurrent=args.max_concurrent,
            cache_dir=args.cache_dir,
            description=resolve_description(args),
            targets=parse_targets(args),
            max_candidates=args.max_candidates,
            dry_run=args.dry_run,
            skip_html=args.skip_html,
            fail_on_error=args.fail_on_error,
        )
    )


if __name__ == "__main__":
    main()
