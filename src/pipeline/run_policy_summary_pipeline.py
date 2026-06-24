"""End-to-end policy summary pipeline from raw PDFs.

Example:
    python -m src.pipeline.run_policy_summary_pipeline --doc-ids ALL26
    python -m src.pipeline.run_policy_summary_pipeline --all --max-concurrent 15
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from src.canonical.build_corpus import build_canonical_corpus
from src.ingestion.chunk_builder import build_structure_chunks, flatten_marker_json_directory
from src.ingestion.marker_converter import convert_pdfs_to_marker_json
from src.llm.context_builder import SectionTextConfig, build_llm_context_windows
from src.llm.extractor import extract_all_contexts, results_to_dataframe
from src.llm.summary_builder import generate_all_summaries, generate_html_report
from src.llm.validator import validate_all_extractions, enforce_column_role


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


STAGES = ("pdf", "marker", "chunks", "canonical", "llm", "summary")


def run_pipeline(
    *,
    input_dir: Path,
    output_dir: Path,
    doc_ids: list[str] | None,
    model: str,
    max_words: int,
    max_concurrent: int,
    cache_dir: Path | None,
    from_stage: str,
    skip_llm: bool,
    skip_html: bool,
    dry_run: bool,
) -> None:
    """Run the final end-to-end PDF-to-summary pipeline."""
    if from_stage not in STAGES:
        raise ValueError(f"from_stage must be one of {STAGES}")

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_dir or (output_dir / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    chunks_df: pd.DataFrame | None = None
    clauses_df: pd.DataFrame | None = None
    sections_df: pd.DataFrame | None = None
    context_windows: pd.DataFrame | None = None
    extractions_df: pd.DataFrame | None = None

    start = stage_index(from_stage)

    if start == stage_index("pdf"):
        logger.info("Running Marker PDF conversion from %s", input_dir)
        marker_json_dir = output_dir / "marker_json_outputs"
        marker_paths = convert_pdfs_to_marker_json(input_dir, marker_json_dir, doc_ids=doc_ids)
        logger.info("Marker JSON files ready: %d", len(marker_paths))

    if start <= stage_index("marker"):
        marker_json_dir = output_dir / "marker_json_outputs"
        logger.info("Flattening Marker JSON from %s", marker_json_dir)
        documents_df, pages_df, marker_blocks_df, marker_blocks_useful_df, run_log_df = flatten_marker_json_directory(
            marker_json_dir,
            doc_ids=doc_ids,
        )
        documents_df.to_csv(output_dir / "documents.csv", index=False)
        pages_df.to_csv(output_dir / "pages.csv", index=False)
        run_log_df.to_csv(output_dir / "marker_flatten_run_log.csv", index=False)
        marker_blocks_df.to_parquet(output_dir / "marker_blocks_all.parquet", index=False)
        marker_blocks_useful_df.to_parquet(output_dir / "marker_blocks_useful.parquet", index=False)
        logger.info(
            "Flattened %d documents, %d pages, %d useful Marker blocks",
            len(documents_df), len(pages_df), len(marker_blocks_useful_df),
        )

        logger.info("Building structure_chunks_enriched from Marker blocks")
        chunks_df = build_structure_chunks(marker_blocks_useful_df)
        chunks_df.to_parquet(output_dir / "structure_chunks_enriched.parquet", index=False)
        # Alias retained for the final pipeline's internal stage naming.
        chunks_df.to_parquet(output_dir / "chunks.parquet", index=False)
        chunks_df.to_csv(output_dir / "chunks.csv", index=False)
        logger.info("Built %d chunks", len(chunks_df))
    elif start == stage_index("chunks"):
        chunks_df = read_parquet_required(output_dir / "chunks.parquet")
        chunks_df = filter_doc_ids(chunks_df, doc_ids)
        logger.info("Loaded %d chunks", len(chunks_df))

    if start <= stage_index("chunks"):
        if chunks_df is None:
            chunks_df = read_parquet_required(output_dir / "chunks.parquet")
            chunks_df = filter_doc_ids(chunks_df, doc_ids)
        logger.info("Building canonical clauses and sections")
        clauses_df, sections_df = build_canonical_corpus(chunks_df, output_dir, doc_ids=doc_ids)
        logger.info("Built %d clauses and %d sections", len(clauses_df), len(sections_df))
    elif start == stage_index("canonical"):
        clauses_df = filter_doc_ids(read_parquet_required(output_dir / "canonical_clauses.parquet"), doc_ids)
        sections_df = filter_doc_ids(read_parquet_required(output_dir / "sections.parquet"), doc_ids)
        logger.info("Loaded %d clauses and %d sections", len(clauses_df), len(sections_df))

    if start <= stage_index("canonical"):
        if clauses_df is None or sections_df is None:
            clauses_df = filter_doc_ids(read_parquet_required(output_dir / "canonical_clauses.parquet"), doc_ids)
            sections_df = filter_doc_ids(read_parquet_required(output_dir / "sections.parquet"), doc_ids)

        logger.info("Building LLM context windows (max_words=%d)", max_words)
        context_windows = build_llm_context_windows(
            sections_df,
            clauses_df,
            SectionTextConfig(max_words_per_window=max_words),
        )
        context_windows.to_parquet(output_dir / "llm_context_windows.parquet", index=False)
        logger.info("Built %d LLM context windows", len(context_windows))

        if dry_run:
            estimate_cost(context_windows, model)
            write_metadata(output_dir, started, model, doc_ids, context_windows, None, None, dry_run=True)
            return

        if skip_llm:
            logger.info("Skipping LLM extraction as requested")
            write_metadata(output_dir, started, model, doc_ids, context_windows, None, None, dry_run=False)
            return

        logger.info("Running LLM extraction with %s", model)
        results = extract_all_contexts(
            context_windows_df=context_windows,
            model=model,
            cache_dir=cache_dir,
            max_concurrent=max_concurrent,
            progress_callback=progress_logger(started),
        )
        extractions_df = results_to_dataframe(results)
        raw_path = output_dir / "llm_extractions_raw.parquet"
        extractions_df.to_parquet(raw_path, index=False)
        logger.info("Saved raw LLM extractions to %s", raw_path)

        logger.info("Validating extracted quotes against source clauses")
        extractions_df = validate_all_extractions(extractions_df, clauses_df)
        extractions_df = enforce_column_role(extractions_df, clauses_df)
        extractions_df.to_parquet(output_dir / "llm_extractions.parquet", index=False)
        logger.info("Saved validated LLM extractions")
    else:
        extractions_df = filter_doc_ids(read_parquet_required(output_dir / "llm_extractions.parquet"), doc_ids)
        logger.info("Loaded %d validated extractions", len(extractions_df))

    if extractions_df is None or extractions_df.empty:
        logger.warning("No extractions available; stopping before summary generation")
        return

    logger.info("Building grouped extracted summaries")
    summaries = generate_all_summaries(extractions_df, output_dir, model=model)
    for doc_id, summary in summaries.items():
        doc_dir = output_dir / doc_id
        write_json(doc_dir / "extracted_summary.json", summary)
        if not skip_html:
            generate_html_report(summary, doc_dir / "summary_report.html")

    write_metadata(
        output_dir,
        started,
        model,
        doc_ids,
        context_windows,
        extractions_df,
        summaries,
        dry_run=False,
    )
    logger.info("Pipeline complete in %.1fs", time.time() - started)


def progress_logger(started: float):
    """Build a progress callback for LLM extraction."""
    def _progress(done: int, total: int) -> None:
        if done % 50 == 0 or done == total:
            elapsed = time.time() - started
            rate = done / elapsed if elapsed else 0.0
            eta = (total - done) / rate if rate else 0.0
            logger.info("  %d/%d windows (%.1f/s, ETA %.0fs)", done, total, rate, eta)
    return _progress


def estimate_cost(context_windows: pd.DataFrame, model: str) -> None:
    """Log a rough cost estimate."""
    total_words = int(context_windows["n_words"].sum()) if not context_windows.empty else 0
    input_tokens = int(total_words * 1.3) + len(context_windows) * 800
    output_tokens = len(context_windows) * 250
    prices = {
        "gpt-4o-mini": (0.15 / 1_000_000, 0.60 / 1_000_000),
        "gpt-4o": (2.50 / 1_000_000, 10.00 / 1_000_000),
    }
    in_price, out_price = prices.get(model, prices["gpt-4o-mini"])
    logger.info("Dry run: %d calls, estimated cost $%.2f", len(context_windows), input_tokens * in_price + output_tokens * out_price)


def write_metadata(
    output_dir: Path,
    started: float,
    model: str,
    doc_ids: list[str] | None,
    context_windows: pd.DataFrame | None,
    extractions_df: pd.DataFrame | None,
    summaries: dict[str, dict] | None,
    *,
    dry_run: bool,
) -> None:
    """Write run metadata."""
    validation_stats = {}
    if extractions_df is not None and "validation_status" in extractions_df.columns:
        validation_stats = extractions_df["validation_status"].value_counts().to_dict()
    write_json(output_dir / "pipeline_metadata.json", {
        "model": model,
        "doc_ids": doc_ids,
        "n_windows": int(len(context_windows)) if context_windows is not None else 0,
        "n_extracted": int(len(extractions_df)) if extractions_df is not None else 0,
        "n_summary_items": int(sum(s.get("total_items", 0) for s in summaries.values())) if summaries else 0,
        "validation_stats": validation_stats,
        "elapsed_seconds": round(time.time() - started, 1),
        "dry_run": dry_run,
    })


def filter_doc_ids(df: pd.DataFrame, doc_ids: list[str] | None) -> pd.DataFrame:
    """Filter a DataFrame by doc_id when requested."""
    if doc_ids is None or df.empty or "doc_id" not in df.columns:
        return df
    wanted = {str(doc_id) for doc_id in doc_ids}
    return df[df["doc_id"].astype(str).isin(wanted)].copy()


def read_parquet_required(path: Path) -> pd.DataFrame:
    """Read a required parquet file with a clear error."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_parquet(path)


def stage_index(stage: str) -> int:
    return STAGES.index(stage)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PDF-to-summary policy pipeline")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/car_policies"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/policy_summary_pipeline"))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--doc-ids", nargs="+", default=None)
    parser.add_argument("--all", action="store_true", dest="process_all")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-words", type=int, default=2500)
    parser.add_argument("--max-concurrent", type=int, default=10)
    parser.add_argument("--from-stage", choices=STAGES, default="pdf")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--skip-html", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.doc_ids and not args.process_all:
        parser.error("Specify --doc-ids or --all")
    return args


def main() -> None:
    args = parse_args()
    run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        doc_ids=None if args.process_all else args.doc_ids,
        model=args.model,
        max_words=args.max_words,
        max_concurrent=args.max_concurrent,
        cache_dir=args.cache_dir,
        from_stage=args.from_stage,
        skip_llm=args.skip_llm,
        skip_html=args.skip_html,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
