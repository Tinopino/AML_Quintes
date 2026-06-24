"""Run Marker PDF conversion and write one JSON file per policy PDF."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


MARKER_CONFIG = {
    "output_format": "json",
    "disable_image_extraction": True,
}


def convert_pdfs_to_marker_json(
    input_dir: Path,
    json_dir: Path,
    *,
    doc_ids: Iterable[str] | None = None,
    skip_existing: bool = True,
) -> list[Path]:
    """Convert selected PDFs to Marker JSON files.

    This mirrors the Marker setup used in ``preprocessingaml.py``.
    """
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

    wanted = {str(doc_id) for doc_id in doc_ids} if doc_ids else None
    pdf_paths = sorted(
        path for path in input_dir.iterdir()
        if path.suffix.lower() == ".pdf" and (wanted is None or path.stem in wanted)
    )
    if not pdf_paths:
        raise FileNotFoundError(f"No matching PDFs found in {input_dir}")

    json_dir.mkdir(parents=True, exist_ok=True)
    existing_paths = [json_dir / f"{pdf_path.stem}.json" for pdf_path in pdf_paths]
    if skip_existing and all(path.exists() for path in existing_paths):
        for pdf_path in pdf_paths:
            logger.info("Marker JSON exists, skipping %s", pdf_path.name)
        return existing_paths

    try:
        from marker.config.parser import ConfigParser
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise ImportError(
            "marker-pdf is required for the production preprocessing path. "
            "Install with `pip install marker-pdf accelerate`."
        ) from exc

    config_parser = ConfigParser(MARKER_CONFIG)
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )

    output_paths: list[Path] = []
    for pdf_path in pdf_paths:
        out_path = json_dir / f"{pdf_path.stem}.json"
        if skip_existing and out_path.exists():
            logger.info("Marker JSON exists, skipping %s", pdf_path.name)
            output_paths.append(out_path)
            continue
        logger.info("Running Marker on %s", pdf_path.name)
        rendered = converter(str(pdf_path))
        rendered_obj = to_python_obj(rendered)
        out_path.write_text(
            json.dumps(rendered_obj, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        output_paths.append(out_path)
    return output_paths


def to_python_obj(value: Any) -> Any:
    """Convert Marker/Pydantic objects into plain Python JSON values."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [to_python_obj(item) for item in value]
    if isinstance(value, dict):
        return {key: to_python_obj(item) for key, item in value.items()}
    return value
