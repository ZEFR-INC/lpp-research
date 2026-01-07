"""Inference script for LLM uncertainty meta-model.

This module loads a persisted meta-model (saved via the training pipeline) and
performs inference on new LLM outputs containing token-level logprobs.

Input Expectations
------------------
The input file can be one of:
    1. JSONL: each line is a JSON object with at minimum:
        {
            "content_id": "abc123",
            "top_logprobs": [ {"token": "YES", "logprob": -0.12}, ... ]
        }
    2. JSON: either a list of such objects OR a list produced by training
       preprocessing (records containing a "logprobs_data" list). If the
       training-style records are used, they must include a field
       "content_id" at the top level.
    3. CSV: must contain columns: content_id, top_logprobs (stringified list)

Required fields per record:
    - content_id (string or int convertible to string)
    - Either:
        * top_logprobs: list[dict] OR JSON string representing it
        * logprobs_data: list with first element containing key "top_logprobs"

The script will normalize records so the feature engineering pipeline can run:
    normalized_record = {"logprobs_data": [{"top_logprobs": <list>}]}.

Model Metadata
--------------
The training pipeline persists two files side by side:
    best_meta_model.pkl   (pickle of sklearn Pipeline)
    best_meta_model.json  (metadata with selected_features & optimal_threshold)

We load both. Features are extracted with `create_feature_pipeline(include_filtered=True)`.
Only the features listed in metadata["selected_features"] are passed into the
model; any missing required feature columns are created with value 0.0 and a
warning is logged.

CLI Usage
---------
    python -m src.inference \
        --model-path models/best_meta_model.pkl \
        --input-file new_samples.jsonl \
        --output-file predictions.csv \
        --output-format csv

Optional arguments:
    --threshold <float> : override metadata optimal threshold.
    --max-records <int> : limit number of records for quick smoke runs.
    --verbose            : enable debug-level logging.

Output
------
CSV/JSON containing columns:
    content_id, meta_model_probability, meta_model_prediction, threshold_used

Exit code is 0 on success and >0 on failures.
"""

from __future__ import annotations

import argparse
import json
import pandas as pd
import structlog
import sys

from feature_engineering import create_feature_pipeline
from meta_model import MetaModelPipeline
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = structlog.getLogger(__name__)


# ==================== Data Loading & Normalization ====================


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
                else:
                    logger.warning(
                        "Skipping non-dict JSONL line",
                        line_no=line_no,
                        content=line[:80],
                    )  # pragma: no cover (covered logically, line unmapped)
            except json.JSONDecodeError as e:
                logger.warning(
                    "Malformed JSONL line", line_no=line_no, error=str(e)
                )  # pragma: no cover
    return records


def _read_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    if isinstance(data, dict):
        return [data]
    logger.warning("JSON root not list/dict; returning empty")  # pragma: no cover
    return []  # pragma: no cover


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    df = pd.read_csv(path)
    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rec: Dict[str, Any] = {"content_id": row.get("content_id")}
        top_logprobs_raw = row.get("top_logprobs")
        if isinstance(top_logprobs_raw, str):
            try:
                rec["top_logprobs"] = json.loads(top_logprobs_raw)
            except json.JSONDecodeError:
                # attempt to parse comma-separated logprob placeholder -> not token-level
                rec["top_logprobs"] = []
        else:
            rec["top_logprobs"] = top_logprobs_raw
        records.append(rec)
    return records


def load_input_file(path: Path) -> List[Dict[str, Any]]:
    """Load inference input file into list of raw records.

    Supports JSONL, JSON, CSV.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl(path)
    if path.suffix.lower() == ".json":
        return _read_json(path)
    if path.suffix.lower() == ".csv":
        return _read_csv(path)
    raise ValueError(f"Unsupported input file extension: {path.suffix}")


# ==================== Validation & Normalization ====================


REQUIRED_TOP_KEYS = {"token", "logprob"}


def _extract_top_logprobs(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract top_logprobs list from flexible raw record formats."""
    if "top_logprobs" in raw and isinstance(raw["top_logprobs"], list):
        return [d for d in raw["top_logprobs"] if isinstance(d, dict)]
    # Training format: logprobs_data -> [{"top_logprobs": [...]}, ...]
    if "logprobs_data" in raw and isinstance(raw["logprobs_data"], list):
        first = raw["logprobs_data"][0] if raw["logprobs_data"] else {}
        if isinstance(first, dict):
            tlp = first.get("top_logprobs", [])
            if isinstance(tlp, list):
                return [d for d in tlp if isinstance(d, dict)]
    return []


def validate_record(raw: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate a single raw record.

    Returns (is_valid, reason_if_invalid).
    """
    if "content_id" not in raw:
        return False, "missing content_id"
    top = _extract_top_logprobs(raw)
    if not top:
        return False, "missing/empty top_logprobs"
    # Basic structural validation
    for item in top[:2]:  # check first couple
        if not REQUIRED_TOP_KEYS.issubset(item.keys()):
            return False, "top_logprobs items missing required keys"
    return True, ""


def normalize_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize raw record into feature-engineering compatible structure."""
    return {
        "content_id": str(raw.get("content_id")),
        "logprobs_data": [{"top_logprobs": _extract_top_logprobs(raw)}],
    }


def prepare_records(
    raw_records: List[Dict[str, Any]], max_records: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Validate and normalize all records.

    Returns (normalized_records, skipped_reasons).
    """
    normalized: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for idx, raw in enumerate(raw_records):
        if max_records is not None and len(normalized) >= max_records:
            break
        is_valid, reason = validate_record(raw)
        if not is_valid:
            skipped.append(f"index={idx} reason={reason}")
            continue
        normalized.append(normalize_record(raw))
    return normalized, skipped


# ==================== Feature Extraction & Alignment ====================


def extract_features(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Run feature engineering on normalized records.

    Args:
        records: List of normalized records each containing logprobs_data.

    Returns:
        DataFrame of extracted features (sorted by column name for deterministic order).
    """
    feature_pipeline = create_feature_pipeline(include_filtered=True)
    features_df = feature_pipeline.transform(records)
    # Ensure deterministic column ordering
    return features_df.reindex(sorted(features_df.columns), axis=1)


def align_features(features_df: pd.DataFrame, required: List[str]) -> pd.DataFrame:
    """Ensure all required feature columns exist; add missing with zeros.

    Preserves order of `required`.
    """
    missing = [c for c in required if c not in features_df.columns]
    if missing:
        logger.warning(
            "Adding missing feature columns with zeros",
            missing_count=len(missing),
            missing=missing,
        )
        for m in missing:
            features_df[m] = 0.0
    return features_df[required]


# ==================== Model Loading & Prediction ====================


def load_model_and_metadata(model_path: Path) -> Tuple[Any, Dict[str, Any]]:
    """Load persisted model pipeline and its metadata JSON.

    Returns (model, metadata_dict).
    """
    meta_pipeline = MetaModelPipeline()  # default config just for load helper
    model = meta_pipeline.load_model(model_path)
    metadata_path = model_path.with_suffix(".json")
    if not metadata_path.exists():
        logger.warning("Metadata JSON not found; proceeding without selected_features")
        metadata = {"selected_features": [], "optimal_threshold": 0.5}
    else:
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    return model, metadata


def predict(model: Any, features_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Generate probability and binary predictions from model.

    Args:
        model: Fitted sklearn Pipeline or estimator.
        features_df: Aligned feature DataFrame.
        threshold: Decision threshold for converting probabilities to binary predictions.

    Returns:
        DataFrame with columns meta_model_probability and meta_model_prediction.
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features_df)[:, 1]
    else:
        # fallback: derive probability-like scores
        if hasattr(model, "decision_function"):
            scores = model.decision_function(features_df)
            # min-max scale to [0,1]
            mn, mx = scores.min(), scores.max()
            probs = (scores - mn) / (mx - mn + 1e-12)
        else:
            preds = model.predict(features_df)
            probs = preds.astype(float)
    preds_binary = (probs >= threshold).astype(int)
    return pd.DataFrame(
        {
            "meta_model_probability": probs,
            "meta_model_prediction": preds_binary,
        }
    )


# ==================== Orchestration ====================


def run_inference(
    model_path: Path,
    input_path: Path,
    output_path: Path,
    threshold_override: Optional[float] = None,
    output_format: str = "csv",
    max_records: Optional[int] = None,
) -> pd.DataFrame:
    """Execute full inference workflow.

    Returns predictions DataFrame.
    """
    logger.info("Loading model", model_path=str(model_path))
    model, metadata = load_model_and_metadata(model_path)
    selected_features: List[str] = metadata.get("selected_features", [])
    optimal_threshold: float = metadata.get("optimal_threshold", 0.5)
    threshold = (
        threshold_override if threshold_override is not None else optimal_threshold
    )
    logger.info(
        "Model metadata loaded",
        selected_features_count=len(selected_features),
        threshold=threshold,
    )

    logger.info("Loading input data", input_file=str(input_path))
    raw_records = load_input_file(input_path)
    normalized_records, skipped = prepare_records(raw_records, max_records=max_records)
    logger.info(
        "Input validation complete",
        total=len(raw_records),
        valid=len(normalized_records),
        skipped=len(skipped),
    )
    if skipped:
        logger.warning(
            "Skipped records", details=skipped[:10], truncated=len(skipped) > 10
        )
    if not normalized_records:
        raise ValueError("No valid records for inference after validation")

    features_df = extract_features(normalized_records)
    if selected_features:
        features_df = align_features(features_df, selected_features)
    else:
        logger.warning("No selected_features provided; using all extracted features")

    preds_df = predict(model, features_df, threshold)
    content_ids = [r["content_id"] for r in normalized_records]
    preds_df.insert(0, "content_id", content_ids)
    preds_df["threshold_used"] = threshold

    # Persist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format.lower() == "csv":
        preds_df.to_csv(output_path, index=False)
    elif output_format.lower() == "json":
        preds_df.to_json(output_path, orient="records", indent=2)
    else:
        raise ValueError(f"Unsupported output_format: {output_format}")
    logger.info(
        "Inference complete",
        output=str(output_path),
        predictions=len(preds_df),
        positive=(preds_df["meta_model_prediction"] == 1).sum(),
    )
    return preds_df


# ==================== CLI ====================


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run meta-model inference")
    parser.add_argument(
        "--model-path",
        default=Path("models/best_meta_model.pkl"),
        type=Path,
        help="Path to best_meta_model.pkl",
    )
    parser.add_argument(
        "--input-file",
        default=Path("preprocessed_samples.json"),
        type=Path,
        help="Path to input JSONL/JSON/CSV file",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("results/predictions.csv"),
        help="Where to save predictions",
    )
    parser.add_argument(
        "--output-format",
        choices=["csv", "json"],
        default="csv",
        help="Output file format",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Override optimal threshold from metadata",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        help="Limit number of records (debug/smoke)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(list(argv) if argv is not None else None)


def _configure_logging(verbose: bool) -> None:
    import logging

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


def main(argv: Optional[Iterable[str]] = None) -> int:
    """CLI entrypoint.

    Parses arguments, configures logging, runs inference workflow.

    Args:
        argv: Optional iterable of CLI argument strings (used for testing).

    Returns:
        Process exit code (0 success, >0 failure).
    """
    args = _parse_args(argv)
    _configure_logging(args.verbose)
    try:
        run_inference(
            model_path=args.model_path,
            input_path=args.input_file,
            output_path=args.output_file,
            threshold_override=args.threshold,
            output_format=args.output_format,
            max_records=args.max_records,
        )
        return 0
    except Exception as e:
        logger.error("Inference failed", error=str(e))
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
