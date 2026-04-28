"""Command line interface for training, statistics, and prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fault_prediction.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_STATS_PATH,
    DEFAULT_THRESHOLD_PATH,
    IDENTIFIER_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN,
)
from fault_prediction.data import load_factory_data, validate_schema
from fault_prediction.models import load_artifact, predict_scores, save_artifact, train_model
from fault_prediction.statistics import run_statistical_tests


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer | np.floating):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(data: dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")
    return output_path


def _maybe_sample(df: pd.DataFrame, sample_size: int | None, random_state: int) -> pd.DataFrame:
    if sample_size is None or sample_size >= len(df):
        return df
    validate_schema(df, require_target=True)
    sampled, _ = train_test_split(
        df,
        train_size=sample_size,
        random_state=random_state,
        stratify=df[TARGET_COLUMN],
    )
    return sampled.reset_index(drop=True)


def train_command(args: argparse.Namespace) -> int:
    df = load_factory_data(args.data)
    df = _maybe_sample(df, args.sample_size, args.random_state)
    result = train_model(
        df,
        profile=args.profile,
        test_size=args.test_size,
        random_state=args.random_state,
        threshold_metric=args.threshold_metric,
        n_jobs=args.n_jobs,
    )

    model_path = save_artifact(result.artifact, args.model_out)

    metrics_payload = {
        "model_path": str(model_path),
        "rows_used": int(len(df)),
        "metrics": result.metrics,
        "metadata": result.artifact["metadata"],
    }
    metrics_path = _write_json(metrics_payload, args.metrics_out)

    threshold_path = Path(args.threshold_out)
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    result.threshold_curve.to_csv(threshold_path, index=False)

    print(f"Saved model artifact: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved threshold search: {threshold_path}")
    print(json.dumps(result.metrics, indent=2, default=_json_default))
    return 0


def stats_command(args: argparse.Namespace) -> int:
    df = load_factory_data(args.data)
    results = run_statistical_tests(df, alpha=args.alpha)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"Saved statistical test results: {output_path}")
    print(results.head(args.preview_rows).to_string(index=False))
    return 0


def predict_command(args: argparse.Namespace) -> int:
    artifact = load_artifact(args.model)
    model = artifact["model"]
    threshold = float(artifact.get("threshold", 0.5))
    df = load_factory_data(args.input)
    validate_schema(df, require_target=False)

    scores = predict_scores(model, df)
    predictions = (scores >= threshold).astype(int)

    output = pd.DataFrame(
        {
            "predicted_probability_fault": scores,
            "predicted_machine_status": predictions,
        }
    )
    for column in reversed(IDENTIFIER_COLUMNS):
        if column in df.columns:
            output.insert(0, column, df[column])
    if TARGET_COLUMN in df.columns:
        output[TARGET_COLUMN] = df[TARGET_COLUMN]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Saved predictions: {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fault-predict",
        description="Train and operate the machine fault prediction workflow.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the fault classifier")
    train_parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    train_parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH)
    train_parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS_PATH)
    train_parser.add_argument("--threshold-out", type=Path, default=DEFAULT_THRESHOLD_PATH)
    train_parser.add_argument("--profile", choices=["fast", "standard"], default="standard")
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--threshold-metric", choices=["f1", "recall"], default="f1")
    train_parser.add_argument("--sample-size", type=int, default=None)
    train_parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    train_parser.add_argument("--n-jobs", type=int, default=-1)
    train_parser.set_defaults(func=train_command)

    stats_parser = subparsers.add_parser("stats", help="Run feature p-value tests")
    stats_parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    stats_parser.add_argument("--output", type=Path, default=DEFAULT_STATS_PATH)
    stats_parser.add_argument("--alpha", type=float, default=0.05)
    stats_parser.add_argument("--preview-rows", type=int, default=10)
    stats_parser.set_defaults(func=stats_command)

    predict_parser = subparsers.add_parser("predict", help="Score rows with a trained model")
    predict_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    predict_parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH)
    predict_parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/metrics/predictions.csv"),
    )
    predict_parser.set_defaults(func=predict_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
