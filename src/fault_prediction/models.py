"""Model construction, training, evaluation, and artifact persistence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from fault_prediction import __version__
from fault_prediction.config import (
    CATEGORICAL_MODEL_COLUMNS,
    NUMERIC_MODEL_COLUMNS,
    RANDOM_STATE,
    SOURCE_COLUMNS_DROPPED_AFTER_ENGINEERING,
)
from fault_prediction.data import split_features_target, validate_schema
from fault_prediction.features import FeatureEngineer


@dataclass(frozen=True)
class TrainResult:
    """Container for a trained model artifact and validation outputs."""

    artifact: dict[str, Any]
    metrics: dict[str, float]
    threshold_curve: pd.DataFrame


def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing that is fit only on the training split."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(NUMERIC_MODEL_COLUMNS)),
            ("categorical", categorical_pipeline, list(CATEGORICAL_MODEL_COLUMNS)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _estimators(profile: str, random_state: int, n_jobs: int) -> list[tuple[str, Any]]:
    """Return the soft-voting base estimators for the selected runtime profile."""

    if profile not in {"fast", "standard"}:
        raise ValueError("profile must be 'fast' or 'standard'")

    if profile == "fast":
        forest_estimators = 40
        boosting_estimators = 40
        max_depth = 5
        neighbors = 21
    else:
        forest_estimators = 160
        boosting_estimators = 120
        max_depth = 7
        neighbors = 51

    return [
        (
            "decision_tree",
            DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=random_state,
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=forest_estimators,
                max_depth=max_depth,
                min_samples_leaf=20,
                max_features="log2",
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=n_jobs,
            ),
        ),
        (
            "gradient_boosting",
            GradientBoostingClassifier(
                n_estimators=boosting_estimators,
                learning_rate=0.05,
                max_depth=3,
                min_samples_leaf=20,
                random_state=random_state,
            ),
        ),
        (
            "adaboost",
            AdaBoostClassifier(
                n_estimators=boosting_estimators,
                learning_rate=0.05,
                random_state=random_state,
            ),
        ),
        (
            "knn",
            KNeighborsClassifier(n_neighbors=neighbors, weights="distance"),
        ),
        (
            "logistic_regression",
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=random_state,
            ),
        ),
    ]


def build_model(
    *,
    profile: str = "standard",
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
) -> Pipeline:
    """Build the end-to-end machine fault classifier."""

    classifier = VotingClassifier(
        estimators=_estimators(profile=profile, random_state=random_state, n_jobs=n_jobs),
        voting="soft",
        n_jobs=n_jobs,
    )

    return Pipeline(
        steps=[
            ("features", FeatureEngineer(drop_source_features=True)),
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def predict_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities from a trained classifier."""

    probabilities = model.predict_proba(X)
    return probabilities[:, 1]


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute core binary-classification metrics."""

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_score is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["average_precision"] = float(average_precision_score(y_true, y_score))

    return metrics


def find_best_threshold(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    *,
    metric: str = "f1",
    thresholds: np.ndarray | None = None,
) -> tuple[float, pd.DataFrame]:
    """Search probability thresholds and return the best threshold plus diagnostics."""

    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 181)

    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        row = evaluate_predictions(y_true, y_pred, y_score)
        row["threshold"] = float(threshold)
        rows.append(row)

    threshold_curve = pd.DataFrame(rows)
    if metric not in threshold_curve.columns:
        raise ValueError(f"Unsupported threshold metric: {metric}")

    best_row = threshold_curve.sort_values(
        by=[metric, "recall", "balanced_accuracy"],
        ascending=[False, False, False],
    ).iloc[0]
    return float(best_row["threshold"]), threshold_curve


def get_feature_names(model: Pipeline) -> list[str]:
    """Return feature names after preprocessing for a fitted model."""

    preprocessor = model.named_steps["preprocessor"]
    return [str(name) for name in preprocessor.get_feature_names_out()]


def train_model(
    df: pd.DataFrame,
    *,
    profile: str = "standard",
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    threshold_metric: str = "f1",
    n_jobs: int = -1,
) -> TrainResult:
    """Train and evaluate the fault classifier on a stratified holdout split."""

    validate_schema(df, require_target=True)
    X, y = split_features_target(df)
    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    model = build_model(profile=profile, random_state=random_state, n_jobs=n_jobs)
    model.fit(X_train, y_train)

    y_score = predict_scores(model, X_test)
    threshold, threshold_curve = find_best_threshold(
        y_test,
        y_score,
        metric=threshold_metric,
    )
    y_pred = (y_score >= threshold).astype(int)
    metrics = evaluate_predictions(y_test, y_pred, y_score)
    metrics["threshold"] = threshold
    metrics["test_size"] = float(test_size)
    metrics["validation_rows"] = float(len(y_test))

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "profile": profile,
        "random_state": random_state,
        "threshold_metric": threshold_metric,
        "dropped_after_feature_engineering": list(SOURCE_COLUMNS_DROPPED_AFTER_ENGINEERING),
        "feature_names": get_feature_names(model),
        "metrics": metrics,
    }
    artifact = {
        "model": model,
        "threshold": threshold,
        "metadata": metadata,
    }

    return TrainResult(artifact=artifact, metrics=metrics, threshold_curve=threshold_curve)


def save_artifact(artifact: dict[str, Any], path: str | Path) -> Path:
    """Persist a trained model artifact with metadata."""

    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)
    return artifact_path


def load_artifact(path: str | Path) -> dict[str, Any]:
    """Load a trained model artifact, including backward compatibility for raw models."""

    artifact = joblib.load(path)
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact

    return {
        "model": artifact,
        "threshold": 0.5,
        "metadata": {
            "loaded_as_legacy_model": True,
        },
    }


def serializable_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Normalize metric values for JSON output."""

    return {key: float(value) for key, value in metrics.items()}
