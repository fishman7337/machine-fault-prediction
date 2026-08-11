"""Feature engineering for machine fault prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from fault_prediction.config import (
    ENGINEERED_COLUMNS,
    IDENTIFIER_COLUMNS,
    REQUIRED_INPUT_COLUMNS,
    SOURCE_COLUMNS_DROPPED_AFTER_ENGINEERING,
    TARGET_COLUMN,
)
from fault_prediction.data import validate_schema


def engineer_features(
    data: pd.DataFrame,
    *,
    drop_source_features: bool = True,
    drop_identifier_columns: bool = True,
    drop_target: bool = True,
) -> pd.DataFrame:
    """Create engineered features and remove columns that should not enter the model.

    ``Temperature Gradient`` and ``Power Indicator`` are derived from source sensor
    columns. By default, the source columns used only to make those engineered
    features are removed afterwards to avoid feeding redundant representations into
    the estimator.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Feature engineering expects a pandas DataFrame")

    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    df = data.copy()
    ambient = pd.to_numeric(df["Ambient T (C)"], errors="coerce")
    process = pd.to_numeric(df["Process T (C)"], errors="coerce")
    rotation_speed = pd.to_numeric(df["Rotation Speed (rpm)"], errors="coerce")
    torque = pd.to_numeric(df["Torque (Nm)"], errors="coerce")

    df[ENGINEERED_COLUMNS[0]] = process - ambient
    df[ENGINEERED_COLUMNS[1]] = torque / rotation_speed.replace(0, np.nan)

    columns_to_drop: list[str] = []
    if drop_identifier_columns:
        columns_to_drop.extend(IDENTIFIER_COLUMNS)
    if drop_source_features:
        columns_to_drop.extend(SOURCE_COLUMNS_DROPPED_AFTER_ENGINEERING)
    if drop_target and TARGET_COLUMN in df.columns:
        columns_to_drop.append(TARGET_COLUMN)

    return df.drop(columns=[column for column in columns_to_drop if column in df.columns])


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible feature engineering transformer."""

    def __init__(
        self,
        *,
        drop_source_features: bool = True,
        drop_identifier_columns: bool = True,
    ) -> None:
        """Configure source and identifier column removal.

        Args:
            drop_source_features: Remove sensor columns replaced by engineered features.
            drop_identifier_columns: Remove non-predictive product identifiers.
        """
        self.drop_source_features = drop_source_features
        self.drop_identifier_columns = drop_identifier_columns

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> FeatureEngineer:
        """Validate the feature schema without learning state.

        Args:
            X: Machine sensor features.
            y: Optional targets accepted for scikit-learn compatibility.

        Returns:
            This fitted transformer instance.
        """
        validate_schema(X, require_target=False)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply deterministic machine-sensor feature engineering.

        Args:
            X: Raw machine sensor features.

        Returns:
            Engineered predictors with configured columns removed.
        """
        return engineer_features(
            X,
            drop_source_features=self.drop_source_features,
            drop_identifier_columns=self.drop_identifier_columns,
            drop_target=True,
        )
