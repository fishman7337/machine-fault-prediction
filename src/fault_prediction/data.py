"""Data loading and validation helpers."""

from pathlib import Path

import pandas as pd

from fault_prediction.config import (
    DEFAULT_DATA_PATH,
    REQUIRED_INPUT_COLUMNS,
    REQUIRED_TRAINING_COLUMNS,
    TARGET_COLUMN,
)


def load_factory_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the factory machine dataset.

    The original project stored CSV content with an ``.xls`` extension. This loader
    accepts either the corrected CSV path or an older spreadsheet-looking path.
    """

    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    if data_path.suffix.lower() in {".xls", ".xlsx"}:
        try:
            return pd.read_excel(data_path)
        except ValueError:
            return pd.read_csv(data_path)

    return pd.read_csv(data_path)


def validate_schema(df: pd.DataFrame, *, require_target: bool = True) -> pd.DataFrame:
    """Validate that a dataframe contains the columns required by the pipeline."""

    required = REQUIRED_TRAINING_COLUMNS if require_target else REQUIRED_INPUT_COLUMNS
    missing = [column for column in required if column not in df.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing required columns: {missing_text}")

    if require_target:
        target_values = set(df[TARGET_COLUMN].dropna().unique())
        if not target_values.issubset({0, 1}):
            raise ValueError(f"{TARGET_COLUMN} must contain only 0 and 1 values")

    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a validated dataframe into raw features and target."""

    validate_schema(df, require_target=True)
    return df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN].astype(int)


def target_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return target counts and percentages for monitoring class imbalance."""

    validate_schema(df, require_target=True)
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    distribution = counts.rename("count").to_frame()
    distribution["percentage"] = distribution["count"] / len(df)
    return distribution
