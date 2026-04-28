"""Project-level constants for the fault prediction workflow."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RANDOM_STATE = 28
TARGET_COLUMN = "Machine Status"
QUALITY_COLUMN = "Quality"

IDENTIFIER_COLUMNS = ("Unique ID", "Product ID")
RAW_SENSOR_COLUMNS = (
    "Ambient T (C)",
    "Process T (C)",
    "Rotation Speed (rpm)",
    "Torque (Nm)",
    "Tool Wear (min)",
)
ENGINEERED_COLUMNS = ("Temperature Gradient", "Power Indicator")

SOURCE_COLUMNS_DROPPED_AFTER_ENGINEERING = (
    "Ambient T (C)",
    "Process T (C)",
    "Rotation Speed (rpm)",
    "Torque (Nm)",
)
NUMERIC_MODEL_COLUMNS = ("Tool Wear (min)", "Temperature Gradient", "Power Indicator")
CATEGORICAL_MODEL_COLUMNS = (QUALITY_COLUMN,)

REQUIRED_INPUT_COLUMNS = (
    *IDENTIFIER_COLUMNS,
    QUALITY_COLUMN,
    *RAW_SENSOR_COLUMNS,
)
REQUIRED_TRAINING_COLUMNS = (*REQUIRED_INPUT_COLUMNS, TARGET_COLUMN)

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "factory_data.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "fault_voting_classifier.joblib"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "metrics.json"
DEFAULT_THRESHOLD_PATH = PROJECT_ROOT / "reports" / "metrics" / "threshold_search.csv"
DEFAULT_STATS_PATH = PROJECT_ROOT / "reports" / "metrics" / "statistical_tests.csv"
