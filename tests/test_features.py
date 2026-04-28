import numpy as np

from fault_prediction.config import (
    ENGINEERED_COLUMNS,
    IDENTIFIER_COLUMNS,
    SOURCE_COLUMNS_DROPPED_AFTER_ENGINEERING,
    TARGET_COLUMN,
)
from fault_prediction.features import engineer_features


def test_engineering_drops_redundant_source_features(factory_data):
    engineered = engineer_features(factory_data)

    for column in ENGINEERED_COLUMNS:
        assert column in engineered.columns

    for column in SOURCE_COLUMNS_DROPPED_AFTER_ENGINEERING:
        assert column not in engineered.columns

    for column in IDENTIFIER_COLUMNS:
        assert column not in engineered.columns

    assert TARGET_COLUMN not in engineered.columns


def test_engineered_feature_values(factory_data):
    row = factory_data.iloc[[0]]
    engineered = engineer_features(row)

    expected_gradient = row["Process T (C)"].iloc[0] - row["Ambient T (C)"].iloc[0]
    expected_power = row["Torque (Nm)"].iloc[0] / row["Rotation Speed (rpm)"].iloc[0]

    assert np.isclose(engineered["Temperature Gradient"].iloc[0], expected_gradient)
    assert np.isclose(engineered["Power Indicator"].iloc[0], expected_power)
