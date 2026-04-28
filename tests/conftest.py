from __future__ import annotations

import pandas as pd
import pytest

from fault_prediction.config import DEFAULT_DATA_PATH, TARGET_COLUMN
from fault_prediction.data import load_factory_data


@pytest.fixture(scope="session")
def factory_data() -> pd.DataFrame:
    return load_factory_data(DEFAULT_DATA_PATH)


@pytest.fixture(scope="session")
def balanced_sample(factory_data: pd.DataFrame) -> pd.DataFrame:
    normal = factory_data[factory_data[TARGET_COLUMN] == 0].sample(120, random_state=1)
    fault = factory_data[factory_data[TARGET_COLUMN] == 1].sample(80, random_state=1)
    return pd.concat([normal, fault], ignore_index=True).sample(frac=1, random_state=1)
