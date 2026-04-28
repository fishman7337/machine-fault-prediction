from fault_prediction.config import REQUIRED_TRAINING_COLUMNS, TARGET_COLUMN
from fault_prediction.data import target_distribution, validate_schema


def test_factory_data_schema(factory_data):
    validate_schema(factory_data)
    assert list(REQUIRED_TRAINING_COLUMNS) == list(factory_data.columns)
    assert set(factory_data[TARGET_COLUMN].unique()) == {0, 1}


def test_target_distribution_sums_to_one(factory_data):
    distribution = target_distribution(factory_data)
    assert distribution["count"].sum() == len(factory_data)
    assert distribution["percentage"].sum() == 1
