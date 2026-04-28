# Notebooks

The notebook workflow is split into small, output-free notebooks:

- `01_data_understanding.ipynb`
- `02_feature_engineering_and_statistics.ipynb`
- `03_training_and_thresholding.ipynb`
- `04_evaluation_and_prediction.ipynb`

The original CA1 notebook is preserved in `archive/original_factory_machine_status_classification.ipynb` for assessment traceability.

The production workflow lives in `src/fault_prediction/` and should be used for repeatable training, testing, and prediction. New reusable logic should be added to the package first, then imported into notebooks when exploration needs it.
