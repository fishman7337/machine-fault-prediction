# Notebooks

## Maintained Original

`00_original_ca1_submission.ipynb` is the maintained original CA1 notebook. It is kept as the academic assessment artifact and should remain close to the submitted notebook style and narrative.

## Modular Workflow

The production-facing notebook workflow is split into small, output-free notebooks:

- `01_data_understanding.ipynb`
- `02_feature_engineering_and_statistics.ipynb`
- `03_training_and_thresholding.ipynb`
- `04_evaluation_and_prediction.ipynb`

The production workflow lives in `src/fault_prediction/` and should be used for repeatable training, testing, and prediction. New reusable logic should be added to the package first, then imported into notebooks when exploration needs it.

The split notebooks intentionally keep the original report-style markdown format: academic context, numbered headings, explanatory paragraphs, and interpretation notes after code cells.
