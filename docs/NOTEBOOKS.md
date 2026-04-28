# Notebook Guide

The original project notebook was large and difficult to review as a single artifact. The notebook workflow is now split into smaller notebooks that each have one purpose and import the tested production package.

## Notebook Order

| Notebook | Purpose |
| --- | --- |
| `notebooks/01_data_understanding.ipynb` | Load the raw dataset, validate schema, inspect missing values, and review target balance. |
| `notebooks/02_feature_engineering_and_statistics.ipynb` | Demonstrate engineered features, confirm source-feature removal, and run p-value tests. |
| `notebooks/03_training_and_thresholding.ipynb` | Train a fast model sample and inspect validation metrics plus threshold selection. |
| `notebooks/04_evaluation_and_prediction.ipynb` | Score holdout rows and preview the model output contract. |

## Archive

The original assessment notebook is retained at:

```text
notebooks/archive/original_factory_machine_status_classification.ipynb
```

It is preserved for academic traceability. New development should happen in the split notebooks or, preferably, in `src/fault_prediction/` with notebook cells importing reusable package functions.

## Quality Rules

- Keep split notebooks output-free before committing.
- Keep long-running model training in the CLI unless a notebook is explicitly for experimentation.
- Keep generated model artifacts out of Git.
- Add reusable logic to `src/fault_prediction/`, not directly inside notebooks.
- Use the notebooks as readable workflow demonstrations, not as the source of production behavior.

## Recommended Flow

```bash
python -m pip install -e ".[dev,notebook]"
pytest
fault-predict stats --data data/raw/factory_data.csv
fault-predict train --profile fast --sample-size 1000
```
