# Notebook Guide

The project now keeps both the maintained original CA1 notebook and a split production-facing notebook workflow. This protects the academic submission narrative while making the improved project easier to review, run, and maintain.

## Maintained Original

| Notebook | Purpose |
| --- | --- |
| `notebooks/00_original_ca1_submission.ipynb` | Original CA1 assessment notebook maintained for academic traceability and narrative continuity. |

The original notebook is intentionally not forced to be output-free because it represents the assessment artifact. Avoid rewriting its narrative unless the goal is to correct a clear factual or technical issue.

## Modular Notebook Order

| Notebook | Purpose |
| --- | --- |
| `notebooks/01_data_understanding.ipynb` | Load the raw dataset, validate schema, inspect missing values, and review target balance. |
| `notebooks/02_feature_engineering_and_statistics.ipynb` | Demonstrate engineered features, confirm source-feature removal, and run p-value tests. |
| `notebooks/03_training_and_thresholding.ipynb` | Train a fast model sample and inspect validation metrics plus threshold selection. |
| `notebooks/04_evaluation_and_prediction.ipynb` | Score holdout rows and preview the model output contract. |

## Quality Rules

- Keep split notebooks output-free before committing.
- Keep long-running model training in the CLI unless a notebook is explicitly for experimentation.
- Keep generated model artifacts out of Git.
- Add reusable logic to `src/fault_prediction/`, not directly inside notebooks.
- Use the notebooks as readable workflow demonstrations, not as the source of production behavior.
- Keep the original notebook available as `00_original_ca1_submission.ipynb`.

## Recommended Flow

```bash
python -m pip install -e ".[dev,notebook]"
pytest
fault-predict stats --data data/raw/factory_data.csv
fault-predict train --profile fast --sample-size 1000
```
