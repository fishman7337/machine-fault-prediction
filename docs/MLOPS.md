# MLOps Workflow

## Reproducibility

- Package code lives in `src/fault_prediction/`.
- The raw dataset lives in `data/raw/factory_data.csv`.
- Randomness is controlled by `RANDOM_STATE = 28`.
- Training creates a model artifact plus JSON/CSV reports.

## Local Quality Gates

```bash
python -m pip install -r requirements-dev.txt
ruff check .
pytest
```

The test suite also checks that the split notebooks are valid, ordered, and output-free. The maintained original CA1 notebook is kept for traceability and is not held to the same output-free rule.

Security checks can be run locally with:

```bash
python -m pip install -r requirements-security.txt
bandit -c pyproject.toml -r src
pip-audit --skip-editable
```

## CI

GitHub Actions runs on pushes and pull requests to `main`.

The CI matrix tests:

- Python 3.10
- Python 3.11
- Python 3.12

The CI workflow includes:

- A test matrix that installs development dependencies, runs Ruff, and runs pytest.
- A security job that runs Bandit static analysis and `pip-audit` dependency vulnerability auditing.

## Training

Fast smoke training:

```bash
fault-predict train --profile fast --sample-size 1000
```

Standard training:

```bash
fault-predict train --profile standard
```

Generated outputs:

- `models/fault_voting_classifier.joblib`
- `reports/metrics/metrics.json`
- `reports/metrics/threshold_search.csv`

Generated models and metric outputs are ignored by Git. Promote model artifacts through a model registry or release asset rather than committing them.

## Statistical Review

Run p-value tests before major modeling changes:

```bash
fault-predict stats --data data/raw/factory_data.csv
```

Use the results as a feature-review signal, not as a replacement for validation metrics.

## Deployment Contract

Prediction inputs must include all required raw columns, even though some are dropped after feature engineering. The pipeline owns feature creation, imputation, scaling, encoding, and prediction.

## Monitoring

Track these after deployment:

- Input column presence and data types.
- Missing value rates per feature.
- Prediction probability distribution.
- Predicted fault rate.
- Label-based precision, recall, and false-negative rate when ground truth arrives.
- Distribution drift for `Tool Wear (min)`, `Temperature Gradient`, `Power Indicator`, and `Quality`.

## Retraining Trigger Examples

- Fault rate changes materially.
- Sensor calibration or upstream data collection changes.
- Missing values increase beyond expected levels.
- Recall drops below the agreed operating target.
- A new machine class or product quality category appears.
