# Predictive Maintenance Fault Classifier

A reproducible machine learning project for predicting factory machine fault status from sensor readings. The original notebook analysis has been preserved, and the production workflow has been extracted into a tested Python package with CLI commands, CI, statistical tests, and MLOps documentation.

## Academic Context

This project was completed for Singapore Polytechnic, School of Computing, Diploma in Applied AI & Analytics, under the AI & Machine Learning module (`ST1511`), CA1 Part A.

- Student: Goh Kun Ming, DAAA student
- Academic year: AY24/25, Year 1 Semester 2
- Lecturer: Adjunct Lecturer Tai Hock Lin (Andy)

## What changed

- Corrected the raw dataset extension from `factory_data.xls` to `data/raw/factory_data.csv` because the file is CSV content.
- Moved the notebook and slides into proper `notebooks/` and `reports/` folders.
- Added a reusable `fault_prediction` package under `src/`.
- Fixed the feature engineering issue: after creating `Temperature Gradient` and `Power Indicator`, the source columns used only to create those engineered features are removed from the model input.
- Added p-value based statistical testing for numeric and categorical feature association with machine status.
- Added pytest coverage, Ruff linting, GitHub Actions CI, project metadata, and MLOps documentation.

## Project Structure

```text
.
|-- data/
|   |-- raw/factory_data.csv
|   `-- processed/
|-- docs/
|   |-- ACADEMIC_CONTEXT.md
|   |-- DATA_CARD.md
|   |-- FEATURE_ENGINEERING.md
|   |-- MLOPS.md
|   |-- MODEL_CARD.md
|   |-- PROJECT_STRUCTURE.md
|   `-- STATISTICAL_TESTS.md
|-- models/
|-- notebooks/
|   `-- factory_machine_status_classification.ipynb
|-- reports/
|   |-- figures/
|   |-- metrics/
|   `-- presentations/
|-- src/fault_prediction/
|-- tests/
|-- .github/workflows/ci.yml
`-- pyproject.toml
```

## Quickstart

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebook]"
pytest
```

Run the statistical p-value tests:

```bash
fault-predict stats --data data/raw/factory_data.csv
```

Train a fast smoke-test model:

```bash
fault-predict train --profile fast --sample-size 1000
```

Train the standard model:

```bash
fault-predict train --profile standard
```

Score data with a saved model:

```bash
fault-predict predict --model models/fault_voting_classifier.joblib --input data/raw/factory_data.csv
```

## Dataset

The raw data contains 20,000 rows and these fields:

- `Unique ID`
- `Product ID`
- `Quality`
- `Ambient T (C)`
- `Process T (C)`
- `Rotation Speed (rpm)`
- `Torque (Nm)`
- `Tool Wear (min)`
- `Machine Status`

`Machine Status` is the binary target: `0` is normal and `1` is fault/abnormal. The target is imbalanced, so the training workflow reports balanced accuracy, precision, recall, F1, ROC AUC, and average precision.

## Modeling Approach

The production model is an end-to-end scikit-learn pipeline:

1. Validate the input schema.
2. Engineer `Temperature Gradient = Process T (C) - Ambient T (C)`.
3. Engineer `Power Indicator = Torque (Nm) / Rotation Speed (rpm)`.
4. Drop identifiers and redundant source features after engineering.
5. Impute missing values using training-split statistics only.
6. Scale numeric features and one-hot encode `Quality`.
7. Train a soft-voting classifier using decision tree, random forest, gradient boosting, AdaBoost, k-nearest neighbors, and logistic regression estimators.
8. Search a decision threshold on the validation split for the chosen metric.

## Quality Gates

Local checks:

```bash
ruff check .
pytest
```

CI runs the same checks on Python 3.10, 3.11, and 3.12.

## Documentation

- [Academic context](docs/ACADEMIC_CONTEXT.md)
- [Project structure](docs/PROJECT_STRUCTURE.md)
- [Data card](docs/DATA_CARD.md)
- [Feature engineering](docs/FEATURE_ENGINEERING.md)
- [Statistical tests](docs/STATISTICAL_TESTS.md)
- [Model card](docs/MODEL_CARD.md)
- [MLOps workflow](docs/MLOPS.md)

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
