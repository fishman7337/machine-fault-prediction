# Assessment Report

## Project Title

Predictive Maintenance Fault Classifier.

## Academic Context

This project was completed for Singapore Polytechnic, School of Computing, Diploma in Applied AI & Analytics, under the AI & Machine Learning module (`ST1511`), CA1 Part A.

- Student: Goh Kun Ming, DAAA student
- Academic year: AY24/25, Year 1 Semester 2
- Lecturer: Adjunct Lecturer Tai Hock Lin (Andy)

## Problem Statement

Factory machine failures can interrupt production, increase maintenance cost, and reduce operational reliability. The project uses machine sensor and product quality data to classify whether a machine is operating normally or is likely to be in an abnormal fault state.

## Objective

The objective is to build a supervised binary classification workflow that:

- Cleans and validates the provided factory dataset.
- Engineers meaningful features from raw sensor readings.
- Evaluates feature association using statistical tests.
- Trains and evaluates machine learning classifiers.
- Produces a repeatable predictive maintenance workflow that can be tested, documented, and improved.

## Dataset Summary

The canonical dataset is `data/raw/factory_data.csv`. It contains 20,000 rows and these fields:

- `Unique ID`
- `Product ID`
- `Quality`
- `Ambient T (C)`
- `Process T (C)`
- `Rotation Speed (rpm)`
- `Torque (Nm)`
- `Tool Wear (min)`
- `Machine Status`

`Machine Status` is the target. `0` represents normal operation and `1` represents an abnormal or fault state.

## Data Quality Review

The dataset contains missing values in:

- `Quality`
- `Process T (C)`
- `Rotation Speed (rpm)`

The production workflow imputes missing values inside the scikit-learn pipeline after the train/test split. This prevents validation information from leaking into training preprocessing.

The target is imbalanced, with far more normal records than fault records. This means accuracy alone is not enough to judge performance.

## Feature Engineering

The workflow creates:

- `Temperature Gradient = Process T (C) - Ambient T (C)`
- `Power Indicator = Torque (Nm) / Rotation Speed (rpm)`

The improved project explicitly removes source columns that are only kept to create engineered features:

- `Ambient T (C)`
- `Process T (C)`
- `Rotation Speed (rpm)`
- `Torque (Nm)`

Identifier columns are also removed:

- `Unique ID`
- `Product ID`

This makes the model input easier to audit and avoids keeping redundant representations of the same information.

## Statistical Testing

The project includes p-value based feature review:

- Welch t-test for numeric feature mean differences.
- Mann-Whitney U test for numeric distribution differences.
- Chi-square test for categorical association between `Quality` and `Machine Status`.

These tests support feature understanding, but they do not replace model validation or domain judgment.

## Modeling Approach

The production model is an end-to-end scikit-learn pipeline. It performs:

1. Schema validation.
2. Feature engineering.
3. Missing-value imputation.
4. Numeric scaling.
5. Categorical encoding.
6. Soft-voting classification.
7. Decision-threshold search.

The voting classifier combines:

- Decision tree
- Random forest
- Gradient boosting
- AdaBoost
- K-nearest neighbors
- Logistic regression

## Evaluation Approach

The workflow reports:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1 score
- ROC AUC
- Average precision
- Selected probability threshold

Balanced accuracy, recall, F1, and average precision are important because the target class is imbalanced.

## Notebook Structure

The original CA1 notebook is maintained as:

```text
notebooks/00_original_ca1_submission.ipynb
```

The improved workflow is split into:

- `notebooks/01_data_understanding.ipynb`
- `notebooks/02_feature_engineering_and_statistics.ipynb`
- `notebooks/03_training_and_thresholding.ipynb`
- `notebooks/04_evaluation_and_prediction.ipynb`

The original notebook preserves the academic report style. The split notebooks provide a cleaner workflow for review, maintenance, and future improvement.

## MLOps Improvements

The repository now includes:

- Python package code under `src/fault_prediction/`.
- CLI commands for training, statistical testing, and prediction.
- Pytest coverage for data, feature engineering, statistics, training, notebooks, and documentation.
- Ruff linting.
- GitHub Actions CI.
- Data, model, feature, notebook, and MLOps documentation.
- Git ignore rules for generated model and metric artifacts.

## Limitations

- The dataset is imbalanced.
- The model predicts fault status but does not diagnose root cause.
- The data is tabular and does not include temporal sequence behavior.
- The selected threshold should be reviewed against real maintenance costs before operational use.
- Statistical significance does not prove causality.

## Business Implications

A reliable fault prediction model can help maintenance teams prioritize inspections and reduce unplanned downtime. In a real deployment, the model should be used as decision support rather than as a fully automated safety-critical control system.

## Reproducibility

Recommended local checks:

```bash
python -m pip install -e ".[dev,notebook]"
ruff check .
pytest
fault-predict stats --data data/raw/factory_data.csv
fault-predict train --profile fast --sample-size 1000
```

Full training can be run with:

```bash
fault-predict train --profile standard
```
