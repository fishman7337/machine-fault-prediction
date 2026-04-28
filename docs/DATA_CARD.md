# Data Card

## Dataset

`data/raw/factory_data.csv`

The file contains 20,000 factory machine observations used to predict whether a machine is normal or abnormal.

## Columns

| Column | Type | Description |
| --- | --- | --- |
| `Unique ID` | integer | Row-level machine identifier. Dropped before modeling. |
| `Product ID` | string | Product identifier. Dropped before modeling. |
| `Quality` | categorical | Product quality class: `L`, `M`, or `H`. |
| `Ambient T (C)` | numeric | Ambient temperature in Celsius. |
| `Process T (C)` | numeric | Process temperature in Celsius. |
| `Rotation Speed (rpm)` | numeric | Machine rotation speed. |
| `Torque (Nm)` | numeric | Torque reading. |
| `Tool Wear (min)` | numeric | Tool wear duration. |
| `Machine Status` | binary | Target: `0` normal, `1` abnormal/fault. |

## Missing Values

Known missing values are present in:

- `Quality`
- `Process T (C)`
- `Rotation Speed (rpm)`

The production pipeline imputes missing values inside the scikit-learn pipeline after the train/test split, which prevents leakage from validation data into training preprocessing.

## Class Balance

The target is highly imbalanced:

- Normal: 19,322 rows
- Fault/abnormal: 678 rows

Balanced accuracy, recall, F1, ROC AUC, and average precision are reported because plain accuracy can be misleading for this dataset.

## Intended Use

This dataset is appropriate for supervised classification practice and predictive maintenance workflow development. It should not be treated as a fully validated production monitoring dataset without further domain review.
