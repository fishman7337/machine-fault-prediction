# Feature Engineering

## Engineered Features

The production pipeline creates two derived features:

| Feature | Formula | Purpose |
| --- | --- | --- |
| `Temperature Gradient` | `Process T (C) - Ambient T (C)` | Captures thermal load above ambient conditions. |
| `Power Indicator` | `Torque (Nm) / Rotation Speed (rpm)` | Captures load relative to rotation speed. |

## Corrected Feature Policy

The original notebook created engineered features but kept all source columns in the model input. The package now removes redundant source columns after feature engineering:

- `Ambient T (C)`
- `Process T (C)`
- `Rotation Speed (rpm)`
- `Torque (Nm)`

The model receives:

- `Tool Wear (min)`
- `Temperature Gradient`
- `Power Indicator`
- encoded `Quality`

This reduces duplicate representations and makes the feature contract easier to audit.

## Columns Always Removed

Identifier columns are removed before modeling:

- `Unique ID`
- `Product ID`

The target column is removed from the feature matrix:

- `Machine Status`

## Leakage Control

Feature creation uses only row-level sensor values. Imputation, scaling, and encoding are fit inside the scikit-learn pipeline on the training split only.
