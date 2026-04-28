# Statistical Tests

The project includes p-value tests to support feature review before modeling.

Run:

```bash
fault-predict stats --data data/raw/factory_data.csv
```

Default output:

```text
reports/metrics/statistical_tests.csv
```

## Numeric Features

For numeric features, the workflow compares normal machines (`Machine Status = 0`) against fault machines (`Machine Status = 1`) using:

- Welch t-test: compares group means without assuming equal variance.
- Mann-Whitney U test: compares distributions without assuming normality.

Numeric features tested:

- `Ambient T (C)`
- `Process T (C)`
- `Rotation Speed (rpm)`
- `Torque (Nm)`
- `Tool Wear (min)`
- `Temperature Gradient`
- `Power Indicator`

## Categorical Features

For `Quality`, the workflow uses a chi-square test of independence against `Machine Status`.

## Interpretation

The default alpha is `0.05`.

- `p_value < 0.05`: statistically significant association.
- `p_value >= 0.05`: no statistically significant association at the selected alpha.

Statistical significance does not prove business usefulness or causal impact. It is an evidence layer to combine with model performance, drift monitoring, and domain review.
