# Project Structure

This repository separates exploratory assets from the production workflow.

```text
data/raw/
```

Versioned source data. The canonical dataset is `factory_data.csv`.

```text
data/processed/
```

Generated datasets. This folder is intentionally ignored except for `.gitkeep`.

```text
notebooks/
```

Exploratory notebooks. The notebook is preserved for auditability, but reusable logic should live in `src/fault_prediction/`.

```text
src/fault_prediction/
```

Production package for loading data, validating schema, engineering features, training models, running statistical tests, and serving predictions.

```text
models/
```

Generated model artifacts. Files are ignored so large or stale models do not get committed accidentally.

```text
reports/
```

Generated metrics, figures, and presentation materials. The original slide deck is preserved under `reports/presentations/`.

```text
tests/
```

Pytest coverage for schema validation, feature engineering, statistics, and training smoke checks.
