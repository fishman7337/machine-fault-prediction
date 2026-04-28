# Project Structure

This repository separates exploratory assets from the production workflow.

```text
docs/ACADEMIC_CONTEXT.md
```

Academic provenance for the assessment, including institution, diploma, module, student, academic period, and lecturer.

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

Maintained original CA1 notebook plus split exploratory workflow notebooks. The original notebook is preserved as `00_original_ca1_submission.ipynb`, but reusable logic should live in `src/fault_prediction/`.

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
