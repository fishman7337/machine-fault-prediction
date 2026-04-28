# Contributing

Thank you for improving the Predictive Maintenance Fault Classifier.

## Development Setup

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebook]"
```

## Workflow

1. Create a branch for the change.
2. Keep raw data in `data/raw/` and generated outputs in ignored folders such as `models/` or `reports/metrics/`.
3. Add or update tests for behavior changes.
4. Run `ruff check .` and `pytest`.
5. Document user-facing changes in `README.md` or `docs/`.

## Pull Request Checklist

- The project installs with `python -m pip install -e ".[dev]"`.
- Tests pass locally.
- Linting passes locally.
- Any change to feature engineering is documented in `docs/FEATURE_ENGINEERING.md`.
- Any change to model behavior updates `docs/MODEL_CARD.md`.
- Generated model artifacts are not committed.

## Coding Standards

- Keep pipeline behavior deterministic with `RANDOM_STATE`.
- Prefer scikit-learn transformers and pipelines over ad hoc preprocessing.
- Keep data leakage out of the workflow by fitting imputers, scalers, and encoders only on the training split.
- Keep notebooks for exploration and put reusable logic in `src/fault_prediction/`.
