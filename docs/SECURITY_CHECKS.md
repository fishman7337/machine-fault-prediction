# Security Checks

The repository uses two automated security checks in GitHub Actions.

## Static Security Scan

Bandit scans the production source code under `src/` for common Python security risks.

Run locally:

```bash
python -m pip install -r requirements-security.txt
bandit -c pyproject.toml -r src
```

## Dependency Vulnerability Audit

`pip-audit` checks installed Python dependencies against known vulnerability advisories.
Run it from a clean virtual environment so unrelated globally installed packages do not affect the result.

Run locally:

```bash
python -m pip install -r requirements-security.txt
pip-audit --skip-editable
```

## CI Coverage

The GitHub Actions workflow has two jobs:

- `test`: installs development dependencies, runs Ruff, and runs pytest on Python 3.10, 3.11, and 3.12.
- `security`: installs security dependencies, runs Bandit, and runs `pip-audit` on Python 3.12.

## Security Expectations

- Do not commit secrets, credentials, or private operational data.
- Do not commit generated model artifacts.
- Keep dependency files updated when changing package extras.
- Treat security scan failures as release blockers unless a documented false positive is reviewed and accepted.
