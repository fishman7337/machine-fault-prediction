# Security Policy

## Supported Versions

This repository currently supports the latest version on the `main` branch.

## Reporting a Vulnerability

Please report suspected vulnerabilities privately to the project maintainer. Include:

- A concise description of the issue.
- Steps to reproduce.
- Impact on data, model artifacts, or deployment.
- Suggested fix, if known.

## Data Handling

Do not commit secrets, production credentials, private customer data, or sensitive operational data. The included dataset is treated as project sample data. Generated model files and predictions are ignored by default.

## Automated Checks

Security automation is documented in [docs/SECURITY_CHECKS.md](docs/SECURITY_CHECKS.md). CI runs Bandit static analysis and `pip-audit` dependency vulnerability auditing.
