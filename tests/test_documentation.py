import re
from pathlib import Path

REQUIRED_DOCS = [
    Path("README.md"),
    Path("CODE_OF_CONDUCT.md"),
    Path("CONTRIBUTING.md"),
    Path("SECURITY.md"),
    Path("LICENSE"),
    Path("docs/ACADEMIC_CONTEXT.md"),
    Path("docs/ASSESSMENT_REPORT.md"),
    Path("docs/DOCUMENTATION_INDEX.md"),
    Path("docs/DATA_CARD.md"),
    Path("docs/FEATURE_ENGINEERING.md"),
    Path("docs/MLOPS.md"),
    Path("docs/MODEL_CARD.md"),
    Path("docs/NOTEBOOKS.md"),
    Path("docs/PROJECT_STRUCTURE.md"),
    Path("docs/SECURITY_CHECKS.md"),
    Path("docs/STATISTICAL_TESTS.md"),
    Path("requirements.txt"),
    Path("requirements-dev.txt"),
    Path("requirements-notebook.txt"),
    Path("requirements-security.txt"),
    Path("requirements-all.txt"),
]


def _markdown_links(markdown: str) -> list[str]:
    return re.findall(r"\[[^\]]+\]\(([^)]+)\)", markdown)


def test_required_documentation_files_exist():
    for path in REQUIRED_DOCS:
        assert path.exists(), f"Missing documentation file: {path}"


def test_readme_local_documentation_links_resolve():
    readme = Path("README.md")
    for link in _markdown_links(readme.read_text(encoding="utf-8")):
        if link.startswith(("http://", "https://", "mailto:")):
            continue
        target = (readme.parent / link.split("#", maxsplit=1)[0]).resolve()
        assert target.exists(), f"README link does not resolve: {link}"


def test_assessment_report_preserves_academic_context():
    report = Path("docs/ASSESSMENT_REPORT.md").read_text(encoding="utf-8")
    required_phrases = [
        "Singapore Polytechnic",
        "Diploma in Applied AI & Analytics",
        "AI & Machine Learning",
        "ST1511",
        "CA1 Part A",
        "Goh Kun Ming",
        "Tai Hock Lin",
    ]
    for phrase in required_phrases:
        assert phrase in report


def test_ci_workflow_includes_security_checks():
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    required_phrases = [
        "Security checks",
        "bandit -c pyproject.toml -r src",
        "pip-audit --skip-editable",
    ]
    for phrase in required_phrases:
        assert phrase in workflow
