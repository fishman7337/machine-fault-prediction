import json
from pathlib import Path

EXPECTED_NOTEBOOKS = [
    "01_data_understanding.ipynb",
    "02_feature_engineering_and_statistics.ipynb",
    "03_training_and_thresholding.ipynb",
    "04_evaluation_and_prediction.ipynb",
]
ORIGINAL_NOTEBOOK = "00_original_ca1_submission.ipynb"


def _load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_split_notebooks_exist_in_expected_order():
    notebook_dir = Path("notebooks")

    for notebook_name in EXPECTED_NOTEBOOKS:
        assert (notebook_dir / notebook_name).exists()

    discovered = [
        path.name
        for path in sorted(notebook_dir.glob("[0-9][0-9]_*.ipynb"))
        if path.name in EXPECTED_NOTEBOOKS
    ]
    assert discovered == EXPECTED_NOTEBOOKS


def test_split_notebooks_are_valid_and_output_free():
    notebook_dir = Path("notebooks")

    for notebook_name in EXPECTED_NOTEBOOKS:
        notebook = _load_notebook(notebook_dir / notebook_name)
        assert notebook["nbformat"] == 4
        assert notebook["cells"]

        for cell in notebook["cells"]:
            assert cell["cell_type"] in {"markdown", "code"}
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert cell["outputs"] == []


def test_split_notebooks_keep_original_report_style_markdown():
    notebook_dir = Path("notebooks")

    for notebook_name in EXPECTED_NOTEBOOKS:
        notebook = _load_notebook(notebook_dir / notebook_name)
        markdown = "\n".join(
            "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
        )
        markdown_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "markdown"]
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

        assert len(markdown_cells) >= len(code_cells) * 2
        assert "Factory Machine Status" in markdown
        assert "Notebook Objective" in markdown
        assert "<hr/>" in markdown
        assert "<h1>" in markdown
        assert "<h2>" in markdown
        assert "Interpretation:" in markdown
        assert "Within this section" in markdown


def test_original_notebook_is_maintained():
    original_path = Path("notebooks") / ORIGINAL_NOTEBOOK
    assert original_path.exists()
    original = _load_notebook(original_path)
    assert original["nbformat"] == 4
    assert len(original["cells"]) > 100
