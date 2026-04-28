import json
from pathlib import Path

EXPECTED_NOTEBOOKS = [
    "01_data_understanding.ipynb",
    "02_feature_engineering_and_statistics.ipynb",
    "03_training_and_thresholding.ipynb",
    "04_evaluation_and_prediction.ipynb",
]


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


def test_original_notebook_is_archived():
    archive_path = Path("notebooks/archive/original_factory_machine_status_classification.ipynb")
    assert archive_path.exists()
