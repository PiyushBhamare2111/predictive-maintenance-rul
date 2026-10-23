"""Streamlit AppTest: every page renders without exceptions (needs data + trained artifacts)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (ROOT / "models" / "FD001" / "metrics.json").exists()
    or not (ROOT / "data" / "CMAPSS" / "train_FD001.txt").exists(),
    reason="needs downloaded data and trained artifacts",
)


@pytest.mark.parametrize("page", ["Engine explorer", "Maintenance policy", "Model comparison", "About"])
def test_page_runs(page):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, at.exception


def test_engine_explorer_test_engines_and_other_subset():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    at.radio(key="source").set_value("Official test engines (cut off before failure)").run()
    assert not at.exception, at.exception
    if len(at.sidebar.selectbox[0].options) > 1:
        at.sidebar.selectbox[0].set_value("FD004").run()
        assert not at.exception, at.exception
