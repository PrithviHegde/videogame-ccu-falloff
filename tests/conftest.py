"""Shared test setup.

app.py is a single Streamlit script, so importing it runs the whole page once
(Streamlit calls this "bare mode"; it logs warnings but works). That takes a
few seconds, so we do it once per test session and share the module.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def app():
    import app as app_module

    return app_module
