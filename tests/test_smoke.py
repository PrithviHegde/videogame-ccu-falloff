"""Smoke tests: does the app load at all, on both views?

These run the real app.py against the real parquet files in processed/.
They don't check that the numbers are right, only that nothing crashes
and no red error box is shown.
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_FILE = str(Path(__file__).resolve().parent.parent / "app.py")


def assert_healthy(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    # The app catches data-loading failures and shows st.error instead of
    # raising, so an error box also counts as a failure.
    assert not at.error, [e.value for e in at.error]


def test_monthly_trends_loads():
    at = AppTest.from_file(APP_FILE, default_timeout=120).run()
    assert_healthy(at)
    assert at.sidebar.radio[0].value == "Monthly Trends"


def test_months_since_release_loads():
    at = AppTest.from_file(APP_FILE, default_timeout=120).run()
    compare_by = next(r for r in at.sidebar.radio if r.label == "Compare by")
    compare_by.set_value("Months since release").run()
    assert_healthy(at)


def test_daily_deep_dive_loads():
    at = AppTest.from_file(APP_FILE, default_timeout=120).run()
    at.sidebar.radio[0].set_value("Daily Deep Dive").run()
    assert_healthy(at)
