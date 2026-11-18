"""Smoke tests: does the app load, and do the main controls work?

These run the real app.py against processed/launch_daily.parquet.
They don't check that the numbers are right, only that nothing crashes
and no red error box is shown.
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_FILE = str(Path(__file__).resolve().parent.parent / "app.py")


def assert_healthy(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def radio(at: AppTest, label: str):
    return next(r for r in at.radio if r.label == label)


def open_app() -> AppTest:
    return AppTest.from_file(APP_FILE, default_timeout=120).run()


# AppIDs in processed/launch_daily.parquet, in build_launch_data.GAMES order.
ALL_GAMES = [1245620, 1091500, 2358720, 377160, 72850]


def test_default_view_one_game_180_days():
    at = open_app()
    assert_healthy(at)
    assert at.multiselect[0].value == [ALL_GAMES[0]]
    assert at.slider[0].value == 180
    cards = " ".join(m.value for m in at.markdown)
    for stage in ("Launch", "Falloff", "Settled"):   # single-game stage cards
        assert stage in cards


def test_comparing_all_games_both_layouts():
    at = open_app()
    at.multiselect[0].set_value(ALL_GAMES).run()
    assert_healthy(at)
    assert any("stages" in c.value for c in at.caption)   # stage strip under the chart
    radio(at, "Layout").set_value("Side by side").run()
    assert_healthy(at)


def test_calendar_axis_players_and_short_window():
    at = open_app()
    at.multiselect[0].set_value(ALL_GAMES[:2]).run()
    radio(at, "Time axis").set_value("Calendar date").run()
    radio(at, "Show").set_value("Players").run()
    at.slider[0].set_value(14).run()
    assert_healthy(at)


def test_no_games_selected():
    at = open_app()
    at.multiselect[0].set_value([]).run()
    assert_healthy(at)
    assert at.info
