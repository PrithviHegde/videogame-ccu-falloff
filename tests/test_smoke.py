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


def open_app(clear_filters: bool = False) -> AppTest:
    at = AppTest.from_file(APP_FILE, default_timeout=120).run()
    if clear_filters:     # on load, the filters frame the default game
        at.button[0].click().run()
    return at


# AppIDs in processed/launch_daily.parquet, in build_launch_data.GAMES order.
ALL_GAMES = [1245620, 1091500, 2358720, 377160, 72850]


def test_default_view_one_game_180_days():
    at = open_app()
    assert_healthy(at)
    assert at.multiselect(key="games").value == [ALL_GAMES[0]]
    assert at.slider(key="days").value == 180
    cards = " ".join(m.value for m in at.markdown)
    for stage in ("Launch", "Falloff", "Settled"):   # single-game stage cards
        assert stage in cards


def test_comparing_all_games_both_layouts():
    at = open_app(clear_filters=True)
    at.multiselect(key="games").set_value(ALL_GAMES).run()
    assert_healthy(at)
    assert any("stages" in c.value for c in at.caption)   # stage strip under the chart
    radio(at, "Layout").set_value("Side by side").run()
    assert_healthy(at)


def test_calendar_axis_players_and_short_window():
    at = open_app(clear_filters=True)
    at.multiselect(key="games").set_value(ALL_GAMES[:2]).run()
    radio(at, "Time axis").set_value("Calendar date").run()
    radio(at, "Show").set_value("Players").run()
    at.slider(key="days").set_value(14).run()
    assert_healthy(at)


def test_no_games_selected():
    at = open_app()
    at.multiselect(key="games").set_value([]).run()
    assert_healthy(at)
    assert at.info


def test_filters_follow_the_default_game():
    at = open_app()
    assert_healthy(at)
    assert at.selectbox(key="f_genre").value == "RPG"           # Elden Ring is Action|RPG
    assert at.selectbox(key="f_mode").value == "Single-player"
    options = at.multiselect(key="games").options
    assert "Elden Ring" in options and "Cyberpunk 2077" in options
    assert "Rocket League" not in options                        # not similar


def test_picking_games_moves_the_filters():
    at = open_app()
    at.multiselect(key="games").set_value([1245620, 1091500]).run()   # + Cyberpunk (offered)
    assert_healthy(at)
    assert at.slider(key="f_year").value == (2018, 2024)        # 2020-2022, ± 2 years
    assert at.multiselect(key="games").value == [1245620, 1091500]
    at.button[0].click().run()                                  # Clear filters: all games again
    at.multiselect(key="games").set_value([1245620, 1091500, 377160, 72850]).run()
    assert_healthy(at)
    assert at.slider(key="f_year").value[0] <= 2011             # now framing Skyrim (2011) too
    assert len(at.multiselect(key="games").value) == 4


def test_filters_narrow_the_game_list():
    at = open_app(clear_filters=True)
    at.selectbox(key="f_mode").set_value("Multiplayer").run()
    assert_healthy(at)
    options = at.multiselect(key="games").options
    assert 0 < len(options) < len(ALL_GAMES) + 20
    assert all(name not in options for name in ("Elden Ring", "Skyrim"))
    assert "Factorio" in options                  # "Both" counts as multiplayer too
    assert at.multiselect(key="games").value      # a matching game is picked, not an empty chart


def test_genre_and_publisher_filters():
    at = open_app(clear_filters=True)
    at.selectbox(key="f_genre").set_value("Sports").run()
    assert at.multiselect(key="games").options == ["Rocket League"]
    at.selectbox(key="f_genre").set_value("All genres").run()
    at.radio(key="f_publisher").set_value("Bethesda Softworks").run()
    assert_healthy(at)
    assert "Fallout 4" in at.multiselect(key="games").options
    assert "Elden Ring" not in at.multiselect(key="games").options


def test_filters_with_no_match_and_clear():
    at = open_app(clear_filters=True)
    at.slider(key="f_price").set_value((0, 0)).run()            # free games only
    at.selectbox(key="f_mode").set_value("Single-player").run()
    assert_healthy(at)
    assert not at.multiselect(key="games").options
    assert any("No games match" in i.value for i in at.info)
    at.button[0].click().run()                                  # Clear filters
    assert_healthy(at)
    assert len(at.multiselect(key="games").options) >= len(ALL_GAMES)
