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
    # Filters start open; clear_filters is kept so older tests read the same.
    return AppTest.from_file(APP_FILE, default_timeout=120).run()


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
    at.button_group(key="layout").set_value("Side by side").run()
    assert_healthy(at)


def test_calendar_axis_players_and_short_window():
    at = open_app(clear_filters=True)
    at.multiselect(key="games").set_value(ALL_GAMES[:2]).run()
    at.button_group(key="axis").set_value("Calendar date").run()
    at.button_group(key="units").set_value("Players").run()
    at.slider(key="days").set_value(14).run()
    assert_healthy(at)


def test_no_games_selected():
    at = open_app()
    at.multiselect(key="games").set_value([]).run()
    assert_healthy(at)
    assert at.info


def test_filters_start_open_and_list_every_game():
    at = open_app()
    assert_healthy(at)
    assert at.selectbox(key="f_genre").value == "All genres"
    assert at.slider(key="f_price").value == (0, 70)
    assert len(at.multiselect(key="games").options) == 17
    assert not at.button                                         # no "Clear all" until a filter is on


def test_filters_never_drop_picked_games():
    at = open_app()                                              # Elden Ring picked
    at.selectbox(key="f_genre").set_value("Sports").run()        # Elden Ring isn't Sports
    assert_healthy(at)
    games = at.multiselect(key="games")
    assert games.value == [1245620]                              # still on the chart
    assert set(games.options) == {"Elden Ring", "Rocket League"} # picked + the one match
    assert "1 of 17 match the filters" in " ".join(m.value for m in at.sidebar.markdown)


def test_genre_counts_follow_other_filters():
    at = open_app()
    genre = at.selectbox(key="f_genre")
    assert any(o.startswith("RPG (") for o in genre.options)
    at.selectbox(key="f_mode").set_value("Multiplayer").run()
    rpg = next(o for o in at.selectbox(key="f_genre").options if o.startswith("RPG ("))
    assert rpg in ("RPG (2)", "RPG (3)")                         # multiplayer RPGs only


def test_filters_narrow_the_game_list():
    at = open_app()
    at.selectbox(key="f_mode").set_value("Multiplayer").run()
    assert_healthy(at)
    options = at.multiselect(key="games").options
    assert "Factorio" in options                  # "Both" counts as multiplayer too
    assert "Skyrim" not in " ".join(options)


def test_clear_all_resets_every_filter_including_sliders():
    at = open_app()
    at.slider(key="f_price").set_value((0, 0)).run()             # free games only
    at.slider(key="f_year").set_value((2020, 2022)).run()
    at.selectbox(key="f_mode").set_value("Single-player").run()
    assert_healthy(at)
    assert any("No games match" in i.value for i in at.info) or at.multiselect(key="games").value
    at.button(key="clear_filters").click().run()
    assert_healthy(at)
    assert at.slider(key="f_price").value == (0, 70)
    assert at.slider(key="f_year").value[0] <= 2011
    assert at.selectbox(key="f_mode").value == "All game types"
    assert len(at.multiselect(key="games").options) == 17
