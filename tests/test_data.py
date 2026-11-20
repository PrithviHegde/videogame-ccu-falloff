"""Unit tests for the data logic, on tiny hand-made tables.

Each test builds a few rows where the right answer is obvious, then checks
that the function returns it.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def build():
    spec = importlib.util.spec_from_file_location("build_launch_data", ROOT / "build_launch_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def daily(values, start="2024-01-01"):
    return pd.Series([float(v) for v in values],
                     index=pd.date_range(start, periods=len(values), freq="D"))


# ------------------------ read_steamdb_export ------------------------

def test_read_export_parses_commas_and_takes_daily_max(build, tmp_path):
    f = tmp_path / "steamdb_chart_1.csv"
    f.write_text(
        '﻿"DateTime","Players","Average Players"\n'
        '"2024-01-01 00:00:00","1,500",\n'
        '"2024-01-02 00:00:00",900,\n'
        '"2024-01-03 00:00:00",,\n'           # blank: dropped, not zero
        '"2024-01-04 00:10:00",100,90\n'      # 10-minute rows: keep the day's max
        '"2024-01-04 00:20:00",300,90\n',
        encoding="utf-8",
    )
    s = build.read_steamdb_export(f)
    assert s.to_dict() == {
        pd.Timestamp("2024-01-01"): 1500.0,
        pd.Timestamp("2024-01-02"): 900.0,
        pd.Timestamp("2024-01-04"): 300.0,
    }


def test_read_export_rejects_wrong_columns(build, tmp_path):
    f = tmp_path / "bad.csv"
    f.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        build.read_steamdb_export(f)


# ---------------------------- launch_day ----------------------------

def test_launch_day_is_the_release_date(build):
    s = daily([3, 900, 1000, 800], start="2024-01-01")
    assert build.launch_day(s, "2024-01-02") == pd.Timestamp("2024-01-02")


def test_launch_day_keeps_release_date_even_if_small(build):
    # A small launch (like Hollow Knight) still starts on its release date.
    s = daily([0, 40, 60, 90, 80], start="2024-01-01")
    assert build.launch_day(s, "2024-01-02") == pd.Timestamp("2024-01-02")


def test_launch_day_moves_to_next_day_when_release_date_is_empty(build):
    # Unlocked after midnight UTC: nothing recorded on the official date.
    s = daily([900, 1000, 800], start="2024-01-03")
    assert build.launch_day(s, "2024-01-02") == pd.Timestamp("2024-01-03")


def test_launch_day_moves_to_next_day_when_release_date_is_near_zero(build):
    # Like Fallout 4: a few early unlocks on the official date, launch the next day.
    s = daily([14, 450, 470, 430], start="2024-01-02")
    assert build.launch_day(s, "2024-01-02") == pd.Timestamp("2024-01-03")


# --------------------------- launch_window ---------------------------

def test_launch_window_starts_at_release_and_keeps_gaps(build):
    s = daily([1, 500, 400, 300, 200, 100])
    s = s.drop(pd.Timestamp("2024-01-05"))     # one missing day after launch
    w = build.launch_window(s, "2024-01-02", days=5)
    assert w["day"].tolist() == [0, 1, 2, 3, 4]
    assert w["date"].iloc[0] == pd.Timestamp("2024-01-02")
    assert np.isnan(w["players"].iloc[3])      # not filled in
    assert w["players"].iloc[4] == 100


# ---------------------------- check_window ----------------------------

def test_check_window_rejects_data_starting_after_release(build):
    s = daily([500, 400, 300], start="2024-01-10")
    w = build.launch_window(s, "2024-01-01", days=3)
    assert "after the release date" in build.check_window(s, w, "2024-01-01")


def test_check_window_rejects_too_many_missing_days(build, monkeypatch):
    monkeypatch.setattr(build, "MAX_MISSING_DAYS", 1)
    monkeypatch.setattr(build, "DAYS", 5)
    s = daily([500, 400, 300, 200, 100]).drop(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    w = build.launch_window(s, "2024-01-01", days=5)
    assert "no data" in build.check_window(s, w, "2024-01-01")


def test_check_window_accepts_complete_data(build):
    s = daily([500, 400, 300, 200, 100])
    w = build.launch_window(s, "2024-01-01", days=5)
    assert build.check_window(s, w, "2024-01-01") is None


# --------------------------- app: smoothing ---------------------------

def game_rows(values, appid=1):
    return pd.DataFrame({
        "appid": appid,
        "game": "Test",
        "day": range(len(values)),
        "date": pd.date_range("2024-01-01", periods=len(values), freq="D"),
        "players": [float(v) for v in values],
    })


def test_add_smoothing_pct_and_rolling_mean(app):
    out = app.add_smoothing(game_rows([100, 50, 50, 50, 50]), window=3)
    assert out["pct"].tolist() == [100, 50, 50, 50, 50]
    assert out["smooth"].iloc[1] == pytest.approx(200 / 3)   # mean of days 0-2
    assert out["smooth"].iloc[3] == pytest.approx(50)


def test_add_smoothing_keeps_games_separate(app):
    both = pd.concat([game_rows([10, 10, 10], appid=1), game_rows([1000, 500, 250], appid=2)])
    out = app.add_smoothing(both, window=3).set_index(["appid", "day"])
    assert out.loc[(1, 1), "smooth"] == pytest.approx(10)
    assert out.loc[(2, 2), "pct"] == pytest.approx(25)


# ------------------------- app: leveling-off day -------------------------

def series(values):
    return pd.Series([float(v) for v in values], index=range(len(values)))


def test_leveling_off_day_finds_the_bend(app):
    # Steep fall for 4 days, then nearly flat.
    assert app.leveling_off_day(series([100, 75, 50, 25, 10, 9, 8, 7, 6, 5])) == 4


def test_leveling_off_day_starts_from_the_peak(app):
    # Rises to a peak on day 2 first; the rise is not part of the falloff.
    assert app.leveling_off_day(series([20, 60, 100, 50, 20, 15, 12, 10, 9, 8])) == 4


def test_leveling_off_day_none_without_a_bend(app):
    assert app.leveling_off_day(series([5, 5, 5, 5, 5])) is None          # flat
    assert app.leveling_off_day(series([100, 80, 60, 40, 20, 0])) is None  # straight line


# --------------------------- app: game_stages ---------------------------

def test_game_stages_numbers(app):
    values = [80, 100, 75, 50, 25, 10, 10, 10, 10, 10]
    g = app.add_smoothing(game_rows(values), window=1)   # window 1 = no smoothing
    s = app.game_stages(g, tail=3)
    assert s["peak"] == 100 and s["peak_day"] == 1
    assert s["falloff_start"] == 1
    assert s["knee"] == 5
    assert s["falloff_days"] == 4
    assert s["lost_pct"] == pytest.approx(90)     # 100 -> 10
    assert s["tail_start"] == 7
    assert s["settled"] == pytest.approx(10)      # average of the last 3 days
    assert s["settled_pct"] == pytest.approx(10)
    assert s["half_life_days"] == pytest.approx(4 * np.log(0.5) / np.log(0.1))


def test_game_stages_settled_is_where_it_ends_up(app):
    # Still sliding after the bend: settled uses the last days, not the whole stage.
    values = [100, 50, 20, 18, 16, 14, 12, 10, 8, 6]
    g = app.add_smoothing(game_rows(values), window=1)
    s = app.game_stages(g, tail=2)
    assert s["settled"] == pytest.approx(7)       # mean of 8 and 6


def test_game_stages_flat_game_has_no_falloff(app):
    g = app.add_smoothing(game_rows([5, 5, 5, 5, 5]), window=1)
    s = app.game_stages(g, tail=2)
    assert not s["has_falloff"] and s["knee"] is None and s["lost_pct"] is None
    assert s["settled"] == pytest.approx(5)
    assert s["shape"] == app.GROWS          # ends level with its launch week


def test_game_stages_shape_falls_off(app):
    g = app.add_smoothing(game_rows([100, 90, 60, 30, 15, 10, 10, 10, 10, 10]), window=1)
    s = app.game_stages(g, tail=3, late_peak_day=3, grows_peak_day=6)
    assert s["shape"] == app.FALLS_OFF
    assert not s["has_growth"] and s["has_falloff"]


def test_game_stages_shape_peaks_late(app):
    # Climbs to day 4, then falls well below launch week.
    values = [40, 60, 80, 90, 100, 60, 30, 15, 10, 10, 10, 10]
    g = app.add_smoothing(game_rows(values), window=1)
    s = app.game_stages(g, tail=3, late_peak_day=3, grows_peak_day=8)
    assert s["shape"] == app.LATE_PEAK
    assert s["has_growth"] and s["growth_day"] == 4
    assert s["has_falloff"] and s["falloff_start"] == 4


def test_game_stages_shape_grows(app):
    values = [10, 12, 15, 20, 30, 40, 50, 60, 70, 80]
    g = app.add_smoothing(game_rows(values), window=1)
    s = app.game_stages(g, tail=2, late_peak_day=3, grows_peak_day=6)
    assert s["shape"] == app.GROWS
    assert s["has_growth"] and s["growth_day"] == 9 and not s["has_falloff"]
    launch_week = sum(values[:7]) / 7
    assert s["launch_week"] == pytest.approx(launch_week)
    assert s["end_vs_launch"] == pytest.approx(100 * 75 / launch_week)


def test_game_stages_small_dip_is_not_a_falloff(app):
    # Grows, then dips 10%: below MIN_FALLOFF_PCT, so no falloff stage.
    values = [10, 20, 40, 80, 100, 98, 95, 92, 90, 90, 90, 90]
    g = app.add_smoothing(game_rows(values), window=1)
    s = app.game_stages(g, tail=3, late_peak_day=3, grows_peak_day=6)
    assert not s["has_falloff"]


def test_change_and_vs_launch_text(app):
    assert app.change_text(1.95) == "+95%"
    assert app.change_text(4.2) == "4.2×"
    assert app.vs_launch_text(5) == "5% of launch week"
    assert app.vs_launch_text(414) == "4.1× launch week"


def test_compact_numbers(app):
    assert app.compact(950) == "950"
    assert app.compact(1_234) == "1.2k"
    assert app.compact(67_401) == "67k"
    assert app.compact(2_415_714) == "2.4M"


# ------------------------------ app: filters ------------------------------

@pytest.fixture
def info(app):
    raw = pd.DataFrame({
        "appid": [1, 2, 3, 4],
        "game": ["Solo RPG", "Shooter", "Co-op Sim", "Odd Tags"],
        "release_date": ["2015-11-10", "2020-11-04", "2020-08-14", "2024-12-05"],
        "genres": ["Action|RPG", "Action|Free To Play", "Simulation|Indie", "Racing|Strategy|Free To Play"],
        "genres_override": [None, None, "", "Action"],
        "publishers": ["Big Co|Partner", "Big Co", "Small Studio", "Other"],
        "main_mode": ["Single-player", "Multiplayer", "Both", "Multiplayer"],
        "launch_price_usd": [59.99, 0.0, 30.0, None],
    })
    return app.prepare_info(raw)


def test_prepare_info_genres(app, info):
    assert info.loc[2, "genre_list"] == ["Action"]          # "Free To Play" is not a genre
    assert info.loc[3, "genre_list"] == ["Simulation", "Indie"]   # blank override: Steam's genres
    assert info.loc[4, "genre_list"] == ["Action"]          # override replaces Steam's tags
    assert info.loc[1, "publisher_list"] == ["Big Co", "Partner"]
    assert info.loc[1, "year"] == 2015


def test_filter_games_no_filters_keeps_everything(app, info):
    assert app.filter_games(info) == {1, 2, 3, 4}


def test_filter_games_genre_and_publisher_match_any(app, info):
    assert app.filter_games(info, genres=["Action"]) == {1, 2, 4}
    assert app.filter_games(info, publishers=["Partner", "Small Studio"]) == {1, 3}


def test_filter_games_mode_year_price(app, info):
    assert app.filter_games(info, modes=["Multiplayer"]) == {2, 4}
    assert app.filter_games(info, years=(2016, 2021)) == {2, 3}
    assert app.filter_games(info, prices=(0, 0)) == {2}            # free games
    assert app.filter_games(info, prices=(0, 70)) == {1, 2, 3}     # unknown price is left out


def test_filter_games_combines_filters(app, info):
    assert app.filter_games(info, genres=["Action"], modes=["Single-player"]) == {1}


def test_game_meta(app, info):
    assert app.game_meta(info, 1) == ("Action, RPG · Big Co, Partner · $59.99 at launch · "
                                      "Single-player · 2015 · not yet reviewed")
    assert "Free to play" in app.game_meta(info, 2)
    assert app.game_meta(info, 99) == ""


def test_reviewed_flag_from_looks_good_column(app):
    raw = pd.DataFrame({
        "appid": [1, 2, 3], "game": ["A", "B", "C"],
        "release_date": ["2020-01-01"] * 3, "genres": ["RPG"] * 3, "publishers": ["X"] * 3,
        "main_mode": ["Single-player"] * 3, "launch_price_usd": [10, 10, 10],
        "Looks Good?": ["Yes", None, "TRUE"],
    })
    info = app.prepare_info(raw)
    assert info["reviewed"].tolist() == [True, False, True]
    assert app.game_meta(info, 2).endswith("not yet reviewed")
    assert not app.game_meta(info, 1).endswith("not yet reviewed")


# ------------------------------ app: snapping ------------------------------

SPANS = dict(price_span=(0, 70), year_span=(2010, 2025))


def test_snap_one_game_frames_it_loosely(app, info):
    s = app.snap_filters(info, [1], **SPANS)              # Solo RPG, $59.99, 2015
    assert s["f_price"] == (45, 70)                         # $59.99 ± $10, to $5 steps
    assert s["f_year"] == (2013, 2017)                      # 2015 ± 2
    assert s["f_genre"] == "Action"                         # Action (3 games) beats RPG (1)
    assert s["f_mode"] == "Single-player"
    assert s["f_publisher"] == app.ANY_PUBLISHER            # not snapped for one game


def test_snap_keeps_the_selected_games_in_the_list(app, info):
    for sel in ([1], [2], [1, 2], [2, 3]):
        s = app.snap_filters(info, sel, **SPANS)
        allowed = app.filter_games(
            info,
            genres=[s["f_genre"]] if s["f_genre"] != app.ANY_GENRE else (),
            publishers=[s["f_publisher"]] if s["f_publisher"] != app.ANY_PUBLISHER else (),
            modes=[s["f_mode"], "Both"] if s["f_mode"] != app.ANY_TYPE else (),
            years=s["f_year"], prices=s["f_price"])
        assert set(sel) <= allowed, sel


def test_snap_mixed_selection_leaves_unshared_filters_open(app, info):
    s = app.snap_filters(info, [1, 3], **SPANS)             # Action|RPG + Simulation|Indie
    assert s["f_genre"] == app.ANY_GENRE
    assert s["f_mode"] == "Single-player"                   # Single-player + Both
    s = app.snap_filters(info, [1, 2], **SPANS)             # Single-player + Multiplayer
    assert s["f_mode"] == app.ANY_TYPE
    assert s["f_publisher"] == "Big Co"                     # shared by 2+ games


def test_snap_unknown_price_leaves_price_open(app, info):
    assert app.snap_filters(info, [4], **SPANS)["f_price"] == (0, 70)


def test_snap_nothing_selected_is_all_defaults(app, info):
    s = app.snap_filters(info, [], **SPANS)
    assert s == {"f_price": (0, 70), "f_year": (2010, 2025), "f_genre": app.ANY_GENRE,
                 "f_mode": app.ANY_TYPE, "f_publisher": app.ANY_PUBLISHER}
