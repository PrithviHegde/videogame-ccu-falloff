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
