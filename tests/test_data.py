"""Unit tests for the data logic in app.py, on tiny hand-made tables.

Each test builds a few rows where the right answer is obvious, then checks
that the function returns it.
"""
import numpy as np
import pandas as pd
import pytest


# ----------------------------- clean_months -----------------------------

def test_clean_months_drops_bad_rows(app):
    raw = pd.DataFrame({
        "appid":      [10,        10,        "oops",    0,         20,        30],
        "year_month": ["2020-01", "2020-02", "2020-01", "2020-01", "not-a-month", "2020-01"],
        "avg_ccu":    [100,       80,        50,        50,        50,        -5],
        "peak_ccu":   [200,       150,       90,        90,        90,        10],
    })
    out = app.clean_months(raw)
    # Only the two valid rows for appid 10 survive.
    assert out["appid"].tolist() == [10, 10]
    assert out["year_month"].tolist() == ["2020-01", "2020-02"]


def test_clean_months_keeps_last_duplicate_and_sorts(app):
    raw = pd.DataFrame({
        "appid":      [10,        10,        10],
        "year_month": ["2020-02", "2020-01", "2020-01"],
        "avg_ccu":    [80,        100,       111],
        "peak_ccu":   [150,       200,       222],
    })
    out = app.clean_months(raw)
    assert out["year_month"].tolist() == ["2020-01", "2020-02"]
    assert out["avg_ccu"].tolist() == [111, 80]


def test_clean_months_adds_month_ordinal_and_blanks_negative_peak(app):
    raw = pd.DataFrame({
        "appid":      [10,        10],
        "year_month": ["2019-12", "2020-01"],
        "avg_ccu":    [100,       90],
        "peak_ccu":   [-1,        180],
    })
    out = app.clean_months(raw)
    # December -> January is one month apart, even across a year boundary.
    assert out["month_ordinal"].diff().iloc[1] == 1
    assert np.isnan(out["peak_ccu"].iloc[0])
    assert out["peak_ccu"].iloc[1] == 180


# ----------------------------- filter_games -----------------------------

@pytest.fixture
def games():
    return pd.DataFrame({
        "appid":      [1,               2,          3],
        "genres":     ["Action | RPG",  "JRPG",     ""],
        "publishers": ["Valve",         "Square Enix", "Valve Corporation"],
    }).set_index("appid", drop=False)


def test_filter_games_no_filters_returns_everything(app, games):
    assert len(app.filter_games(games, "All genres", "")) == 3


def test_filter_games_genre_is_exact_not_substring(app, games):
    out = app.filter_games(games, "RPG", "")
    # "JRPG" contains "RPG" but must not match.
    assert out["appid"].tolist() == [1]


def test_filter_games_publisher_is_case_insensitive_substring(app, games):
    out = app.filter_games(games, "All genres", "  valve ")
    assert out["appid"].tolist() == [1, 3]


def test_filter_games_combines_genre_and_publisher(app, games):
    assert app.filter_games(games, "JRPG", "valve").empty


# ----------------------------- monthly_stats -----------------------------

def make_view(rows):
    """rows: (appid, year_month, avg_ccu, peak_ccu). x_position = calendar month."""
    view = pd.DataFrame(rows, columns=["appid", "year_month", "avg_ccu", "peak_ccu"])
    dates = pd.to_datetime(view["year_month"] + "-01")
    view["month_ordinal"] = dates.dt.year * 12 + dates.dt.month
    view["x_position"] = view["month_ordinal"]
    return view


def test_monthly_stats_basic_numbers(app):
    view = make_view([
        (10, "2020-01", 100.0, 300.0),
        (10, "2020-02", 50.0, 120.0),
        (10, "2020-03", 40.0, 90.0),
    ])
    row = app.monthly_stats(view, "Calendar months", 12).iloc[0]
    assert row["months"] == 3
    assert row["first_month"] == "2020-01"
    assert row["last_month"] == "2020-03"
    assert row["peak_ccu"] == 300
    assert row["peak_monthly_average"] == 100
    assert row["mean_ccu"] == pytest.approx(190 / 3)
    assert row["end_to_peak_pct"] == pytest.approx(40.0)            # 40 / 100
    assert row["largest_monthly_decline_pct"] == pytest.approx(-50.0)  # 100 -> 50


def test_monthly_stats_ignores_drops_across_missing_months(app):
    view = make_view([
        (10, "2020-01", 100.0, 100.0),
        # February is missing, so 100 -> 10 must not count as a monthly drop.
        (10, "2020-03", 10.0, 10.0),
        (10, "2020-04", 9.0, 9.0),
    ])
    row = app.monthly_stats(view, "Calendar months", 12).iloc[0]
    assert row["largest_monthly_decline_pct"] == pytest.approx(-10.0)  # 10 -> 9


def test_monthly_stats_keeps_games_separate(app):
    view = make_view([
        (10, "2020-01", 100.0, 100.0),
        (10, "2020-02", 90.0, 90.0),
        (20, "2020-01", 5.0, 8.0),
        (20, "2020-02", 10.0, 12.0),
    ])
    stats = app.monthly_stats(view, "Calendar months", 12).set_index("appid")
    assert stats.loc[10, "largest_monthly_decline_pct"] == pytest.approx(-10.0)
    assert stats.loc[20, "largest_monthly_decline_pct"] == pytest.approx(100.0)
    assert stats.loc[20, "end_to_peak_pct"] == pytest.approx(100.0)


def test_monthly_stats_release_view_needs_full_window_for_end_ratio(app):
    view = make_view([
        (10, "2020-01", 100.0, 100.0),
        (10, "2020-02", 60.0, 60.0),
        (10, "2020-03", 30.0, 30.0),
        (20, "2020-01", 100.0, 100.0),
        (20, "2020-02", 60.0, 60.0),
    ])
    # In the release view, x_position counts months since release from 0.
    view["x_position"] = view.groupby("appid").cumcount()
    stats = app.monthly_stats(view, "Months since release", 3).set_index("appid")
    assert stats.loc[10, "end_to_peak_pct"] == pytest.approx(30.0)
    # Game 20 never reaches the last month of the 3-month window.
    assert np.isnan(stats.loc[20, "end_to_peak_pct"])


def test_monthly_stats_empty_view_returns_empty_table_with_columns(app):
    view = make_view([])
    stats = app.monthly_stats(view, "Calendar months", 12)
    assert stats.empty
    assert "end_to_peak_pct" in stats.columns


# ------------------------------ rank_stats ------------------------------

@pytest.fixture
def stats():
    return pd.DataFrame({
        "appid":                       [1,     2,     3,      4],
        "months":                      [12,    12,    12,     1],
        "peak_ccu":                    [500.0, 900.0, 100.0,  9999.0],
        "mean_ccu":                    [300.0, 200.0, 50.0,   9999.0],
        "end_to_peak_pct":             [40.0,  np.nan, 80.0,  100.0],
        "largest_monthly_decline_pct": [-30.0, -70.0, -5.0,   np.nan],
    })


def test_rank_stats_peak_is_highest_first_and_respects_minimum_months(app, stats):
    out = app.rank_stats(stats, "Observed peak CCU", minimum_months=3)
    # Game 4 has the biggest peak but only 1 observed month.
    assert out["appid"].tolist() == [2, 1, 3]


def test_rank_stats_average_ccu(app, stats):
    out = app.rank_stats(stats, "Average CCU", minimum_months=3)
    assert out["appid"].tolist() == [1, 2, 3]


def test_rank_stats_steepest_drop_is_most_negative_first(app, stats):
    out = app.rank_stats(stats, "Steepest monthly drop", minimum_months=3)
    assert out["appid"].tolist() == [2, 1, 3]


def test_rank_stats_drops_games_missing_the_ranked_value(app, stats):
    out = app.rank_stats(stats, "End / peak ratio", minimum_months=3)
    assert out["appid"].tolist() == [3, 1]
