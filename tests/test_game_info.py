"""Checks on data/game_info.csv, the table the team edits by hand.

If a spreadsheet export changes a value into something the app can't read
(a price as "$59.99", a typo in main_mode, a missing game), this fails the
pull request's check before the change reaches the live app.
"""
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
INFO = ROOT / "data" / "game_info.csv"
LAUNCH = ROOT / "processed" / "launch_daily.parquet"


@pytest.fixture(scope="module")
def info():
    return pd.read_csv(INFO)


def test_every_game_in_the_app_has_one_row(info):
    games = set(pd.read_parquet(LAUNCH, columns=["appid"])["appid"])
    assert set(info["appid"]) == games, "game_info.csv and launch_daily.parquet list different games"
    assert not info["appid"].duplicated().any()


def test_main_mode_values(info):
    allowed = {"Single-player", "Multiplayer", "Both"}
    bad = info.loc[~info["main_mode"].isin(allowed), ["game", "main_mode"]]
    assert bad.empty, f"main_mode must be one of {sorted(allowed)}:\n{bad}"


def test_launch_prices_are_plain_numbers(info):
    price = pd.to_numeric(info["launch_price_usd"], errors="coerce")
    bad = info.loc[price.isna() | ~price.between(0, 100), ["game", "launch_price_usd"]]
    assert bad.empty, f"launch_price_usd must be a number from 0 to 100 (no $ sign):\n{bad}"


def test_release_dates_are_iso(info):
    parsed = pd.to_datetime(info["release_date"], format="%Y-%m-%d", errors="coerce")
    bad = info.loc[parsed.isna(), ["game", "release_date"]]
    assert bad.empty, f"release_date must look like 2022-02-24:\n{bad}"
