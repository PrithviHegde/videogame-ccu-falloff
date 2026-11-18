"""Build processed/launch_daily.parquet from SteamDB chart exports.

1. On each game's SteamDB charts page (https://steamdb.info/app/<appid>/charts/),
   set the range to "All" and export the data. Download it by hand: SteamDB does
   not allow scraping or automated downloads.
2. Save the file as data/raw/steamdb/steamdb_chart_<appid>.csv (the default name).
   data/raw/ is git-ignored: raw exports are never committed.
3. Add the game to GAMES below, then run:
       python build_launch_data.py

The output keeps only the first DAYS days after launch for each game: one row
per day with that day's peak number of concurrent players.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw" / "steamdb"
OUT_PATH = ROOT / "processed" / "launch_daily.parquet"
DAYS = 180

# AppID -> display name. Order here is the order games appear in the app.
GAMES = {
    1245620: "Elden Ring",
    1091500: "Cyberpunk 2077",
    2358720: "Black Myth: Wukong",
    377160: "Fallout 4",
    72850: "The Elder Scrolls V: Skyrim",
}


def read_steamdb_export(path: Path) -> pd.Series:
    """Daily peak players from a SteamDB chart export, indexed by UTC date.

    Exports are daily for older history and every 10 minutes for the latest
    year; both are reduced to one value per day (the day's highest count).
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "DateTime" not in df or "Players" not in df:
        raise ValueError(f"{path.name}: expected DateTime and Players columns, got {list(df.columns)}")
    when = pd.to_datetime(df["DateTime"], errors="coerce")
    players = pd.to_numeric(df["Players"].astype(str).str.replace(",", ""), errors="coerce")
    s = pd.Series(players.to_numpy(), index=when).dropna()
    s = s[s.index.notna()]
    return s.groupby(s.index.normalize()).max().sort_index()


def launch_day(daily: pd.Series, share: float = 0.10, sustain: int = 3) -> pd.Timestamp:
    """First day that starts `sustain` days in a row at or above `share` of the
    game's highest-ever daily peak.

    SteamDB days are UTC, so this is usually the store release date or the day
    after. Small pre-release counts (press copies, early unlocks in some
    regions) stay below the threshold, and a one-day glitch can't pass for
    launch (Fallout 4's export has a lone 471,955 on Nov 1, 2015, nine days
    before release, between days of single-digit counts).
    """
    if daily.empty or daily.max() <= 0:
        raise ValueError("no players recorded")
    days = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    above = daily.reindex(days).ge(share * daily.max())
    run = above.rolling(sustain).sum().shift(-(sustain - 1)).eq(sustain)
    if not run.any():
        raise ValueError(f"no {sustain}-day run above {share:.0%} of peak")
    return run.idxmax()


def launch_window(daily: pd.Series, days: int = DAYS) -> pd.DataFrame:
    """The first `days` days from launch, one row per calendar day.

    Days with no export value stay missing (NaN); nothing is filled in.
    """
    start = launch_day(daily)
    dates = pd.date_range(start, periods=days, freq="D")
    return pd.DataFrame({
        "day": range(days),
        "date": dates,
        "players": daily.reindex(dates).to_numpy(),
    })


def build() -> pd.DataFrame:
    frames = []
    for appid, name in GAMES.items():
        path = RAW_DIR / f"steamdb_chart_{appid}.csv"
        if not path.is_file():
            raise SystemExit(f"Missing {path.relative_to(ROOT)}: download it from SteamDB first.")
        window = launch_window(read_steamdb_export(path))
        missing = int(window["players"].isna().sum())
        print(f"{name:32s} launch {window['date'].iloc[0]:%Y-%m-%d}  "
              f"peak {window['players'].max():>11,.0f}  missing days {missing}")
        frames.append(window.assign(appid=appid, game=name))
    out = pd.concat(frames, ignore_index=True)[["appid", "game", "day", "date", "players"]]
    out["appid"] = out["appid"].astype("int32")
    out["day"] = out["day"].astype("int16")
    return out


if __name__ == "__main__":
    data = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {len(data):,} rows to {OUT_PATH.relative_to(ROOT)} "
          f"({OUT_PATH.stat().st_size / 1024:.0f} KB)")
