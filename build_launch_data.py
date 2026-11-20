"""Build processed/launch_daily.parquet from SteamDB chart exports.

1. On each game's SteamDB charts page (https://steamdb.info/app/<appid>/charts/),
   set the range to "All" and export the data. Download it by hand: SteamDB does
   not allow scraping or automated downloads.
2. Save the file as data/raw/steamdb/steamdb_chart_<appid>.csv (the default name).
   data/raw/ is git-ignored: raw exports are never committed.
3. Add the game to GAMES below with its official release date from its Steam
   store page, then run:
       python build_launch_data.py

Day 0 is the official release date. The output keeps the first DAYS days for
each game: one row per day with that day's peak number of concurrent players.
Games whose data doesn't cover those days are skipped, with the reason printed.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw" / "steamdb"
OUT_PATH = ROOT / "processed" / "launch_daily.parquet"
DAYS = 180
MAX_MISSING_DAYS = 10    # more blank days than this in the first DAYS days: skip the game

# AppID -> (display name, official release date on the Steam store).
# Order here is the order games appear in the app; the first one is the default.
GAMES = {
    1245620: ("Elden Ring", "2022-02-24"),
    1091500: ("Cyberpunk 2077", "2020-12-09"),
    2358720: ("Black Myth: Wukong", "2024-08-19"),
    377160: ("Fallout 4", "2015-11-09"),
    72850: ("The Elder Scrolls V: Skyrim", "2011-11-10"),
    1086940: ("Baldur's Gate 3", "2023-08-03"),
    1716740: ("Starfield", "2023-09-05"),
    2623190: ("The Elder Scrolls IV: Oblivion Remastered", "2025-04-22"),
    2767030: ("Marvel Rivals", "2024-12-05"),
    582160: ("Assassin's Creed Origins", "2017-10-26"),
    22380: ("Fallout: New Vegas", "2010-10-19"),
    252950: ("Rocket League", "2015-07-06"),
    367520: ("Hollow Knight", "2017-02-24"),
    1151340: ("Fallout 76", "2020-04-14"),
    1172470: ("Apex Legends", "2020-11-04"),
    1222140: ("Detroit: Become Human", "2020-06-18"),
    427520: ("Factorio", "2020-08-14"),
    220: ("Half-Life 2", "2004-11-16"),
    22330: ("The Elder Scrolls IV: Oblivion (GOTY 2009)", "2009-06-16"),
    22370: ("Fallout 3: Game of the Year Edition", "2009-12-17"),
    550: ("Left 4 Dead 2", "2009-11-16"),
    227300: ("Euro Truck Simulator 2", "2012-10-12"),
    244210: ("Assetto Corsa", "2014-12-19"),
    4356430: ("NBA 2K27", "2026-09-03"),
}


def read_steamdb_export(path: Path) -> pd.Series:
    """Daily peak players from a SteamDB chart export, indexed by UTC date.

    Exports are daily for older history and every 10 minutes for the latest
    year; both are reduced to one value per day (the day's highest count).
    Blank values are dropped, so days without data are simply absent.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "DateTime" not in df or "Players" not in df:
        raise ValueError(f"{path.name}: expected DateTime and Players columns, got {list(df.columns)}")
    when = pd.to_datetime(df["DateTime"], errors="coerce")
    players = pd.to_numeric(df["Players"].astype(str).str.replace(",", ""), errors="coerce")
    s = pd.Series(players.to_numpy(), index=when).dropna()
    s = s[s.index.notna()]
    return s.groupby(s.index.normalize()).max().sort_index()


def launch_day(daily: pd.Series, release: str | pd.Timestamp, share: float = 0.05) -> pd.Timestamp:
    """Day 0: the official release date, or the day after it.

    SteamDB days are UTC and store dates are local, so a game that unlocked
    after midnight UTC has nothing (or a handful of early-unlock players) on its
    official date. If the official date has no value, or less than `share` of
    the busiest day in the following week, day 0 moves to the next day.
    """
    release = pd.Timestamp(release).normalize()
    on_day = daily.get(release)
    next_week = daily[(daily.index > release) & (daily.index <= release + pd.Timedelta(days=7))]
    if on_day is None or (len(next_week) and on_day < share * next_week.max()):
        return release + pd.Timedelta(days=1)
    return release


def launch_window(daily: pd.Series, release: str | pd.Timestamp, days: int = DAYS) -> pd.DataFrame:
    """The first `days` days from day 0, one row per calendar day.

    Days with no export value stay missing (NaN); nothing is filled in.
    """
    start = launch_day(daily, release)
    dates = pd.date_range(start, periods=days, freq="D")
    return pd.DataFrame({
        "day": range(days),
        "date": dates,
        "players": daily.reindex(dates).to_numpy(),
    })


def check_window(daily: pd.Series, window: pd.DataFrame, release: str) -> str | None:
    """Why this game can't be used, or None if it's fine."""
    start = window["date"].iloc[0]
    if daily.index.min() > pd.Timestamp(release) + pd.Timedelta(days=1):
        return f"data starts {daily.index.min():%Y-%m-%d}, after the release date"
    missing = int(window["players"].isna().sum())
    if missing > MAX_MISSING_DAYS:
        last = daily.index.max()
        if last < start + pd.Timedelta(days=DAYS - 1):
            return f"only {(last - start).days + 1} of {DAYS} days have happened yet"
        return f"{missing} of the first {DAYS} days have no data"
    return None


def build() -> pd.DataFrame:
    frames, skipped = [], []
    for appid, (name, release) in GAMES.items():
        path = RAW_DIR / f"steamdb_chart_{appid}.csv"
        if not path.is_file():
            skipped.append((name, f"missing {path.relative_to(ROOT)}"))
            continue
        daily = read_steamdb_export(path)
        window = launch_window(daily, release)
        problem = check_window(daily, window, release)
        if problem:
            skipped.append((name, problem))
            continue
        shift = " (+1 day, UTC)" if window["date"].iloc[0] != pd.Timestamp(release) else ""
        print(f"  {name:44s} day 0 {window['date'].iloc[0]:%Y-%m-%d}{shift:15s} "
              f"peak {window['players'].max():>11,.0f}  missing {int(window['players'].isna().sum())}")
        frames.append(window.assign(appid=appid, game=name))
    if skipped:
        print("\nSkipped:")
        for name, why in skipped:
            print(f"  {name:44s} {why}")
    if not frames:
        raise SystemExit("No usable games.")
    out = pd.concat(frames, ignore_index=True)[["appid", "game", "day", "date", "players"]]
    out["appid"] = out["appid"].astype("int32")
    out["day"] = out["day"].astype("int16")
    return out


if __name__ == "__main__":
    print("Included:")
    data = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {data['appid'].nunique()} games, {len(data):,} rows to "
          f"{OUT_PATH.relative_to(ROOT)} ({OUT_PATH.stat().st_size / 1024:.0f} KB)")
