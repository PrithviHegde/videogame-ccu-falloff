"""Build data/game_info.csv: details about each game, for the app's filters.

Run after build_launch_data.py:
    python build_game_info.py

For every game in processed/launch_daily.parquet, this fetches its Steam store
page data (one request per game, a couple of seconds apart) and writes:

  From Steam (refreshed on every run):
    genres, publishers, developers   - "|"-separated lists
    steam_categories                 - Steam's feature tags (Single-player, Online Co-op, ...)
    mode_from_steam                  - a first guess at single/multiplayer from those tags
    free_now, price_now_usd          - today's list price (not used for filtering)

  Filled in and reviewed by the team (kept as they are when the script reruns):
    launch_price_usd                 - base-edition price on Steam at release; 0 = free
    main_mode                        - Single-player, Multiplayer or Both
    genres_override                  - optional: replaces Steam's genres when they're
                                       misleading (e.g. Marvel Rivals is tagged "Racing")
    notes                            - where the values came from
    Looks Good?, Reviewer Notes      - the team's review (same headers as the review
                                       Google Sheet, so its CSV download drops straight in)

The Steam store API is public and keyless; please keep requests slow.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

from build_launch_data import GAMES, OUT_PATH as LAUNCH_PATH

ROOT = Path(__file__).resolve().parent
INFO_PATH = ROOT / "data" / "game_info.csv"
STORE_URL = "https://store.steampowered.com/api/appdetails?appids={appid}&cc=us&l=en"
PAUSE_SECONDS = 2.0

REVIEW_COLUMNS = ["Looks Good?", "Reviewer Notes"]   # named as in the team's review sheet
HAND_COLUMNS = ["launch_price_usd", "main_mode", "notes", "genres_override", *REVIEW_COLUMNS]
MODES = ["Single-player", "Multiplayer", "Both"]
MULTI_TAGS = {"Multi-player", "Online PvP", "Online Co-op", "MMO", "PvP", "Co-op",
              "LAN PvP", "LAN Co-op", "Shared/Split Screen", "Cross-Platform Multiplayer"}


def fetch_store(appid: int) -> dict:
    """The Steam store's data for one game, or {} if Steam doesn't return it."""
    req = urllib.request.Request(STORE_URL.format(appid=appid),
                                 headers={"User-Agent": "videogame-ccu-falloff build script"})
    with urllib.request.urlopen(req, timeout=30) as response:
        body = json.load(response).get(str(appid), {})
    return body.get("data", {}) if body.get("success") else {}


def mode_from_categories(categories: list[str]) -> str:
    """First guess from Steam's tags. Steam tags many single-player games as
    multiplayer too (e.g. Elden Ring's online co-op), so main_mode is checked by hand."""
    single = "Single-player" in categories
    multi = bool(MULTI_TAGS & set(categories))
    if single and multi:
        return "Both"
    return "Multiplayer" if multi else "Single-player"


def store_row(appid: int, store: dict) -> dict:
    categories = [c["description"] for c in store.get("categories", [])]
    price = store.get("price_overview") or {}
    return {
        "genres": "|".join(g["description"] for g in store.get("genres", [])),
        "publishers": "|".join(store.get("publishers", [])),
        "developers": "|".join(store.get("developers", [])),
        "steam_categories": "|".join(categories),
        "mode_from_steam": mode_from_categories(categories),
        "free_now": bool(store.get("is_free", False)),
        # "initial" is the list price before any current discount.
        "price_now_usd": price["initial"] / 100 if "initial" in price else (0.0 if store.get("is_free") else None),
    }


def merge_hand_columns(fresh: pd.DataFrame, previous: pd.DataFrame | None) -> pd.DataFrame:
    """Keep the team's hand-filled columns from the existing file."""
    if previous is None:
        for col in HAND_COLUMNS:
            fresh[col] = None
        return fresh
    keep = previous.set_index("appid")[[c for c in HAND_COLUMNS if c in previous]]
    out = fresh.join(keep, on="appid")
    for col in HAND_COLUMNS:
        if col not in out:
            out[col] = None
    return out


def build() -> pd.DataFrame:
    appids = pd.read_parquet(LAUNCH_PATH, columns=["appid"])["appid"].unique().tolist()
    rows = []
    for i, appid in enumerate(appids):
        name, release = GAMES[appid]
        if i:
            time.sleep(PAUSE_SECONDS)
        store = fetch_store(appid)
        if not store:
            print(f"  ! {name}: Steam returned no data; Steam columns left empty")
        rows.append({"appid": appid, "game": name, "release_date": release, **store_row(appid, store)})
        print(f"  {name:44s} {rows[-1]['genres']}")
    previous = pd.read_csv(INFO_PATH) if INFO_PATH.is_file() else None
    return merge_hand_columns(pd.DataFrame(rows), previous)


if __name__ == "__main__":
    info = build()
    INFO_PATH.parent.mkdir(parents=True, exist_ok=True)
    info.to_csv(INFO_PATH, index=False)
    todo = info[info["launch_price_usd"].isna() | info["main_mode"].isna()]
    print(f"\nWrote {len(info)} games to {INFO_PATH.relative_to(ROOT)}")
    if len(todo):
        print(f"{len(todo)} games still need launch_price_usd / main_mode filled in by hand.")
