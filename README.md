# Video Game Player Falloff

**How fast do players leave a big game after launch?**

### ▶ Live app: https://videogame-ccu-falloff.streamlit.app/

An interactive look at the first 180 days after release for well-known Steam games: how many people were playing at once each day, how quickly that number fell, and where it settled. DATA 511 (Data Visualization) final project, University of Washington.

## What it shows

- **Players over time.** Each day's peak number of people playing the game at the same time (peak concurrent players, or CCU), from launch day to day 180, with a 7-day average to smooth out weekend bumps.
- **Three stages for each game.**
  1. **Launch:** the most players at once.
  2. **Falloff:** how much of that the game lost before the decline levels off, e.g. "lost 81% in 56 days".
  3. **Settled:** where it ends up (average of the last 30 days).
- **Comparisons.** Pick several games and view them on one chart (as % of each game's own peak, so big and small games share a scale) or side by side. A strip under the chart lines up each game's stages.
- **Controls.** Zoom from 14 to 180 days, switch between % of peak and player counts, and between days since release and calendar dates.
- **Filters.** Narrow the game list by genre, publisher, single-player or multiplayer, release year and launch price.

Games currently included: see `GAMES` in `build_launch_data.py` (19 games at the moment, from Skyrim in 2011 to Oblivion Remastered in 2025).

> CCU counts people playing at the same moment. It is not the number of different players, and not retention.

## How the falloff stages are found

- **Day 0** is the game's official release date on its Steam store page. SteamDB counts days in UTC, so when nothing (or almost nothing) was recorded on that date because the game unlocked after midnight UTC, day 0 is the next day.
- Games are skipped if their SteamDB history starts after the release date, or more than 10 of their first 180 days have no data. `build_launch_data.py` prints which games were skipped and why.
- **The leveling-off point** is where the decline flattens out: on the 7-day average, the day that sits furthest below a straight line from the peak to day 180 (the "knee" of the curve). Changing the smoothing from 5 to 14 days moves it by a few days at most.
- Stages are always computed on all 180 days, so they don't move when you zoom.

## Run it locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt pytest
streamlit run app.py      # opens http://localhost:8501
pytest                    # runs the tests (a few seconds)
```

## Adding a game

1. On SteamDB, open the game's charts page (`https://steamdb.info/app/<appid>/charts/`), set the range to **All**, and export the data. **Download it by hand:** SteamDB does not allow scraping or automated downloads.
2. Save it as `data/raw/steamdb/steamdb_chart_<appid>.csv` (the default file name). `data/raw/` is ignored by git, so raw downloads are never committed.
3. Add the game to `GAMES` at the top of `build_launch_data.py`, with its official release date from its Steam store page, then run:
   ```bash
   python build_launch_data.py
   ```
   This rewrites `processed/launch_daily.parquet`, which keeps only the first 180 days per game. The app picks it up automatically.
4. Fetch the game's details from the Steam store for the filters:
   ```bash
   python build_game_info.py
   ```
   This updates `data/game_info.csv`. Then fill in the new row's hand-checked columns (any spreadsheet app works):
   - `launch_price_usd`: the base-edition price on Steam at release, `0` for free to play
   - `main_mode`: `Single-player`, `Multiplayer` or `Both` (Steam tags many single-player games as multiplayer, so this is checked by hand)
   - `genres_override` (optional): replaces Steam's genres when they're misleading
   - `notes`: where the values came from

   Re-running the script refreshes the Steam columns but never overwrites these.

## Reviewing game details

Launch prices and single/multiplayer in `data/game_info.csv` are checked by hand. The team reviews them in a shared Google Sheet with the same columns, plus **Looks Good?** and **Reviewer Notes**:

1. In the sheet, check each value against a source (Wikipedia, launch-week press, SteamDB price history), fix it if needed, put the source in `notes`, and mark **Looks Good?** as `Yes`.
2. Only edit `launch_price_usd`, `main_mode`, `genres_override`, `notes`, **Looks Good?** and **Reviewer Notes**. Don't rename, reorder or delete columns.
3. One person downloads the sheet as CSV (File → Download → CSV), saves it over `data/game_info.csv`, runs `python build_game_info.py` and `pytest`, and opens a pull request.

Until every game is marked **Looks Good?**, the app's filter panel shows a warning that these values are drafts. `tests/test_game_info.py` fails the pull request if the CSV has a value the app can't read, such as a price written as `$59.99`.

## Repository layout

```
app.py                         the Streamlit app (data prep, charts, page)
build_launch_data.py           SteamDB exports -> processed/launch_daily.parquet
build_game_info.py             Steam store details -> data/game_info.csv
processed/launch_daily.parquet player counts the app uses (first 180 days per game)
data/game_info.csv             genres, publishers, launch price, single/multiplayer (team-reviewed)
tests/                         smoke tests (app loads) and unit tests (calculations)
.streamlit/config.toml         theme
```

`processed/steam_*.parquet` and `prepare_for_cloud.py` belong to the earlier monthly version of the app and are no longer used.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md): work on a branch, open a pull request, get one teammate's review, then merge. Every merge to `main` redeploys the live app.

## Team

Nate Aune, Cade Teets, Haoye Zhang, Prithvi Hegde, Meg Kaki.

## Data and credits

- Player counts: [SteamDB](https://steamdb.info), daily peak concurrent players, downloaded manually per game. Only the first 180 days after launch are kept here.
- Genres, publishers and developers: the Steam store. Launch prices and main play mode: checked by the team.
- Not affiliated with Valve or SteamDB. Game names are trademarks of their owners.
