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

Games currently included: Elden Ring, Cyberpunk 2077, Black Myth: Wukong, Fallout 4, The Elder Scrolls V: Skyrim.

> CCU counts people playing at the same moment. It is not the number of different players, and not retention.

## How the falloff stages are found

- **Launch day (day 0)** is the first day that starts three days in a row at 10% or more of the game's highest-ever daily peak. This skips small pre-release counts and one-day glitches. Days are in UTC, so day 0 can be the day after the store's release date.
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
3. Add the game to `GAMES` at the top of `build_launch_data.py`, then run:
   ```bash
   python build_launch_data.py
   ```
   This rewrites `processed/launch_daily.parquet`, which keeps only the first 180 days per game. The app picks it up automatically.

## Repository layout

```
app.py                         the Streamlit app (data prep, charts, page)
build_launch_data.py           SteamDB exports -> processed/launch_daily.parquet
processed/launch_daily.parquet the data the app uses (first 180 days per game)
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
- Not affiliated with Valve or SteamDB. Game names are trademarks of their owners.
