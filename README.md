# Steam Player Observatory — Cloud Deployment Kit

This kit packages your existing optimized V2 Streamlit dashboard for Streamlit Community Cloud.
It is a separate deploy folder: **do not upload the entire Steam Project directory**.

## 1. Folder placement

Unzip the kit **inside** your existing Steam Project, keeping this structure:

```text
Steam Project/
  processed/
    steam_games.csv
    steam_monthly.csv
    steam_daily.csv
  steam_cloud_deployment_kit/
    app.py
    prepare_for_cloud.py
    requirements.txt
    .gitignore
    .streamlit/config.toml
    processed/      # initially empty; .parquet files are generated here
```

## 2. Convert data locally (no rebuilding large source ZIPs)

Run these commands in PowerShell:

```powershell
cd "C:\Users\jacky\Desktop\MSDS\DATA511\Steam Project\steam_cloud_deployment_kit"
python -m pip install -r requirements.txt
python prepare_for_cloud.py
```

The script reads your verified CSV files and creates compressed:

```text
processed/steam_games.parquet
processed/steam_monthly.parquet
processed/steam_daily.parquet
```

It preserves rows and CCU numbers, tests the row counts of the output, and never changes the originals.
The output is typically smaller than the original CSVs, but **check the actual displayed size**.

## 3. Check locally (use same cloud code)

```powershell
python -m streamlit run app.py
```

Verify Monthly Trends, Top 10 and Daily Deep Dive (Dota 2) before uploading.

## 4. Create GitHub repo

Make a new repo (suggested name: `steam-player-observatory`).
Use GitHub Desktop to add this **deployment-kit folder**, commit, and publish to GitHub.
Your GitHub repo root should contain:

```text
app.py
requirements.txt
prepare_for_cloud.py
README.md
.gitignore
.streamlit/config.toml
processed/steam_games.parquet
processed/steam_monthly.parquet
processed/steam_daily.parquet
```

DO NOT include the source `Data/` folder, nested Mendeley ZIPs, the old CSVs, or backups.

Note: GitHub browser upload limit is **25 MiB per file**. For any larger Parquet files, use
GitHub Desktop or Git command line instead. Standard Git rejects a single file over 100 MiB.
If a generated Parquet is over 100 MiB, stop and use external data hosting instead of Git LFS blindly.

## 5. Deploy Streamlit

1. Open https://share.streamlit.io/ and sign in with GitHub.
2. **Create app** → **Yup, I have an app**.
3. Choose your GitHub repo, `main` branch and main file `app.py`.
4. Pick an available URL such as `steam-player-observatory`.
5. Deploy and open the resulting `*.streamlit.app` URL.

When you commit updated code/data to the repo, Streamlit redeploys the changes.
Cloud cold starts and shared hosting resource limits may occasionally increase loading time.

## Data attribution / interpretation

- Monthly history: [Kaggle Steam Monthly Average Players](https://www.kaggle.com/datasets/lunthu/steam-monthly-average-players), by lunthu; verify that dataset's redistribution terms before publishing derivative data in a public repo.
- Mendeley Data: Wannigamage, D., Barlow, M., Lakshika, E. & Kasmarik, K. (2020), *Steam Games Dataset: Player count history, Price history and data about games*, DOI [10.17632/ycy3sy3vj2.1](https://doi.org/10.17632/ycy3sy3vj2.1), **CC BY 4.0**. The daily/monthly data here are derived aggregates.
- CCU metrics are concurrent activity proxies, **not** unique users or cohort retention.
- Data has different coverage across games. Do not present all month windows as equal-length launch histories.
