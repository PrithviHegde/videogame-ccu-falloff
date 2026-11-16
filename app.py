"""DATA 511 | Steam Player Observatory

Warm-light dashboard, optimized V2: cached time-window stats, vectorized ranking,
and one-time lazy daily loading. Chart percentage toggle in both views.

Run from the Steam Project folder:
    python -m pip install -r requirements.txt
    python -m streamlit run steam_history_dashboard.py

Expected files:
    processed/steam_games.csv
    processed/steam_monthly.csv
    processed/steam_daily.csv  (only needed for Daily Deep Dive)

Monthly data: Kaggle and Mendeley; daily data: Mendeley (2017-2020).
All charts use actual records. CCU ratios are activity proxies, NOT user retention.
"""
from __future__ import annotations

from html import escape
from pathlib import Path
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="Steam Player Observatory | DATA 511",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "processed"


def preferred_file(stem: str) -> Path:
    """Prefer small Parquet deployment data; allow local CSV during development."""
    parquet = DATA_DIR / f"{stem}.parquet"
    return parquet if parquet.is_file() else DATA_DIR / f"{stem}.csv"


def table_columns(path: str | Path) -> set[str]:
    """Read just a table schema; never load the entire Parquet file for headers."""
    if str(path).endswith(".parquet"):
        import pyarrow.parquet as pq
        return set(pq.read_schema(str(path)).names)
    return set(pd.read_csv(path, nrows=0).columns)


def read_table(path: str | Path, usecols: list[str] | None = None, **csv_kwargs) -> pd.DataFrame:
    if str(path).endswith(".parquet"):
        return pd.read_parquet(path, columns=usecols, engine="pyarrow")
    return pd.read_csv(path, usecols=usecols, **csv_kwargs)


GAMES_PATH = preferred_file("steam_games")
MONTHLY_PATH = preferred_file("steam_monthly")
DAILY_PATH = preferred_file("steam_daily")

INK = "#253249"
MUTED = "#7B8698"
GRID = "#E8ECF2"
BLUE = "#5578B7"
SALMON = "#C88077"
GREEN = "#539885"
PURPLE = "#8876AF"
COLORS = [
    "#5578B7", "#C88077", "#539885", "#8876AF", "#D0A45D",
    "#4C8B9D", "#AB789E", "#7A8D5F", "#AE8970", "#64768F",
]
COLORS_BY_ID = {}

st.markdown(
    """
    <style>
    :root{--paper:#F9FAFB;--ink:#253249;--muted:#7B8698;--line:#E6EAF0}
    [data-testid="stApp"], [data-testid="stAppViewContainer"]{
      background:var(--paper);color:var(--ink)
    }
    [data-testid="stHeader"]{background:transparent}
    .block-container{max-width:1530px;padding-top:1.5rem;padding-bottom:3rem}
    section[data-testid="stSidebar"]{background:#F2F4F8;border-right:1px solid #E3E7EE}
    [data-testid="stSidebar"] .block-container{padding-top:1.1rem}
    h1,h2,h3{color:var(--ink)!important;letter-spacing:-.035em}
    h1{font-size:2rem!important}
    h3{font-size:1.08rem!important}
    div[data-testid="stVerticalBlockBorderWrapper"]{
      background:#FFFFFF;border:1px solid #E7EAF0!important;
      border-radius:16px!important;box-shadow:0 3px 18px rgba(32,48,69,.025)
    }
    [data-testid="stMetric"]{background:#FFF;border:1px solid #E7EAF0;
      border-radius:13px;padding:15px 18px}
    [data-testid="stMetricLabel"] p{color:#708096;font-size:.82rem}
    [data-testid="stMetricValue"]{color:#27364E;letter-spacing:-.035em}
    [data-testid="stCaptionContainer"] p{color:#8793A3;font-size:.77rem}
    .op-brand{display:flex;gap:13px;align-items:center;margin-bottom:13px}
    .op-icon{background:#E6ECF7;border:1px solid #DCE4F3;border-radius:13px;
      width:46px;height:46px;display:flex;align-items:center;justify-content:center;
      font-size:22px}
    .op-name{font-family:Georgia,serif;font-size:1.65rem;color:#24344B;
      letter-spacing:-.035em;font-weight:650;line-height:1.13}
    .op-sub{font-size:.82rem;color:#79869B;padding-top:3px}
    .op-kicker{font-size:.72rem;font-weight:750;letter-spacing:.12em;
      text-transform:uppercase;color:#798BB0;margin-bottom:6px}
    .op-section{font-size:1.12rem;font-weight:700;color:#24334B;margin:3px 0}
    .op-note{font-size:.83rem;color:#7A879A;line-height:1.55}
    .op-rule{border-top:1px solid #E9ECF1;margin:1.1rem 0}
    </style>
    """,
    unsafe_allow_html=True,
)


def nice_int(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{int(round(float(x))):,}"


def nice_pct(x: float | int | None, signed: bool = False) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{float(x):+.1f}%" if signed else f"{float(x):.1f}%"


def clean_months(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a monthly dataset without inventing or filling observations."""
    df = df.copy()
    df["appid"] = pd.to_numeric(df["appid"], errors="coerce")
    df["month_date"] = pd.to_datetime(
        df["year_month"].astype(str) + "-01", errors="coerce"
    )
    for col in ("avg_ccu", "peak_ccu"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["appid", "month_date", "avg_ccu"])
    df = df.loc[(df["appid"] > 0) & (df["avg_ccu"] >= 0)].copy()
    df["appid"] = df["appid"].astype("int64")
    df["peak_ccu"] = df["peak_ccu"].where(df["peak_ccu"] >= 0, np.nan)
    df = df.sort_values(["appid", "month_date"])
    # The merged sources have been verified as unique, but guard defensively.
    df = df.drop_duplicates(["appid", "month_date"], keep="last")
    df["month_ordinal"] = df["month_date"].dt.year * 12 + df["month_date"].dt.month
    return df.reset_index(drop=True)


@st.cache_resource(show_spinner="Loading monthly history (once)…", max_entries=2)
def load_monthly(games_file: str, monthly_file: str, games_mtime: int, monthly_mtime: int):
    games = read_table(games_file, low_memory=False)
    required_games = {"appid", "game_name", "genres", "publishers", "release_date"}
    if missing := required_games - set(games.columns):
        raise ValueError(f"steam_games.csv is missing {sorted(missing)}")
    games["appid"] = pd.to_numeric(games["appid"], errors="coerce")
    games = games.dropna(subset=["appid"]).copy()
    games["appid"] = games["appid"].astype("int64")
    games = games.drop_duplicates("appid")
    for col in ("genres", "publishers", "mendeley_name"):
        if col not in games.columns:
            games[col] = ""
        games[col] = games[col].fillna("").astype(str)
    games["game_name"] = games["game_name"].fillna("Unknown").astype(str)
    games["release_dt"] = pd.to_datetime(games["release_date"], errors="coerce")
    games["release_ord"] = (
        games["release_dt"].dt.year * 12 + games["release_dt"].dt.month
    )
    games["display"] = games["game_name"] + " (" + games["appid"].astype(str) + ")"
    games = games.set_index("appid", drop=False)

    needed = {"appid", "year_month", "avg_ccu", "peak_ccu"}
    header = table_columns(monthly_file)
    if missing := needed - set(header):
        raise ValueError(f"steam_monthly.csv is missing {sorted(missing)}")
    use = sorted(needed | ({"source"} if "source" in header else set()))
    monthly = clean_months(read_table(monthly_file, usecols=use, low_memory=False))
    if "source" not in monthly:
        monthly["source"] = "Unknown"
    monthly["source"] = monthly["source"].fillna("Unknown")
    monthly = monthly.loc[monthly["appid"].isin(games.index)]

    first_month = monthly.groupby("appid")["month_ordinal"].min()
    games["first_month_ordinal"] = first_month
    games["first_release_offset"] = games["first_month_ordinal"] - games["release_ord"]
    # Release-aligned means first observation is in release month or the next.
    # Excludes old games first observed years after launch, or ambiguous EA history.
    games["launch_comparable"] = games["first_release_offset"].between(0, 1)
    monthly["release_month_index"] = monthly["month_ordinal"] - monthly["appid"].map(games["release_ord"])
    monthly["launch_comparable"] = monthly["appid"].map(games["launch_comparable"]).fillna(False)
    return games, monthly


def genres_available(games: pd.DataFrame) -> list[str]:
    out = set()
    for string in games["genres"].dropna():
        out.update(part.strip() for part in string.split("|") if part.strip())
    return sorted(out, key=str.casefold)


def filter_games(games: pd.DataFrame, genre: str, publisher: str) -> pd.DataFrame:
    filtered = games
    if genre != "All genres":
        # Split fields, don't substring-match e.g. "RPG" and "JRPG".
        filtered = filtered.loc[
            filtered["genres"].map(
                lambda value: genre in [x.strip() for x in value.split("|")]
            )
        ]
    if publisher.strip():
        filtered = filtered.loc[
            filtered["publishers"].str.contains(publisher.strip(), case=False, regex=False)
        ]
    return filtered


def select_analysis_window(
    monthly: pd.DataFrame, games: pd.DataFrame,
    alignment: str, start_month: str, end_month: str, release_window: int,
) -> pd.DataFrame:
    """Fast date filtering; keep source frames immutable for cache_resource."""
    if alignment == "Calendar months":
        subset = monthly.loc[monthly["year_month"].between(start_month, end_month)]
        return subset.assign(x_position=subset["month_ordinal"])

    subset = monthly.loc[
        monthly["launch_comparable"]
        & monthly["release_month_index"].between(0, release_window - 1)
    ]
    return subset.assign(x_position=subset["release_month_index"].astype("int32"))

def monthly_stats(view: pd.DataFrame, alignment: str, release_window: int) -> pd.DataFrame:
    """Vectorized per-game statistics; avoids a Python loop across 7,000+ games."""
    columns = [
        "appid", "months", "first_month", "last_month", "peak_ccu", "mean_ccu",
        "peak_monthly_average", "end_to_peak_pct", "largest_monthly_decline_pct",
        "last_month_index",
    ]
    if view.empty:
        return pd.DataFrame(columns=columns)

    # Source monthly data is already sorted by (appid, month_date).
    # Running statistics on all eligible games once is dramatically faster than
    # groupby(appid) + Python for-loop on every Streamlit widget change.
    gb = view.groupby("appid", sort=False, observed=True)
    previous = gb["avg_ccu"].shift()
    previous_month = gb["month_ordinal"].shift()
    consecutive = view["month_ordinal"].sub(previous_month).eq(1) & previous.gt(0)
    drop_pct = (view["avg_ccu"].div(previous).sub(1) * 100).where(consecutive)

    stats = gb.agg(
        months=("avg_ccu", "size"),
        first_month=("year_month", "first"),
        last_month=("year_month", "last"),
        peak_ccu=("peak_ccu", "max"),
        mean_ccu=("avg_ccu", "mean"),
        peak_monthly_average=("avg_ccu", "max"),
        last_avg_ccu=("avg_ccu", "last"),
        last_month_index=("x_position", "max"),
    )
    stats["largest_monthly_decline_pct"] = (
        drop_pct.groupby(view["appid"], sort=False).min().reindex(stats.index)
    )
    eligible_ratio = stats["peak_monthly_average"].gt(0)
    if alignment != "Calendar months":
        eligible_ratio &= stats["last_month_index"].eq(release_window - 1)
    stats["end_to_peak_pct"] = (
        100.0 * stats["last_avg_ccu"] / stats["peak_monthly_average"].replace(0, np.nan)
    ).where(eligible_ratio)
    return stats.reset_index()[columns]


@st.cache_resource(show_spinner="Calculating this time window…", max_entries=2)
def prepared_monthly_window(
    alignment: str, start_month: str, end_month: str,
    release_window: int, games_mtime: int, monthly_mtime: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cache results by dates only, never by chart type, color, or game selection.

    Both returned DataFrames are read-only. Widgets filter them without mutation.
    """
    games, monthly = load_monthly(
        str(GAMES_PATH), str(MONTHLY_PATH), games_mtime, monthly_mtime
    )
    view = select_analysis_window(
        monthly, games, alignment, start_month, end_month, release_window
    )
    stats = monthly_stats(view, alignment, release_window)
    return view, stats

def rank_stats(stats: pd.DataFrame, metric: str, minimum_months: int) -> pd.DataFrame:
    stats = stats.loc[stats["months"] >= minimum_months].copy()
    field_map = {
        "Observed peak CCU": ("peak_ccu", False),
        "Average CCU": ("mean_ccu", False),
        "End / peak ratio": ("end_to_peak_pct", False),
        "Steepest monthly drop": ("largest_monthly_decline_pct", True),
    }
    key, ascending = field_map[metric]
    stats = stats.dropna(subset=[key])
    return stats.sort_values([key, "months", "appid"], ascending=[ascending, False, True])


def custom_fig() -> go.Figure:
    f = go.Figure()
    f.update_layout(
        template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family="Segoe UI, Inter, sans-serif", color=INK, size=12),
        margin=dict(l=12, r=12, t=12, b=12),
        hovermode="closest", legend=dict(orientation="h", y=-.19, x=0, title=""),
    )
    f.update_xaxes(showline=True, linewidth=1, linecolor=GRID, gridcolor=GRID,
                   zeroline=False, tickfont=dict(color=MUTED))
    f.update_yaxes(showline=False, gridcolor=GRID, zeroline=False,
                   tickfont=dict(color=MUTED))
    return f


def complete_selected_grid(
    view: pd.DataFrame, ids: list[int], alignment: str,
    start_month: str, end_month: str, release_window: int,
) -> pd.DataFrame:
    """Insert NaNs at missing x positions, so plotted lines have visible gaps."""
    if alignment == "Calendar months":
        months = pd.period_range(start_month, end_month, freq="M")
        positions = [p.year * 12 + p.month for p in months]
    else:
        positions = list(range(release_window))
    grid = pd.MultiIndex.from_product([ids, positions], names=["appid", "x_position"])
    merged = view.loc[view["appid"].isin(ids)].drop_duplicates(
        ["appid", "x_position"]
    ).set_index(["appid", "x_position"])
    return merged.reindex(grid).reset_index()


def metric_series(grid: pd.DataFrame, chosen_metric: str) -> pd.Series:
    """Normalize only within the chosen analysis window, never fill zeros."""
    if chosen_metric == "Monthly average CCU":
        return grid["avg_ccu"]
    if chosen_metric == "Monthly peak CCU":
        return grid["peak_ccu"]
    max_ccu = grid.groupby("appid")["avg_ccu"].transform("max").replace(0, np.nan)
    return grid["avg_ccu"].div(max_ccu).mul(100)


def genre_benchmark(
    view: pd.DataFrame, stats: pd.DataFrame, metric: str, minimum_months: int,
) -> pd.DataFrame:
    """Percent-of-observed-peak benchmark by release-aligned month, grouped across games."""
    allowed = stats.loc[stats["months"] >= minimum_months, "appid"]
    d = view.loc[view["appid"].isin(allowed), ["appid", "x_position", "avg_ccu"]].copy()
    peaks = d.groupby("appid")["avg_ccu"].transform("max").replace(0, np.nan)
    d["pct"] = 100 * d["avg_ccu"] / peaks
    d = d.dropna(subset=["pct"])
    return d.groupby("x_position").agg(
        q25=("pct", lambda x: x.quantile(.25)),
        med=("pct", "median"),
        q75=("pct", lambda x: x.quantile(.75)),
        n=("appid", "nunique"),
    ).reset_index()


def line_plot(
    grid: pd.DataFrame, games: pd.DataFrame, ids: list[int],
    alignment: str, measure: str, focus: int | None = None,
    benchmark: pd.DataFrame | None = None,
) -> go.Figure:
    fig = custom_fig()
    grid = grid.copy()
    grid["metric_value"] = metric_series(grid, measure)
    # Draw benchmark underneath game lines.
    if benchmark is not None and len(benchmark) and (benchmark["n"] >= 3).any():
        bench = benchmark.copy()
        bench.loc[bench["n"] < 3, ["q25", "med", "q75"]] = np.nan
        fig.add_trace(go.Scatter(
            x=bench["x_position"], y=bench["q75"],
            mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=bench["x_position"], y=bench["q25"], mode="lines",
            line=dict(width=0), fill="tonexty", fillcolor="rgba(113,132,166,.13)",
            hoverinfo="skip", name="Genre IQR", showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=bench["x_position"], y=bench["med"], mode="lines",
            line=dict(color="#9CA7B8", width=2, dash="dash"),
            name="Genre median", hovertemplate="Genre median %{y:.1f}%<extra></extra>",
        ))
    for idx, appid in enumerate(ids):
        game = grid.loc[grid["appid"] == appid].sort_values("x_position")
        if game.empty:
            continue
        name = str(games.loc[appid, "game_name"])
        color = COLORS[idx % len(COLORS)]
        faded = focus is not None and focus != appid
        if alignment == "Calendar months":
            x = [
                pd.Timestamp(year=int((v-1)//12), month=int((v-1)%12+1), day=1)
                for v in game["x_position"]
            ]
            custom = game["year_month"].fillna("").tolist()
            hover_x = "%{x|%b %Y}"
        else:
            x = game["x_position"].astype(int).to_numpy()
            custom = game["year_month"].fillna("").tolist()
            hover_x = "Month %{x} since release"
        suffix = "%" if measure == "Share of observed peak (%)" else " players"
        fig.add_trace(go.Scatter(
            x=x, y=game["metric_value"], mode="lines+markers" if len(ids) <= 3 else "lines",
            name=name, connectgaps=False,
            customdata=np.array(custom).reshape(-1, 1),
            line=dict(color="#C4CAD4" if faded else color, width=1.5 if faded else 3),
            marker=dict(size=5),
            opacity=.48 if faded else 1,
            hovertemplate=(
                "<b>" + escape(name) + "</b><br>" + hover_x + "<br>" +
                "%{y:,.1f}" + suffix + "<br>Source month: %{customdata[0]}<extra></extra>"
            ),
        ))
    fig.update_layout(height=470)
    fig.update_yaxes(
        title=measure, rangemode="tozero",
        ticksuffix="%" if measure == "Share of observed peak (%)" else "",
        range=[0, 105] if measure == "Share of observed peak (%)" else None,
    )
    fig.update_xaxes(title="Calendar month" if alignment == "Calendar months" else "Months since release", nticks=12)
    return fig


def comparison_bars(stats: pd.DataFrame, games: pd.DataFrame, ids: list[int], field: str, title: str, suffix: str) -> go.Figure:
    f = custom_fig()
    g = stats.loc[stats["appid"].isin(ids)].dropna(subset=[field]).copy()
    if g.empty:
        f.add_annotation(x=.5, y=.5, xref="paper", yref="paper", text="Not enough comparable data", showarrow=False, font=dict(color=MUTED))
        f.update_layout(height=270, title=None)
        return f
    order = {appid: idx for idx, appid in enumerate(ids)}
    g["order"] = g["appid"].map(order)
    g = g.sort_values("order", ascending=False)
    labels = [str(games.loc[appid, "game_name"]) for appid in g["appid"]]
    # Use one bar color per selected game; negative bars use warm salmon.
    colors = [SALMON if field == "largest_monthly_decline_pct" else COLORS[order[appid] % len(COLORS)] for appid in g["appid"]]
    f.add_trace(go.Bar(
        x=g[field], y=labels, orientation="h", marker_color=colors,
        text=[f"{v:.1f}%" for v in g[field]], textposition="auto",
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        cliponaxis=False,
    ))
    f.update_layout(height=280, showlegend=False, bargap=.44)
    f.update_xaxes(title=suffix, zeroline=True, zerolinecolor=GRID)
    f.update_yaxes(title="", automargin=True)
    return f


def safe_sources(view: pd.DataFrame, ids: list[int]) -> str:
    sources = sorted(set(view.loc[view["appid"].isin(ids), "source"].dropna().astype(str)))
    return ", ".join(sources) if sources else "Unknown"


@st.cache_resource(show_spinner="Loading daily history once (subsequent selections are fast)…", max_entries=2)
def load_daily_all(path: str, mtime: int) -> pd.DataFrame:
    """Load daily data lazily, once per file version; index by AppID for fast lookup.

    The 1.8M-row file is never reread when game/date/units change. Numeric dtypes
    and parsed timestamps reduce persistent memory usage substantially.
    """
    needed = {
        "appid", "day_utc", "daily_mean_ccu", "daily_peak_ccu", "sample_coverage_ratio",
    }
    header = table_columns(path)
    if missing := needed - header:
        raise ValueError(f"steam_daily.csv is missing {sorted(missing)}")
    optional = {"observation_count"} if "observation_count" in header else set()
    use = sorted(needed | optional)
    dtype = {
        "appid": "int32",
        "daily_mean_ccu": "float32",
        "daily_peak_ccu": "float32",
        "sample_coverage_ratio": "float32",
    }
    if "observation_count" in use:
        dtype["observation_count"] = "int32"
    all_rows = read_table(
        path, usecols=use, dtype=dtype, parse_dates=["day_utc"],
        low_memory=False,
    )
    # Parquet dates can be strings; normalize before any date-range operations.
    all_rows["day_utc"] = pd.to_datetime(all_rows["day_utc"], errors="coerce")
    all_rows = all_rows.dropna(subset=["appid", "day_utc"])
    all_rows = all_rows.sort_values(["appid", "day_utc"])
    all_rows = all_rows.drop_duplicates(["appid", "day_utc"], keep="last")
    # pandas index speeds up game lookups without needing an extra database file.
    return all_rows.set_index("appid", drop=False)


def load_daily_selected(path: str, mtime: int, appid: int) -> pd.DataFrame:
    """Return only one game; never modify the globally cached daily table."""
    all_rows = load_daily_all(path, mtime)
    if appid not in all_rows.index:
        return all_rows.iloc[:0].copy()
    return all_rows.loc[[appid]].reset_index(drop=True).copy()

def daily_fig(daily: pd.DataFrame, measure: str, relative: bool = False) -> go.Figure:
    fig = custom_fig()
    value = "daily_mean_ccu" if measure == "Daily average CCU" else "daily_peak_ccu"
    d = daily.set_index("day_utc")[[value]].asfreq("D")
    if relative:
        highest = d[value].max()
        d[value] = d[value] / highest * 100 if highest > 0 else np.nan
    # Missing dates stay missing; rolling mean requires at least 5 observations.
    d["rolling"] = d[value].rolling(7, min_periods=5).mean()
    axis_suffix = "%" if relative else " players"
    fig.add_trace(go.Scatter(
        x=d.index, y=d[value], name="Daily observed activity", mode="lines",
        line=dict(color="#A7BBD8", width=1.6), connectgaps=False,
        hovertemplate="%{x|%b %d, %Y}<br>%{y:,.1f}" + axis_suffix + "<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=d.index, y=d["rolling"], name="7-day moving average", mode="lines",
        line=dict(color=BLUE, width=3), connectgaps=False,
        hovertemplate="7-day average %{x|%b %d, %Y}<br>%{y:,.1f}" + axis_suffix + "<extra></extra>",
    ))
    fig.update_layout(height=440)
    fig.update_xaxes(title="Observed date")
    fig.update_yaxes(
        title="% of observed peak" if relative else measure,
        rangemode="tozero", ticksuffix="%" if relative else "",
        range=[0, 105] if relative else None,
    )
    return fig

def sidebar_header():
    st.markdown('<div class="op-kicker">DATA 511 · INTERACTIVE VISUALIZATION</div>', unsafe_allow_html=True)
    st.markdown('<div class="op-section">Explore the data</div>', unsafe_allow_html=True)


# ------------------------- App starts here -------------------------
st.markdown(
    '<div class="op-brand"><div class="op-icon">◈</div><div>'
    '<div class="op-name">Steam Player Observatory</div>'
    '<div class="op-sub">Historical activity, falloff & cross-game comparison · DATA 511</div>'
    '</div></div>',
    unsafe_allow_html=True,
)

for file in (GAMES_PATH, MONTHLY_PATH):
    if not file.is_file():
        st.error(f"Required file not found: {file}")
        st.code("processed/steam_games.parquet\nprocessed/steam_monthly.parquet")
        st.stop()

try:
    games, monthly = load_monthly(
        str(GAMES_PATH), str(MONTHLY_PATH), GAMES_PATH.stat().st_mtime_ns,
        MONTHLY_PATH.stat().st_mtime_ns,
    )
except Exception as exc:
    st.error(f"Could not load monthly data: {exc}")
    st.stop()

with st.sidebar:
    sidebar_header()
    page = st.radio("Analysis view", ["Monthly Trends", "Daily Deep Dive"], horizontal=False)
    st.divider()

if page == "Monthly Trends":
    all_months = sorted(monthly["year_month"].unique())
    genre_options = genres_available(games)

    with st.sidebar:
        st.markdown("#### Time & comparison")
        alignment = st.radio("Time axis", ["Calendar months", "Months since release"])
        if alignment == "Calendar months":
            start_month, end_month = st.select_slider(
                "Calendar window", options=all_months,
                value=(all_months[0], all_months[-1]),
            )
            release_window = 12
            minimum_months = 3
        else:
            release_window = st.select_slider("Release window", options=[3, 6, 12, 24], value=12)
            minimum_months = max(2, math.ceil(release_window * .5))
            start_month, end_month = all_months[0], all_months[-1]
            st.caption("Requires a known release date and observations beginning in launch month or the next month.")

        st.markdown("#### Find & rank games")
        select_mode = st.selectbox(
            "Choose games", ["Top 5", "Top 10", "Genre leaders", "Manual search"]
        )
        default_genre = "Action" if select_mode == "Genre leaders" and "Action" in genre_options else "All genres"
        genre = st.selectbox(
            "Genre", ["All genres"] + genre_options,
            index=(["All genres"] + genre_options).index(default_genre),
        )
        publisher_query = st.text_input("Publisher contains (optional)", placeholder="e.g., Valve")
        rank_by = st.selectbox(
            "Rank games by", ["Observed peak CCU", "Average CCU", "End / peak ratio", "Steepest monthly drop"]
        )
        units = st.radio(
            "Line chart units", ["CCU (absolute)", "Relative to peak (%)"],
            horizontal=True, help="100% = this game's highest observed monthly average within the selected window.",
        )
        if units == "Relative to peak (%)":
            measure = "Share of observed peak (%)"
        else:
            measure = st.selectbox(
                "CCU series", ["Monthly average CCU", "Monthly peak CCU"]
            )

    filtered_games = filter_games(games, genre, publisher_query)
    cached_window, cached_stats = prepared_monthly_window(
        alignment, start_month, end_month, release_window,
        GAMES_PATH.stat().st_mtime_ns, MONTHLY_PATH.stat().st_mtime_ns,
    )
    # Cheap subsets only: never recalculate 7,000 game statistics on a style change.
    if genre == "All genres" and not publisher_query.strip():
        window_data, full_stats = cached_window, cached_stats
    else:
        filtered_ids = filtered_games.index
        window_data = cached_window.loc[cached_window["appid"].isin(filtered_ids)]
        full_stats = cached_stats.loc[cached_stats["appid"].isin(filtered_ids)]
    candidates = rank_stats(full_stats, rank_by, minimum_months)

    with st.sidebar:
        if select_mode == "Manual search":
            valid_ids = set(full_stats.loc[full_stats["months"] >= 2, "appid"].astype(int))
            opts = sorted(valid_ids & set(filtered_games.index),
                          key=lambda i: (str(games.loc[i, "game_name"]).casefold(), i))
            favorites = [578080, 271590, 730, 570, 252490]
            defaults = [x for x in favorites if x in opts][:5] or opts[:3]
            ids = st.multiselect(
                "Search and select up to 10 games", options=opts, default=defaults,
                format_func=lambda i: str(games.loc[i, "display"]), max_selections=10,
            )
        else:
            if select_mode == "Genre leaders" and genre == "All genres":
                st.warning("Choose a specific Genre above to see its leaders.")
                ids = []
            else:
                limit = 10 if select_mode == "Top 10" else 5
                ids = candidates.head(limit)["appid"].astype(int).tolist()
        st.caption(f"{len(filtered_games):,} games match metadata filters · {len(candidates):,} have comparable observations")
        if ids:
            focus_options = [None] + ids
            focus = st.selectbox("Emphasize a game", focus_options, format_func=lambda i: "All selected" if i is None else str(games.loc[i, "game_name"]))
        else:
            focus = None

    st.caption(
        f"{len(games):,} total AppIDs with monthly history · {len(filtered_games):,} match filters · "
        f"{len(candidates):,} have at least {minimum_months} observed months in this comparison"
    )
    if alignment != "Calendar months":
        st.info(
            "Release-aligned comparisons include only games whose observed history starts in their release month "
            "or the following month. Early Access / release-date discrepancies can exclude some familiar games."
        )
    if not ids:
        st.warning("No games match this selection. Try another genre, window, or ranking.")
        st.stop()

    selected_stats = full_stats.loc[full_stats["appid"].isin(ids)].copy()
    strongest = selected_stats.loc[selected_stats["peak_ccu"].idxmax()] if len(selected_stats) else None
    retention_valid = selected_stats["end_to_peak_pct"].dropna()
    steepest_valid = selected_stats["largest_monthly_decline_pct"].dropna()

    a, b, c, d = st.columns(4)
    a.metric("Highest observed monthly peak", nice_int(strongest["peak_ccu"]) if strongest is not None else "—")
    b.metric("Median end / peak ratio", nice_pct(retention_valid.median() if len(retention_valid) else None))
    c.metric("Steepest monthly drop", nice_pct(steepest_valid.min() if len(steepest_valid) else None, signed=True))
    d.metric("Games compared", str(len(ids)))

    st.write("")
    with st.container(border=True):
        st.markdown('<div class="op-section">Player trends</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="op-note">{escape(measure)} · '
            f'{escape("Calendar history" if alignment == "Calendar months" else "Release-aligned months")}</div>',
            unsafe_allow_html=True,
        )
        grid = complete_selected_grid(
            window_data, ids, alignment, start_month, end_month, release_window
        )
        benchmark = None
        if (alignment != "Calendar months" and measure == "Share of observed peak (%)"
                and genre != "All genres"):
            benchmark = genre_benchmark(window_data, full_stats, measure, minimum_months)
            st.caption(f"Genre benchmark: {genre} · median and 25th–75th percentile; at least 3 games per month")
        fig = line_plot(grid, games, ids, alignment, measure, focus, benchmark)
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
        st.caption(
            "Relative mode scales each game to its highest monthly average within this chosen window. "
            "CCU is concurrent activity, not unique-player retention. "
            "Missing months are shown as gaps; observations are not interpolated."
        )

    left, right = st.columns(2, gap="medium")
    with left:
        with st.container(border=True):
            st.markdown("#### End / peak ratio")
            st.caption("Last observed monthly average ÷ highest observed monthly average, within the selected window")
            st.plotly_chart(
                comparison_bars(selected_stats, games, ids, "end_to_peak_pct", "End / peak", "%"),
                width="stretch", config={"displaylogo": False},
            )
    with right:
        with st.container(border=True):
            st.markdown("#### Largest monthly decline")
            st.caption("Largest % drop between consecutive observed calendar months; not across missing months")
            st.plotly_chart(
                comparison_bars(selected_stats, games, ids, "largest_monthly_decline_pct", "Drop", "%"),
                width="stretch", config={"displaylogo": False},
            )

    with st.container(border=True):
        st.markdown("#### Selected game comparison")
        table = selected_stats.copy()
        table["Game"] = table["appid"].map(games["game_name"])
        table["Genres"] = table["appid"].map(games["genres"]).fillna("")
        table = table[["Game", "appid", "Genres", "months", "first_month", "last_month", "peak_ccu", "mean_ccu", "end_to_peak_pct", "largest_monthly_decline_pct"]]
        table = table.rename(columns={
            "appid": "AppID", "months": "Observed months", "first_month": "First month",
            "last_month": "Last month", "peak_ccu": "Peak CCU", "mean_ccu": "Avg CCU",
            "end_to_peak_pct": "End / peak (%)", "largest_monthly_decline_pct": "Largest drop (%)",
        })
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "Peak CCU": st.column_config.NumberColumn(format="%d"),
            "Avg CCU": st.column_config.NumberColumn(format="%.1f"),
            "End / peak (%)": st.column_config.NumberColumn(format="%.1f%%"),
            "Largest drop (%)": st.column_config.NumberColumn(format="%.1f%%"),
        })
        st.download_button(
            "Download selected monthly records", data=window_data.loc[
                window_data["appid"].isin(ids)
            ].to_csv(index=False).encode("utf-8-sig"),
            file_name="selected_steam_monthly.csv", mime="text/csv",
        )

    # Dedicated ranking table, recalculated under current filters/time alignment.
    with st.expander("Explore genre / game rankings", expanded=False):
        st.caption(
            "Rankings use only available observations inside the selected time window. "
            "Choose a Genre in the sidebar to compare its leading games."
        )
        leader = candidates.head(25).copy()
        if not leader.empty:
            leader.insert(0, "Rank", np.arange(1, len(leader) + 1))
            leader["Game"] = leader["appid"].map(games["game_name"])
            st.dataframe(
                leader[["Rank", "Game", "appid", "months", "peak_ccu", "mean_ccu", "end_to_peak_pct", "largest_monthly_decline_pct"]].rename(columns={
                    "appid": "AppID", "months": "Months", "peak_ccu": "Peak CCU",
                    "mean_ccu": "Avg CCU", "end_to_peak_pct": "End / Peak %",
                    "largest_monthly_decline_pct": "Largest Drop %",
                }),
                hide_index=True, width="stretch",
            )

    with st.expander("Methodology & known limitations", expanded=False):
        st.markdown(
            "- **Observed peak CCU** is the maximum monthly peak in the available window.\n"
            "- **End / peak** uses monthly *average* CCU in the denominator and numerator; "
            "it is not cohort retention or the percentage of individual players who returned.\n"
            "- **Largest monthly decline** compares only adjacent calendar months with a nonzero prior average.\n"
            "- **Calendar view** may compare games with different observation spans. "
            "See the First / Last Month columns before drawing conclusions.\n"
            "- **Release view** uses recorded release dates, which may represent full release rather than Early Access. "
            "It requires data beginning in release month or the next. An end/peak ratio is shown only "
            "when the last month of the selected release window is observed.\n"
            "- Genre / Publisher metadata exists only for Mendeley-matched games. "
            "Unknown metadata is retained in the unfiltered view.\n"
            "- Some titles have been renamed: AppID 730 was CS:GO during 2017–2020 "
            "and is now CS2; AppID 271590 is listed as GTA V Legacy.\n"
            "- Source: Kaggle SteamCharts monthly history (2012–2025), with Mendeley-only games "
            "added for 2017–2020. No future values or forecasts are generated."
        )

else:
    st.markdown("### Daily Deep Dive")
    st.caption("Mendeley high-frequency records aggregated to daily CCU · Dec 2017 – Aug 2020")
    if not DAILY_PATH.is_file():
        st.error(f"Daily file not found: {DAILY_PATH}")
        st.stop()

    # Updated incremental dataset has daily history for all metadata-matched game IDs.
    has_meta = games.get("has_mendeley_metadata", pd.Series(False, index=games.index))
    candidate_ids = games.index[has_meta.astype(str).str.lower().eq("true")].tolist()
    if not candidate_ids:
        candidate_ids = games.index.tolist()
    candidate_ids = sorted(candidate_ids, key=lambda i: str(games.loc[i, "game_name"]).casefold())
    with st.sidebar:
        st.markdown("#### Daily detail")
        default_id = 570 if 570 in candidate_ids else candidate_ids[0]
        chosen_id = st.selectbox(
            "Search for a game", options=candidate_ids,
            index=candidate_ids.index(default_id),
            format_func=lambda i: str(games.loc[i, "display"]),
        )
        day_measure = st.selectbox("Daily metric", ["Daily average CCU", "Daily observed peak CCU"])
        daily_units = st.radio("Daily chart units", ["CCU (absolute)", "Relative to peak (%)"], horizontal=True)
        coverage_only = st.checkbox("Only ≥70% covered days", value=True)

    try:
        daily = load_daily_selected(str(DAILY_PATH), DAILY_PATH.stat().st_mtime_ns, int(chosen_id))
    except Exception as exc:
        st.error(f"Could not read daily records: {exc}")
        st.stop()
    if daily.empty:
        st.warning("No daily records found for this game. Choose another game.")
        st.stop()

    min_date = daily["day_utc"].min().date()
    max_date = daily["day_utc"].max().date()
    suggested_start = max(min_date, (pd.Timestamp(max_date) - pd.Timedelta(days=120)).date())
    with st.sidebar:
        choice = st.date_input(
            "Observed date range", value=(suggested_start, max_date),
            min_value=min_date, max_value=max_date,
        )
    if not isinstance(choice, (tuple, list)) or len(choice) != 2:
        st.info("Select both a start and end date.")
        st.stop()
    date_a, date_b = choice
    daily = daily.loc[daily["day_utc"].dt.date.between(date_a, date_b)].copy()
    if coverage_only:
        daily = daily.loc[daily["sample_coverage_ratio"] >= .7].copy()
    if daily.empty:
        st.info("No observed days match these settings. Widen the date range.")
        st.stop()

    series_col = "daily_mean_ccu" if day_measure == "Daily average CCU" else "daily_peak_ccu"
    top = daily[series_col].max()
    end = daily.sort_values("day_utc")[series_col].iloc[-1]
    denom = daily["daily_mean_ccu"].max()
    ret = 100 * daily["daily_mean_ccu"].iloc[-1] / denom if denom > 0 else np.nan
    adjacent = daily["day_utc"].diff().dt.days.eq(1)
    previous = daily["daily_mean_ccu"].shift()
    drops = ((daily["daily_mean_ccu"] / previous) - 1) * 100
    drops = drops.where(adjacent & previous.gt(0))
    one, two, three, four = st.columns(4)
    one.metric("Highest observed value", nice_int(top))
    two.metric("Last / peak avg CCU", nice_pct(ret))
    three.metric("Steepest consecutive-day drop", nice_pct(drops.min(), signed=True))
    four.metric("Observed days", nice_int(len(daily)))

    st.write("")
    with st.container(border=True):
        st.markdown(f"#### {escape(str(games.loc[chosen_id, 'game_name']))} — Daily activity")
        st.caption("Thin line = observed daily values; thick line = 7-day moving average (not forecast)")
        st.plotly_chart(
            daily_fig(daily, day_measure, relative=(daily_units == "Relative to peak (%)")),
            width="stretch", config={"displaylogo": False},
        )
        st.caption(
            f"Observed dates: {daily['day_utc'].min():%b %d, %Y} – "
            f"{daily['day_utc'].max():%b %d, %Y} · "
            f"Coverage threshold: {'70%' if coverage_only else 'none'}"
        )
    with st.container(border=True):
        st.markdown("#### Daily records")
        st.dataframe(
            daily[["day_utc", "daily_mean_ccu", "daily_peak_ccu", "sample_coverage_ratio"]]
            .rename(columns={
                "day_utc": "Date", "daily_mean_ccu": "Avg CCU",
                "daily_peak_ccu": "Observed peak CCU", "sample_coverage_ratio": "Sample coverage",
            }),
            hide_index=True, width="stretch",
        )
        st.download_button(
            "Download selected daily records", data=daily.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"steam_daily_{chosen_id}.csv", mime="text/csv",
        )

    st.info(
        "Daily data is loaded once per file version and held in memory for fast switching. "
        "Relative mode is normalized within the visible date range. "
        "Daily data is historical observation data from 2017–2020, not necessarily a game's "
        "launch period. The daily drop calculation uses only consecutive observed days."
    )

st.markdown('<div class="op-rule"></div>', unsafe_allow_html=True)
st.caption(
    "DATA 511 · Independent academic visualization. "
    "Monthly source: Kaggle Steam Monthly Average Players (lunthu). "
    "Mendeley daily history and metadata: Wannigamage, Barlow, Lakshika & Kasmarik (2020), "
    "DOI: 10.17632/ycy3sy3vj2.1, CC BY 4.0; derived daily/monthly aggregates. "
    "Not affiliated with Valve. CCU is not individual-user retention."
)
