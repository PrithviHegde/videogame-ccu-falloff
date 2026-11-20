"""DATA 511 | Steam Player Observatory

How quickly do players leave a big game after launch? A guided look at the
first 180 days after release for a few well-known Steam games.

Run:
    python -m pip install -r requirements.txt
    python -m streamlit run app.py

Data: processed/launch_daily.parquet, built by build_launch_data.py from
SteamDB chart exports. "Players" is each day's peak number of people playing
at the same time (concurrent players), NOT unique users or retention.
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

DATA_PATH = Path(__file__).resolve().parent / "processed" / "launch_daily.parquet"
INFO_PATH = Path(__file__).resolve().parent / "data" / "game_info.csv"
NOT_GENRES = {"Free To Play", "Early Access"}   # business models, not genres
MODES = ["Single-player", "Multiplayer", "Both"]
ANY_GENRE, ANY_TYPE, ANY_PUBLISHER = "All genres", "All game types", "All publishers"
GAME_TYPES = [ANY_TYPE, "Single-player", "Multiplayer"]
PRICE_PAD, YEAR_PAD = 10, 2   # how far snapped filters reach beyond the picked games
REVIEWED = {"yes", "y", "true", "1", "x", "✓", "✔", "ok", "looks good"}   # "Looks Good?" answers

MAX_DAYS = 180           # data loaded and plotted; the slider zooms within it
MIN_DAYS = 14
SMOOTH_DAYS = 7          # rolling average that evens out weekend bumps
TAIL_DAYS = 30           # the settled level is the average of the last 30 days
LAUNCH_WEEK = 7          # "launch week" = days 0-6
LATE_PEAK_DAY = 14       # a 7-day-average top on or after this day counts as growth first
GROWS_PEAK_DAY = 60      # top this late (or ending above launch week) = "Grows"
MIN_FALLOFF_PCT = 25     # smaller drops after the top are not called a falloff

INK = "#253249"
MUTED = "#7B8698"
GRID = "#E8ECF2"
PALETTE = ["#5578B7", "#C88077", "#539885", "#8876AF", "#D0A45D",
           "#4C8B9D", "#AB789E", "#7A8D5F", "#AE8970", "#64768F"]
FALLOFF_TINT = "rgba(200,128,119,.09)"
FALLOFF_INK = "#A4655D"
SETTLED_TINT = "rgba(83,152,133,.09)"
SETTLED_INK = "#3F7A69"
GROWTH_TINT = "rgba(85,120,183,.09)"
GROWTH_INK = "#47689F"
FALLS_OFF, LATE_PEAK, GROWS = "Falls off", "Peaks late", "Grows"
SHAPE_ICON = {FALLS_OFF: "↘", LATE_PEAK: "↗↘", GROWS: "↗"}
PCT = "% of peak"
PLAYERS = "Players"
SINCE_RELEASE = "Days since release"
CALENDAR = "Calendar date"

st.markdown(
    """
    <style>
    [data-testid="stApp"], [data-testid="stAppViewContainer"]{background:#F9FAFB;color:#253249}
    [data-testid="stHeader"]{background:transparent}
    .block-container{max-width:1200px;padding-top:2rem;padding-bottom:3rem}
    h1,h2,h3,h4{color:#253249!important;letter-spacing:-.03em}
    div[data-testid="stVerticalBlockBorderWrapper"]{
      background:#FFFFFF;border:1px solid #E7EAF0!important;
      border-radius:16px!important;box-shadow:0 3px 18px rgba(32,48,69,.025)
    }
    [data-testid="stCaptionContainer"] p{color:#8793A3;font-size:.8rem}
    .op-name{font-family:Georgia,serif;font-size:2rem;color:#24344B;
      letter-spacing:-.035em;font-weight:650;line-height:1.15}
    .op-sub{font-size:.95rem;color:#5F6D82;margin:6px 0 18px;max-width:720px}
    .op-rule{border-top:1px solid #E9ECF1;margin:1.5rem 0 .8rem}
    .op-stages{display:flex;flex-wrap:wrap;gap:10px;align-items:stretch;margin:2px 0 14px}
    .op-card{flex:1 1 200px;background:#FFF;border:1px solid #E7EAF0;border-radius:13px;
      padding:13px 18px;border-top:3px solid var(--accent)}
    .op-step{font-size:.7rem;font-weight:750;letter-spacing:.11em;text-transform:uppercase;
      color:var(--accent)}
    .op-big{font-size:1.55rem;font-weight:650;color:#27364E;letter-spacing:-.03em;
      line-height:1.25;margin-top:4px}
    .op-small{font-size:.82rem;color:#708096;margin-top:2px}
    .op-arrow{align-self:center;color:#B7C0CD;font-size:1.2rem}
    .op-pattern{font-size:.9rem;color:#5F6D82;margin:0 0 8px}
    .op-pattern b{color:#27364E}
    .op-meta{font-size:.8rem;color:#8793A3;margin:-4px 0 10px}
    section[data-testid="stSidebar"]{background:#F2F4F8;border-right:1px solid #E3E7EE}
    .op-side{font-size:.72rem;font-weight:750;letter-spacing:.12em;text-transform:uppercase;color:#798BB0;margin:0 0 4px}
    @media (max-width:640px){.op-arrow{display:none}}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------ data ------------------------------

def add_smoothing(df: pd.DataFrame, window: int = SMOOTH_DAYS) -> pd.DataFrame:
    """Add a centred rolling average and % of peak, per game.

    Peak is the highest single day in the data (the first MAX_DAYS days), so
    zooming the window never rescales the lines. Missing days stay missing.
    """
    df = df.sort_values(["appid", "day"]).copy()
    g = df.groupby("appid")["players"]
    df["smooth"] = g.transform(
        lambda s: s.rolling(window, center=True, min_periods=window // 2 + 1).mean()
    )
    peak = g.transform("max").replace(0, np.nan)
    df["pct"] = 100 * df["players"] / peak
    df["smooth_pct"] = 100 * df["smooth"] / peak
    return df.reset_index(drop=True)


@st.cache_data(show_spinner="Loading player history…")
def load_data(mtime: int) -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df = df.loc[df["day"] < MAX_DAYS]
    # Keep the game order from build_launch_data.GAMES (first game = default).
    df["order"] = df.groupby("appid", sort=False).ngroup()
    return add_smoothing(df).sort_values(["order", "day"]).reset_index(drop=True)


def split_list(value) -> list[str]:
    """'Action|RPG' -> ['Action', 'RPG']; blank -> []."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def prepare_info(info: pd.DataFrame) -> pd.DataFrame:
    """Game details for the filters, indexed by AppID.

    genres_override (filled by hand) replaces Steam's genres when it's set.
    "Free To Play" and "Early Access" are dropped: they're business models,
    not genres (free games are found with the launch-price filter).
    """
    info = info.copy()
    override = info.get("genres_override", pd.Series(index=info.index, dtype=object))
    genres = override.where(override.map(lambda v: bool(split_list(v))), info["genres"])
    info["genre_list"] = genres.map(lambda v: [g for g in split_list(v) if g not in NOT_GENRES])
    info["publisher_list"] = info["publishers"].map(split_list)
    info["year"] = pd.to_datetime(info["release_date"]).dt.year
    info["launch_price_usd"] = pd.to_numeric(info["launch_price_usd"], errors="coerce")
    # Ticked off in the team's review sheet ("Looks Good?" column).
    looks_good = info.get("Looks Good?", pd.Series(index=info.index, dtype=object))
    info["reviewed"] = looks_good.map(lambda v: str(v).strip().casefold() in REVIEWED)
    return info.set_index("appid")


@st.cache_data(show_spinner=False)
def load_info(mtime: int | None) -> pd.DataFrame:
    """data/game_info.csv, prepared for filtering; empty if the file is missing."""
    if mtime is None:
        return pd.DataFrame()
    return prepare_info(pd.read_csv(INFO_PATH))


def filter_games(info: pd.DataFrame, genres=(), publishers=(), modes=(),
                 years: tuple | None = None, prices: tuple | None = None) -> set[int]:
    """AppIDs matching every active filter. A game matches a genre or publisher
    filter if it has any of the chosen ones. None or empty = filter not used."""
    keep = pd.Series(True, index=info.index)
    if genres:
        keep &= info["genre_list"].map(lambda gs: bool(set(gs) & set(genres)))
    if publishers:
        keep &= info["publisher_list"].map(lambda ps: bool(set(ps) & set(publishers)))
    if modes:
        keep &= info["main_mode"].isin(modes)
    if years:
        keep &= info["year"].between(*years)
    if prices:
        keep &= info["launch_price_usd"].between(*prices)   # unknown price: excluded
    return set(info.index[keep])


def game_meta(info: pd.DataFrame, appid: int) -> str:
    """'Action, RPG · FromSoftware, Inc., Bandai Namco · $59.99 at launch · Single-player · 2022'."""
    if appid not in info.index:
        return ""
    row = info.loc[appid]
    price = row["launch_price_usd"]
    parts = [
        ", ".join(row["genre_list"]),
        ", ".join(row["publisher_list"]),
        "" if pd.isna(price) else ("Free to play" if price == 0 else f"${price:,.2f} at launch"),
        row["main_mode"] if isinstance(row["main_mode"], str) else "",
        str(row["year"]),
        "" if row["reviewed"] else "not yet reviewed",
    ]
    return " · ".join(p for p in parts if p)


def snap_filters(info: pd.DataFrame, selected: list[int], price_span: tuple,
                 year_span: tuple) -> dict:
    """Filter values that frame the selected games, so the game list shows similar ones.

    - Launch price and release year: the selected games' range, widened by
      PRICE_PAD dollars and YEAR_PAD years (one game alone would otherwise
      leave a single price and year, i.e. nothing else to add).
    - Genre: one all selected games share; the most common one if several
      (so "games like Elden Ring" means the RPGs, not just Action games).
    - Game type: Single-player or Multiplayer if every selected game fits it
      ("Both" fits either).
    - Publisher: only when 2+ games are selected and share one.
    Anything the selection doesn't share stays at "All".
    """
    out = {"f_price": tuple(price_span), "f_year": tuple(year_span), "f_genre": ANY_GENRE,
           "f_mode": ANY_TYPE, "f_publisher": ANY_PUBLISHER}
    sel = info.loc[[i for i in selected if i in info.index]]
    if sel.empty:
        return out

    prices = sel["launch_price_usd"]
    if prices.notna().all():
        lo = max(price_span[0], int(math.floor((prices.min() - PRICE_PAD) / 5) * 5))
        hi = min(price_span[1], int(math.ceil((prices.max() + PRICE_PAD) / 5) * 5))
        out["f_price"] = (lo, hi)
    out["f_year"] = (max(year_span[0], int(sel["year"].min()) - YEAR_PAD),
                     min(year_span[1], int(sel["year"].max()) + YEAR_PAD))

    def most_common(shared: set, column: str) -> str:
        return max(sorted(shared), key=lambda v: sum(v in vs for vs in info[column]))

    shared_genres = set.intersection(*(set(g) for g in sel["genre_list"]))
    if shared_genres:
        out["f_genre"] = most_common(shared_genres, "genre_list")

    modes = sel["main_mode"]
    if modes.notna().all():
        found = set(modes)
        for mode in ("Single-player", "Multiplayer"):
            if mode in found and found <= {mode, "Both"}:
                out["f_mode"] = mode

    if len(sel) >= 2:
        shared_publishers = set.intersection(*(set(p) for p in sel["publisher_list"]))
        if shared_publishers:
            out["f_publisher"] = most_common(shared_publishers, "publisher_list")
    return out


def leveling_off_day(smooth: pd.Series) -> int | None:
    """Day the decline flattens out (the "knee" of the curve).

    On the smoothed curve from its peak to the end of the data, this is the day
    that sits furthest below a straight line from the peak to the last day
    (the Kneedle method). It doesn't depend on the smoothing much: changing the
    average from 5 to 14 days moves it by a few days at most.
    """
    s = smooth.dropna()
    if len(s) < 4:
        return None
    seg = s.loc[s.idxmax():]
    if len(seg) < 4 or seg.max() == seg.min():
        return None
    x = np.linspace(0, 1, len(seg))
    y = (seg.to_numpy() - seg.min()) / (seg.max() - seg.min())
    gap = (1 - x) - y
    if gap.max() <= 0:
        return None
    return int(seg.index[int(np.argmax(gap))])


def game_stages(game: pd.DataFrame, tail: int = TAIL_DAYS, late_peak_day: int = LATE_PEAK_DAY,
                grows_peak_day: int = GROWS_PEAK_DAY) -> dict:
    """Shape and stages for one game, from its full data.

    Shape (on the 7-day average):
      - "Falls off": busiest in the first two weeks, then declines.
      - "Peaks late": climbs for 2-8 weeks, then declines.
      - "Grows": busiest after ~2 months, or ends above its launch week.

    Stages:
      - Growth: from launch to the top of the 7-day average, if that's 2+ weeks in.
      - Falloff: from the top to the leveling-off day, if the game lost at
        least MIN_FALLOFF_PCT% on the way.
      - Settled / end: the average daily peak over the last `tail` days,
        i.e. where the game ends up.

    "End vs launch week" (last `tail` days vs days 0-6) works for every shape,
    so games that collapse and games that take off sit on one scale.

    Computed once on all MAX_DAYS days, so nothing moves with the window slider.
    """
    game = game.sort_values("day")
    smooth = game.set_index("day")["smooth"]
    top = game["players"].max()
    peak_row = game.loc[game["players"].idxmax()]
    last_day = int(game["day"].max())
    tail_start = last_day - tail + 1
    launch_week = game.loc[game["day"] < LAUNCH_WEEK, "players"].mean()
    end = game.loc[game["day"] >= tail_start, "players"].mean()
    top_day = int(smooth.idxmax())                # top of the 7-day average
    end_vs_launch = 100 * end / launch_week if launch_week > 0 else np.nan

    if top_day >= grows_peak_day or end >= launch_week:
        shape = GROWS
    elif top_day >= late_peak_day:
        shape = LATE_PEAK
    else:
        shape = FALLS_OFF

    out = {
        "shape": shape,
        "peak": top,
        "peak_day": int(peak_row["day"]),
        "peak_date": peak_row["date"],
        "launch_week": launch_week,
        "has_growth": top_day >= late_peak_day,
        "growth_day": top_day,
        "growth_top": smooth.max(),
        "growth_x": smooth.max() / launch_week if launch_week > 0 else np.nan,
        "falloff_start": top_day,
        "has_falloff": False,
        "knee": None,
        "lost_pct": None,
        "falloff_days": None,
        "half_life_days": None,
        "settled": end,
        "settled_pct": 100 * end / top,
        "end_vs_launch": end_vs_launch,
        "last_day": last_day,
        "tail_start": tail_start,
    }
    knee = leveling_off_day(smooth)
    if knee is None or knee <= top_day:
        return out
    kept = smooth[knee] / smooth[top_day]
    lost = 100 * (1 - kept)
    if lost < MIN_FALLOFF_PCT:
        return out
    out.update(
        has_falloff=True,
        knee=knee,
        lost_pct=lost,
        falloff_days=knee - top_day,
        # Kept for the "halves every N days" wording we may switch to later.
        half_life_days=(knee - top_day) * math.log(0.5) / math.log(kept) if 0 < kept < 1 else None,
    )
    return out


def compact(n: float) -> str:
    """67,401 -> '67k', 2,415,714 -> '2.4M'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 10_000:
        return f"{n / 1_000:.0f}k"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return f"{n:.0f}"


def change_text(ratio: float) -> str:
    """1.95 -> '+95%', 4.2 -> '4.2×'."""
    return f"{ratio:.1f}×" if ratio >= 2 else f"+{100 * (ratio - 1):.0f}%"


def up_text(ratio: float) -> str:
    """Card headline: 1.28 -> 'Up 28%', 5.3 -> 'Up 5.3×'."""
    return f"Up {ratio:.1f}×" if ratio >= 2 else f"Up {100 * (ratio - 1):.0f}%"


def vs_launch_text(pct: float) -> str:
    """Where a game ends up, compared with its launch week."""
    return f"{pct / 100:.1f}× launch week" if pct >= 200 else f"{pct:.0f}% of launch week"


def growth_text(s: dict) -> str:
    ratio = s["growth_x"]
    return (f"up {ratio:.1f}× by day {s['growth_day']}" if ratio >= 2
            else f"up {100 * (ratio - 1):.0f}% by day {s['growth_day']}")


def falloff_text(s: dict) -> str:
    return f"lost {s['lost_pct']:.0f}% in {s['falloff_days']} days"


def settled_text(s: dict) -> str:
    return f"≈ {compact(s['settled'])} players ({vs_launch_text(s['end_vs_launch'])})"


def shape_summary(s: dict) -> str:
    """One line per game: shape, then its stages, then where it ends up."""
    parts = []
    if s["has_growth"]:
        parts.append(growth_text(s))
    if s["has_falloff"]:
        parts.append(falloff_text(s))
    parts.append(f"ends at {vs_launch_text(s['end_vs_launch'])}")
    return f"{SHAPE_ICON[s['shape']]} {s['shape']}: " + ", ".join(parts)


# ----------------------------- charts -----------------------------

def day_label(day: int) -> str:
    return "Launch day" if day == 0 else f"Day {day} after launch"


def hover_text(game: pd.DataFrame, name: str) -> list[str]:
    return [
        f"<b>{escape(name)}</b><br>{day_label(r.day)} ({r.date:%b %d, %Y})<br>"
        f"Peak that day: {r.players:,.0f} players ({r.pct:.0f}% of top)<br>"
        f"7-day average: {r.smooth:,.0f} ({r.smooth_pct:.0f}%)"
        for r in game.itertuples()
    ]


def to_x(game: pd.DataFrame, day: float, axis: str):
    """Position of a day number on the chosen axis."""
    if axis == SINCE_RELEASE:
        return day
    return game["date"].iloc[0] + pd.Timedelta(days=float(day))


def x_range(data: pd.DataFrame, ids: list[int], axis: str, days: int):
    if axis == CALENDAR:
        sel = data.loc[data["appid"].isin(ids)]
        start = sel.loc[sel["day"].eq(0), "date"].min()
        end = sel.loc[sel["day"].eq(days - 1), "date"].max()
        return [start - pd.Timedelta(days=3), end + pd.Timedelta(days=3)]
    return [-1, days]


def style_axes(fig: go.Figure, units: str, **where) -> None:
    fig.update_xaxes(showline=True, linecolor=GRID, gridcolor=GRID, zeroline=False,
                     tickfont=dict(color=MUTED), **where)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED),
                     rangemode="tozero", ticksuffix="%" if units == PCT else "",
                     range=[0, 105] if units == PCT else None, **where)


def base_layout(fig: go.Figure, height: int) -> None:
    fig.update_layout(
        template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family="Segoe UI, Inter, sans-serif", color=INK, size=13),
        margin=dict(l=10, r=10, t=30, b=10), height=height, hovermode="closest",
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=GRID, font=dict(color=INK)),
        legend=dict(orientation="h", y=-0.15, x=0, title=""),
        # Animate between window sizes instead of jumping.
        transition=dict(duration=500, easing="cubic-in-out"),
        uirevision="keep",
    )


def add_game(fig: go.Figure, game: pd.DataFrame, name: str, color: str, units: str,
             axis: str, show_daily: bool, legend: bool, **where) -> None:
    raw, smooth = ("pct", "smooth_pct") if units == PCT else ("players", "smooth")
    x = game["date"] if axis == CALENDAR else game["day"]
    text = hover_text(game, name)
    if show_daily:
        fig.add_trace(go.Scatter(
            x=x, y=game[raw], mode="lines", name=f"{name} · daily peak",
            line=dict(color=color, width=1), opacity=.35, connectgaps=False,
            text=text, hovertemplate="%{text}<extra></extra>", showlegend=False,
        ), **where)
    fig.add_trace(go.Scatter(
        x=x, y=game[smooth], mode="lines", name=name, connectgaps=False,
        line=dict(color=color, width=3), text=text,
        hovertemplate="%{text}<extra></extra>", showlegend=legend,
    ), **where)


def add_stage_bands(fig: go.Figure, game: pd.DataFrame, s: dict, units: str, axis: str,
                    days: int, labels: bool, xref: str = "x", yref: str = "y") -> None:
    """Shade each stage: growth, falloff, settled.

    Labels go where the curve isn't: growth labels top-left (a rising curve
    starts low), falloff and settled labels top-right of their bands (a falling
    curve ends low), and a grower's end label bottom-right (it ends high).
    """
    end = s["last_day"] + 0.5
    paper = f"{yref} domain"     # 0 = bottom of this plot, 1 = top

    def band(x0, x1, tint):
        fig.add_shape(type="rect", xref=xref, yref=paper, x0=to_x(game, x0, axis),
                      x1=to_x(game, x1, axis), y0=0, y1=1, fillcolor=tint,
                      line_width=0, layer="below")

    def label(day, text, color, right, top=True):
        fig.add_annotation(
            x=to_x(game, day, axis), y=0.97 if top else 0.04, xref=xref, yref=paper,
            xanchor="right" if right else "left", yanchor="top" if top else "bottom",
            showarrow=False, align="right" if right else "left", text=text,
            font=dict(size=12, color=color), xshift=-8 if right else 8,
        )

    level = s["settled_pct"] if units == PCT else s["settled"]
    tail_line = dict(type="line", xref=xref, yref=yref,
                     x0=to_x(game, max(s["knee"] or 0, s["tail_start"]) - 0.5, axis),
                     x1=to_x(game, end, axis), y0=level, y1=level,
                     line=dict(color=SETTLED_INK, width=2, dash="dot"))

    if s["has_growth"]:
        band(-0.5, s["growth_day"], GROWTH_TINT)
        if labels:
            label(0, f"<b>GROWTH</b><br>{growth_text(s)}", GROWTH_INK, right=False)

    if s["has_falloff"]:
        start, knee = s["falloff_start"], s["knee"]
        band(start, knee, FALLOFF_TINT)
        band(knee, end, SETTLED_TINT)
        fig.add_shape(type="line", xref=xref, yref=paper, x0=to_x(game, knee, axis),
                      x1=to_x(game, knee, axis), y0=0, y1=1,
                      line=dict(color=SETTLED_INK, width=1.3, dash="dash"))
        fig.add_shape(**tail_line)
        if labels:
            # Keep the falloff label inside the visible window when it's zoomed in.
            label(min(knee, days - 1), f"<b>FALLOFF</b><br>{falloff_text(s)}",
                  FALLOFF_INK, right=True)
            if days - 1 - knee >= 35:   # only when the settled band has room in the window
                label(knee, f"<b>SETTLED</b> from day {knee}<br>{settled_text(s)}",
                      SETTLED_INK, right=False)
    else:
        fig.add_shape(**tail_line)
        if labels and days - 1 >= s["tail_start"]:
            label(days - 1, f"<b>DAYS {s['tail_start']}–{s['last_day']}</b><br>{settled_text(s)}",
                  SETTLED_INK, right=True, top=False)


def add_stage_strip(fig: go.Figure, ids: list[int], stages: dict, names: dict,
                    colors: dict, row: int) -> None:
    """One bar per game on the days axis: ↗ growth, ↘ falloff, then where it ends up."""
    for appid in reversed(ids):          # first game on top
        s, name, color = stages[appid], names[appid], colors[appid]
        end = s["last_day"] + 1
        end_label = f"≈ {compact(s['settled'])} · {vs_launch_text(s['end_vs_launch'])}"
        segments = []
        if s["has_growth"]:
            segments.append((0, s["growth_day"], f"↗ {change_text(s['growth_x'])}", .45,
                             f"Growth: {growth_text(s)}"))
        else:
            segments.append((0, s["falloff_start"], "", .15,
                             f"Launch: peak {s['peak']:,.0f} players on day {s['peak_day']}"))
        if s["has_falloff"]:
            start, knee = s["falloff_start"], s["knee"]
            segments += [
                (start, knee, f"↘ −{s['lost_pct']:.0f}% in {s['falloff_days']}d", .85,
                 f"Falloff: {falloff_text(s)} (days {start}–{knee})"),
                (knee, end, end_label, .25, f"Settled from day {knee}: {settled_text(s)}"),
            ]
        else:
            rest_start = s["growth_day"] if s["has_growth"] else s["falloff_start"]
            if s["has_growth"] and end - rest_start < 40:
                # No room after the growth: show where it ends up on the growth bar.
                x0, x1, text, alpha, hover = segments[0]
                segments[0] = (x0, x1, f"{text} · ends {vs_launch_text(s['end_vs_launch'])}",
                               alpha, f"{hover}; days {s['tail_start']}–{s['last_day']}: "
                                      f"{settled_text(s)}")
            segments.append((rest_start, end, end_label if end - rest_start >= 40 else "", .25,
                             f"Days {s['tail_start']}–{s['last_day']}: {settled_text(s)}"))
        label = f"{name}  {SHAPE_ICON[s['shape']]}"
        for x0, x1, text, alpha, hover in segments:
            if x1 <= x0:
                continue
            fig.add_trace(go.Bar(
                x=[x1 - x0], base=[x0], y=[label], orientation="h",
                marker=dict(color=color, opacity=alpha, line_width=0),
                text=[text], textposition="inside", insidetextanchor="middle",
                textfont=dict(size=11, color="#FFFFFF" if alpha > .5 else INK),
                hovertemplate=(f"<b>{escape(name)}</b> · {escape(s['shape'])}<br>"
                               f"{escape(hover)}<extra></extra>"),
                showlegend=False, width=.62,
            ), row=row, col=1)
    fig.update_layout(barmode="overlay", uniformtext=dict(minsize=9, mode="hide"))
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK, size=12), row=row, col=1)
    fig.update_xaxes(showgrid=False, row=row, col=1)


def trend_chart(data: pd.DataFrame, ids: list[int], stages: dict, colors: dict, names: dict,
                units: str, axis: str, days: int) -> go.Figure:
    single = len(ids) == 1
    strip = not single and axis == SINCE_RELEASE
    if strip:
        strip_share = min(0.55, (34 * len(ids) + 40) / (460 + 34 * len(ids) + 40))
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            row_heights=[1 - strip_share, strip_share])
        where = dict(row=1, col=1)
    else:
        fig, where = go.Figure(), {}
    for appid in ids:
        add_game(fig, data.loc[data["appid"].eq(appid)], names[appid], colors[appid],
                 units, axis, show_daily=single, legend=not single and not strip, **where)
    if single:
        appid = ids[0]
        add_stage_bands(fig, data.loc[data["appid"].eq(appid)], stages[appid],
                        units, axis, days, labels=True)
    if strip:
        add_stage_strip(fig, ids, stages, names, colors, row=2)
    base_layout(fig, 460 + (34 * len(ids) + 40 if strip else 0))
    style_axes(fig, units, **where)
    fig.update_xaxes(range=x_range(data, ids, axis, days))
    title = SINCE_RELEASE if axis == SINCE_RELEASE else None
    fig.update_xaxes(title=title, row=2 if strip else None, col=1 if strip else None)
    fig.update_yaxes(title="Share of peak players" if units == PCT else "Players online at once",
                     **where)
    return fig


def small_multiples(data: pd.DataFrame, ids: list[int], stages: dict, colors: dict,
                    names: dict, units: str, axis: str, days: int) -> go.Figure:
    ncols = 2
    nrows = math.ceil(len(ids) / ncols)
    titles = [f"{escape(names[i])}<br><span style='font-size:11px;color:{MUTED}'>"
              f"{escape(shape_summary(stages[i]))}</span>" for i in ids]
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=titles,
                        vertical_spacing=min(0.16, 0.36 / nrows), horizontal_spacing=0.08)
    for i, appid in enumerate(ids):
        r, c = divmod(i, ncols)
        where = dict(row=r + 1, col=c + 1)
        game = data.loc[data["appid"].eq(appid)]
        add_game(fig, game, names[appid], colors[appid], units, axis,
                 show_daily=True, legend=False, **where)
        n = r * ncols + c + 1
        suffix = "" if n == 1 else str(n)
        add_stage_bands(fig, game, stages[appid], units, axis, days, labels=False,
                        xref=f"x{suffix}", yref=f"y{suffix}")
        style_axes(fig, units, **where)
        fig.update_xaxes(range=x_range(data, [appid], axis, days), **where)
    base_layout(fig, max(340, nrows * 300))
    fig.update_layout(margin=dict(t=60))
    return fig


# ------------------------------ page ------------------------------

SHAPE_BLURB = {
    "Falls off": "busiest in its first two weeks, then declines",
    "Peaks late": "climbs for a few weeks, then declines",
    "Grows": "ends up above its launch week",
}


def stage_cards(s: dict, accent: str, meta: str = "") -> str:
    """The game's shape, then three cards in reading order."""
    def card(step: str, big: str, small: str, color: str) -> str:
        return (f'<div class="op-card" style="--accent:{color}"><div class="op-step">{step}</div>'
                f'<div class="op-big">{big}</div><div class="op-small">{small}</div></div>')

    arrow = '<div class="op-arrow">→</div>'
    pattern = (f'<div class="op-pattern"><b>{SHAPE_ICON[s["shape"]]} {s["shape"]}</b> · '
               f'{SHAPE_BLURB[s["shape"]]}</div>'
               + (f'<div class="op-meta">{escape(meta)}</div>' if meta else ""))
    d = s["peak_date"]
    end_small = f"{vs_launch_text(s['end_vs_launch'])}, average of days {s['tail_start']}–{s['last_day']}"

    if s["shape"] == GROWS:
        first = card("1 · Launch week", f"{s['launch_week']:,.0f} players",
                     "average daily peak, days 0–6", accent)
        growth_small = f"to ≈ {s['growth_top']:,.0f} by day {s['growth_day']} (7-day average)"
        if s["has_falloff"]:
            growth_small += f", then lost {s['lost_pct']:.0f}% by day {s['knee']}"
        second = card("2 · Growth", up_text(s["growth_x"]), growth_small, GROWTH_INK)
        third = card(f"3 · Days {s['tail_start']}–{s['last_day']}", f"≈ {s['settled']:,.0f} players",
                     end_small, SETTLED_INK)
    else:
        if s["shape"] == LATE_PEAK:
            first = card("1 · Climb", up_text(s["growth_x"]),
                         f"from launch week to ≈ {s['growth_top']:,.0f} by day {s['growth_day']} "
                         f"(7-day average)", GROWTH_INK)
        else:
            first = card("1 · Launch", f"{s['peak']:,.0f} players",
                         f"most at once, on {day_label(s['peak_day']).lower()} "
                         f"({d:%b} {d.day}, {d.year})", accent)
        if s["has_falloff"]:
            second = card("2 · Falloff", f"Lost {s['lost_pct']:.0f}%",
                          f"in {s['falloff_days']} days, leveling off on day {s['knee']}",
                          FALLOFF_INK)
        else:
            second = card("2 · Falloff", "—", "No clear leveling-off within 180 days", FALLOFF_INK)
        third = card("3 · Settled", f"≈ {s['settled']:,.0f} players", end_small, SETTLED_INK)
    return f'{pattern}<div class="op-stages">{first}{arrow}{second}{arrow}{third}</div>'


st.markdown(
    '<div class="op-name">How fast do players leave after launch?</div>'
    '<div class="op-sub">The most people playing each game at the same time, '
    'day by day for the first six months after release.</div>',
    unsafe_allow_html=True,
)

data = load_data(DATA_PATH.stat().st_mtime_ns)
names = data.drop_duplicates("appid").set_index("appid")["game"].to_dict()
game_ids = list(names)
colors = {appid: PALETTE[i % len(PALETTE)] for i, appid in enumerate(game_ids)}
stages = {appid: game_stages(data.loc[data["appid"].eq(appid)]) for appid in game_ids}

info = load_info(INFO_PATH.stat().st_mtime_ns if INFO_PATH.is_file() else None)
FILTER_KEYS = ["f_price", "f_year", "f_genre", "f_mode", "f_publisher"]


def clear_filters() -> None:
    for key in FILTER_KEYS:
        st.session_state.pop(key, None)


def mark(label: str, changed: bool) -> str:
    """A colored dot on a filter's label while it's off its default."""
    return f"{label} :red[●]" if changed else label


if len(info):
    sub = info.loc[[i for i in game_ids if i in info.index]]
    all_genres = sorted({g for gs in sub["genre_list"] for g in gs})
    all_publishers = sorted({p for ps in sub["publisher_list"] for p in ps}, key=str.casefold)
    year_span = (int(sub["year"].min()), int(sub["year"].max()))
    top_price = sub["launch_price_usd"].max()
    price_span = (0, int(math.ceil((top_price if pd.notna(top_price) else 70) / 10) * 10))


def follow_selection() -> None:
    """When games are picked or removed, move the filters to frame them."""
    picked = st.session_state.get("games")
    if picked and len(info):
        st.session_state.update(snap_filters(sub, picked, price_span, year_span))


if "games" not in st.session_state:          # first load: default game, filters framing it
    st.session_state["games"] = game_ids[:1]
    follow_selection()

# ---- Sidebar: filters (as in the design sketches) ----
shown = game_ids
with st.sidebar:
    st.markdown('<div class="op-side">Filters</div>', unsafe_allow_html=True)
    if len(info):
        state = st.session_state
        changed = {
            "f_price": tuple(state.get("f_price", price_span)) != price_span,
            "f_year": tuple(state.get("f_year", year_span)) != year_span,
            "f_genre": state.get("f_genre", ANY_GENRE) != ANY_GENRE,
            "f_mode": state.get("f_mode", ANY_TYPE) != ANY_TYPE,
            "f_publisher": state.get("f_publisher", ANY_PUBLISHER) != ANY_PUBLISHER,
        }
        prices = st.slider(mark("Launch price", changed["f_price"]), *price_span,
                           **({} if "f_price" in state else {"value": price_span}),
                           step=5, key="f_price", format="$%d",
                           help="Price on Steam at release; $0 = free to play.")
        years = st.slider(mark("Release year", changed["f_year"]), *year_span,
                          **({} if "f_year" in state else {"value": year_span}), key="f_year")
        genre = st.selectbox(mark("Genre", changed["f_genre"]), [ANY_GENRE] + all_genres,
                             key="f_genre")
        game_type = st.selectbox(mark("Game type", changed["f_mode"]), GAME_TYPES, key="f_mode",
                                 help="The game's main way to play, checked by the team. "
                                      "Games that are both count in either.")
        publisher = st.radio(mark("Publisher", changed["f_publisher"]),
                             [ANY_PUBLISHER] + all_publishers, key="f_publisher")
        active = sum(changed.values())
        st.button("Clear filters" + (f" ({active})" if active else ""), on_click=clear_filters,
                  disabled=not active, width="stretch")
        unreviewed = int((~sub["reviewed"]).sum())
        if unreviewed:
            st.caption(f"⚠️ Launch prices and game types are drafts: the team has "
                       f"checked {len(sub) - unreviewed} of {len(sub)} games so far.")
        if active:
            allowed = filter_games(
                sub,
                genres=[genre] if changed["f_genre"] else (),
                publishers=[publisher] if changed["f_publisher"] else (),
                modes=[game_type, "Both"] if changed["f_mode"] else (),
                years=years if changed["f_year"] else None,
                prices=prices if changed["f_price"] else None,
            )
            shown = [i for i in game_ids if i in allowed]

# ---- Game picker and chart controls ----
with st.container(border=True):
    # Keep the picked games that still match; if the filters removed all of
    # them, start from the first match instead of an empty chart.
    picked = st.session_state.get("games") or []
    kept = [g for g in picked if g in shown]
    st.session_state["games"] = kept if kept or not picked else shown[:1]
    ids = st.multiselect(
        "Games", options=shown, key="games", format_func=names.get, on_change=follow_selection,
        placeholder="Search for a game to add…" if shown else "No games match the filters",
        help="Type to search. Pick several games to compare them. The filters on the left "
             "move to match the games you pick, so this list shows similar games.",
    )
    st.caption(f"Showing {len(shown)} of {len(game_ids)} games: the filters on the left follow "
               f"your picks. Use “Clear filters” to see every game." if len(shown) < len(game_ids)
               else f"All {len(game_ids)} games available.")

    left, mid, right = st.columns([2, 1, 1], gap="large")
    with left:
        days = st.slider("Days after release", min_value=MIN_DAYS, max_value=MAX_DAYS,
                         value=MAX_DAYS, step=1, key="days")
    with mid:
        units = st.radio("Show", [PCT, PLAYERS], horizontal=True,
                         help="% of peak puts games of different sizes on the same scale: "
                              "100% is the game's busiest day.")
    with right:
        axis = st.radio("Time axis", [SINCE_RELEASE, CALENDAR], horizontal=True,
                        format_func=lambda a: "Since release" if a == SINCE_RELEASE else "Calendar")

if not ids:
    st.info("Pick at least one game above." if shown
            else "No games match these filters. Try removing one, or click “Clear filters”.")
    st.stop()

ids = [i for i in game_ids if i in ids]  # keep a stable order and colors

if len(ids) == 1:
    st.markdown(stage_cards(stages[ids[0]], colors[ids[0]], game_meta(info, ids[0]) if len(info) else ""),
                unsafe_allow_html=True)

with st.container(border=True):
    layout = "One chart"
    if len(ids) > 1:
        layout = st.radio("Layout", ["One chart", "Side by side"], horizontal=True,
                          label_visibility="collapsed")
    chart = trend_chart if layout == "One chart" else small_multiples
    fig = chart(data, ids, stages, colors, names, units, axis, days)
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False}, key="trend")
    notes = ["Thick line: 7-day average."]
    if len(ids) == 1 or layout != "One chart":
        notes.append("Thin line: each day's peak.")
        notes.append("Shading marks each stage (growth, falloff, settled); "
                     "the dashed line is where the decline levels off.")
    elif axis == SINCE_RELEASE:
        notes.append("Bars below: each game's stages (↗ growth, ↘ falloff) and where it ends up "
                     "compared with its launch week.")
    else:
        notes.append("Switch the time axis to “Since release” to compare stages.")
    st.caption(" ".join(notes))

with st.expander("How to read this", expanded=False):
    st.markdown(
        "- **Players** is the highest number of people playing the game *at the same time* "
        "on that day (peak concurrent players). It is not the number of different people "
        "who played, and not retention.\n"
        "- **Day 0** is the official release date on Steam. SteamDB counts days in UTC, "
        "so for games that unlocked after midnight UTC, day 0 is the next day.\n"
        "- **The 7-day average** smooths out the weekly rhythm: more people play on weekends.\n"
        "- **% of peak**: 100% is the game's busiest day in its first six months.\n"
        "- **Shapes.** ↘ *Falls off*: busiest in the first two weeks, then declines. "
        "↗↘ *Peaks late*: climbs for 2–8 weeks first. ↗ *Grows*: busiest after about two "
        "months, or ends above its launch week.\n"
        "- **Stages.** *Growth* runs from launch to the top of the 7-day average, when that's "
        "two or more weeks in. *Falloff* runs from the top to the **leveling-off point**, where "
        "the decline flattens out: the day the curve sits furthest below a straight line drawn "
        "from its top to day 180 (only drops of 25% or more count). *Settled* starts there; "
        "its level (the dotted line) is the average daily peak over the last 30 days.\n"
        "- **Compared with launch week**: the last 30 days' average divided by the average of "
        "days 0–6. It works for every shape: 5% means the game kept a twentieth of its launch "
        "crowd, 4× means it ended four times bigger.\n"
        "- Everything uses all 180 days, so nothing moves with the slider."
    )

st.markdown('<div class="op-rule"></div>', unsafe_allow_html=True)
st.caption(
    "DATA 511 · Independent academic visualization. "
    "Player counts: SteamDB (steamdb.info), daily peak concurrent players. "
    "Not affiliated with Valve or SteamDB."
)
