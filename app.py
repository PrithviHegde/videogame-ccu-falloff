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
ANY_GENRE, ANY_TYPE = "All genres", "All game types"
GAME_TYPES = [ANY_TYPE, "Single-player", "Multiplayer"]
REVIEWED = {"yes", "y", "true", "1", "x", "✓", "✔", "ok", "looks good"}   # "Looks Good?" answers

MAX_DAYS = 180           # data loaded and plotted; the slider zooms within it
MIN_DAYS = 14
SMOOTH_DAYS = 7          # rolling average that evens out weekend bumps
CHART_TOP_ONE_GAME = 328   # px taken by everything above and below the chart, with the stage cards
CHART_TOP_SEVERAL = 300    # ... without them (several games)
CHART_MIN_HEIGHT = 320     # the chart never gets shorter than this
CHART_HEIGHT = 350       # one game (px); with several, the line chart is shorter and
STRIP_ROW = 34            # px per game in the stage bars under a multi-game chart
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
    .block-container{max-width:1200px;padding-top:1.4rem;padding-bottom:1rem}
    [data-testid="stMain"] [data-testid="stVerticalBlock"]{gap:.75rem}
    h1,h2,h3,h4{color:#253249!important;letter-spacing:-.03em}
    div[data-testid="stVerticalBlockBorderWrapper"]{
      background:#FFFFFF;border:1px solid #E7EAF0!important;
      border-radius:16px!important;box-shadow:0 3px 18px rgba(32,48,69,.025)
    }
    [data-testid="stCaptionContainer"] p{color:#8793A3;font-size:.8rem}
    .op-name{font-family:Georgia,serif;font-size:1.6rem;color:#24344B;
      letter-spacing:-.035em;font-weight:650;line-height:1.15}
    [data-testid="stMarkdownContainer"]:has(> .op-name){margin-bottom:0!important}
    .op-pick{font-size:1.1rem;font-weight:700;color:#24344B;line-height:1.2}
    .op-picksub{font-size:.8rem;color:#7B8698;margin-top:1px}
    [data-testid="stMarkdownContainer"]:has(> .op-pick){margin-bottom:0!important}
    .op-stages{display:flex;flex-wrap:wrap;gap:10px;align-items:stretch;margin:0}
    /* Streamlit pulls markdown up by -1rem at the bottom, which made the cards
       overlap the chart box below; cancel it for the cards row only. */
    [data-testid="stMarkdownContainer"]:has(> .op-stages){margin-bottom:0!important}
    .op-card{flex:1 1 200px;background:#FFF;border:1px solid #E7EAF0;
      border-left:4px solid var(--accent);border-radius:10px;padding:10px 16px}
    .op-head{display:flex;align-items:center;gap:8px;font-size:.95rem;font-weight:650;color:#27364E}
    .op-num{display:inline-flex;align-items:center;justify-content:center;width:20px;height:20px;
      border-radius:50%;background:var(--accent);color:#FFF;font-size:.72rem;font-weight:700}
    .op-value{display:flex;align-items:baseline;flex-wrap:wrap;gap:0 8px;margin-top:4px}
    .op-big{font-size:1.6rem;font-weight:700;color:#1F2D44;letter-spacing:-.03em;line-height:1.15}
    .op-unit{font-size:.9rem;color:#66748A}
    .op-arrow{align-self:center;color:#B7C0CD;font-size:1.2rem}
    section[data-testid="stSidebar"]{background:#F2F4F8;border-right:1px solid #E3E7EE}
    [data-testid="stSidebarHeader"]{height:2.25rem;min-height:0;padding:.5rem 1rem 0;margin:0}
    [data-testid="stSidebarUserContent"]{padding-top:0}
    section[data-testid="stSidebar"] hr{margin:.5rem 0 .25rem}
    .op-side{font-family:Georgia,serif;font-size:1.35rem;font-weight:650;color:#24344B;
      letter-spacing:-.02em;line-height:1.2;padding-bottom:8px;border-bottom:1px solid #DCE1E9;
      margin:0 0 6px}
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


def filter_args(genre: str, game_type: str, years, prices, year_span: tuple,
                price_span: tuple) -> dict:
    """filter_games() arguments for the sidebar's current values; filters left
    at their defaults are not applied."""
    return dict(
        genres=[genre] if genre != ANY_GENRE else (),
        modes=[game_type, "Both"] if game_type != ANY_TYPE else (),   # "Both" fits either
        years=tuple(years) if tuple(years) != tuple(year_span) else None,
        prices=tuple(prices) if tuple(prices) != tuple(price_span) else None,
    )


def option_counts(info: pd.DataFrame, args: dict, field: str, options: list[str]) -> dict:
    """How many games each option would show, given the other active filters
    (like the counts next to filters on shopping sites)."""
    counts = {}
    for option in options:
        trial = dict(args)
        trial[field] = [option, "Both"] if field == "modes" else [option]
        counts[option] = len(filter_games(info, **trial))
    return counts


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


def base_layout(fig: go.Figure, height: int | None) -> None:
    fig.update_layout(
        template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family="Segoe UI, Inter, sans-serif", color=INK, size=13),
        margin=dict(l=10, r=10, t=12, b=6), height=height, hovermode="closest",
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=GRID, font=dict(color=INK)),
        legend=dict(orientation="h", y=-0.15, x=0, title=""),
        # No transition animation: when a game is added, the chart changes size and
        # content at once, and Plotly's animation then leaves it half-redrawn
        # (lines cut off, axis in the wrong place).
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


def strip_left_margin(ids: list[int], names: dict) -> int:
    """Left margin (px) that fits the longest game name in the stage bars; the
    line chart above uses the same margin so both share one days axis."""
    return int(min(290, 7 * max(len(names[i]) for i in ids) + 44))


def stage_strip(data: pd.DataFrame, ids: list[int], stages: dict, names: dict, colors: dict,
                days: int, left: int) -> go.Figure:
    """One bar per game on the days axis: ↗ growth, ↘ falloff, then where it ends up."""
    fig = go.Figure()
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
                textfont=dict(size=12, color="#FFFFFF" if alpha > .5 else INK),
                hovertemplate=(f"<b>{escape(name)}</b> · {escape(s['shape'])}<br>"
                               f"{escape(hover)}<extra></extra>"),
                showlegend=False, width=.78,
            ))
    base_layout(fig, STRIP_ROW * len(ids) + 58)
    fig.update_layout(barmode="overlay", uniformtext=dict(minsize=9, mode="hide"),
                      margin=dict(l=left, r=10, t=4, b=6))
    fig.update_yaxes(showgrid=False, automargin=False, tickfont=dict(color=INK, size=13))
    fig.update_xaxes(showgrid=False, showline=True, linecolor=GRID, zeroline=False,
                     tickfont=dict(color=MUTED), range=x_range(data, ids, SINCE_RELEASE, days),
                     title=SINCE_RELEASE)
    return fig


def trend_chart(data: pd.DataFrame, ids: list[int], stages: dict, colors: dict, names: dict,
                units: str, axis: str, days: int, left: int | None = None) -> go.Figure:
    """The line chart. It has no fixed height: it stretches to the bottom of the
    window (see the chart box on the page). With several games on the days axis,
    the stage bars are a separate chart below it (`stage_strip`), lined up through
    the shared left margin `left`."""
    single = len(ids) == 1
    strip = left is not None
    fig = go.Figure()
    for appid in ids:
        add_game(fig, data.loc[data["appid"].eq(appid)], names[appid], colors[appid],
                 units, axis, show_daily=single, legend=not single and not strip)
    if single:
        add_stage_bands(fig, data.loc[data["appid"].eq(appid)], stages[appid],
                        units, axis, days, labels=True)
    base_layout(fig, None)
    style_axes(fig, units)
    fig.update_xaxes(range=x_range(data, ids, axis, days),
                     title=SINCE_RELEASE if axis == SINCE_RELEASE and not strip else None)
    fig.update_yaxes(title="Share of peak players" if units == PCT else "Players online at once")
    if strip:
        fig.update_layout(margin=dict(l=left, r=10, t=12, b=4))
        fig.update_yaxes(automargin=False)
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
    base_layout(fig, max(300, nrows * 230))
    fig.update_layout(margin=dict(t=60))
    return fig


# ------------------------------ page ------------------------------

def stage_cards(s: dict, accent: str) -> str:
    """Three cards in reading order: a heading, then a big number with its context
    on the same line. Extra detail is in each card's hover tooltip."""
    def card(num: int, title: str, big: str, unit: str, color: str, tip: str) -> str:
        return (f'<div class="op-card" style="--accent:{color}" title="{escape(tip)}">'
                f'<div class="op-head"><span class="op-num">{num}</span>{title}</div>'
                f'<div class="op-value"><span class="op-big">{big}</span>'
                f'<span class="op-unit">{unit}</span></div></div>')

    arrow = '<div class="op-arrow">→</div>'
    d = s["peak_date"]
    tail = f"days {s['tail_start']}–{s['last_day']}"
    end_tip = f"Average daily peak over {tail}"
    if s["shape"] == GROWS:
        growth_tip = f"7-day average grew to ≈ {s['growth_top']:,.0f} by day {s['growth_day']}"
        if s["has_falloff"]:
            growth_tip += f", then lost {s['lost_pct']:.0f}% by day {s['knee']}"
        cards = [
            card(1, "Launch week", f"{s['launch_week']:,.0f}", "players",
                 accent, "Average daily peak over days 0–6"),
            card(2, "Growth", change_text(s["growth_x"]), f"by day {s['growth_day']}",
                 GROWTH_INK, growth_tip),
            card(3, tail.capitalize(), f"≈ {s['settled']:,.0f}", vs_launch_text(s["end_vs_launch"]),
                 SETTLED_INK, end_tip),
        ]
    else:
        if s["shape"] == LATE_PEAK:
            first = card(1, "Climb", change_text(s["growth_x"]), f"by day {s['growth_day']}",
                         GROWTH_INK, f"From launch week to ≈ {s['growth_top']:,.0f} (7-day average)")
        else:
            first = card(1, "Launch peak", f"{s['peak']:,.0f}", f"on day {s['peak_day']}",
                         accent, f"Most players at once: {d:%b} {d.day}, {d.year}")
        if s["has_falloff"]:
            second = card(2, "Falloff", f"−{s['lost_pct']:.0f}%", f"in {s['falloff_days']} days",
                          FALLOFF_INK, f"Leveling off on day {s['knee']}")
        else:
            second = card(2, "Falloff", "—", "no clear leveling-off", FALLOFF_INK,
                          "No clear leveling-off within 180 days")
        cards = [first, second,
                 card(3, "Settled", f"≈ {s['settled']:,.0f}", vs_launch_text(s["end_vs_launch"]),
                      SETTLED_INK, end_tip)]
    return f'<div class="op-stages">{arrow.join(cards)}</div>'


st.markdown(
    '<div class="op-name">How fast do players leave after launch?</div>',
    unsafe_allow_html=True,
)

data = load_data(DATA_PATH.stat().st_mtime_ns)
names = data.drop_duplicates("appid").set_index("appid")["game"].to_dict()
game_ids = list(names)
colors = {appid: PALETTE[i % len(PALETTE)] for i, appid in enumerate(game_ids)}
stages = {appid: game_stages(data.loc[data["appid"].eq(appid)]) for appid in game_ids}

info = load_info(INFO_PATH.stat().st_mtime_ns if INFO_PATH.is_file() else None)
FILTER_KEYS = ["f_price", "f_year", "f_genre", "f_mode"]

if len(info):
    sub = info.loc[[i for i in game_ids if i in info.index]]
    all_genres = sorted({g for gs in sub["genre_list"] for g in gs})
    year_span = (int(sub["year"].min()), int(sub["year"].max()))
    top_price = sub["launch_price_usd"].max()
    price_span = (0, int(math.ceil((top_price if pd.notna(top_price) else 70) / 10) * 10))
    DEFAULTS = {"f_price": price_span, "f_year": year_span, "f_genre": ANY_GENRE, "f_mode": ANY_TYPE}


def clear_filters() -> None:
    """Reset every filter. Values are set explicitly: removing a slider's key
    leaves the old handles on screen."""
    st.session_state.update(DEFAULTS)
    st.session_state["chips"] = None


def remove_chip() -> None:
    """Clicking an active-filter chip resets that one filter."""
    key = st.session_state.get("chip_keys", {}).get(st.session_state.get("chips"))
    if key:
        st.session_state[key] = DEFAULTS[key]
    st.session_state["chips"] = None


def mark(label: str, changed: bool) -> str:
    """A colored dot on a filter's label while it's in use."""
    return f"{label} :red[●]" if changed else label


DEFINITIONS = """
**Launch peak**: the most people playing at the same time, and the day it happened.

**Growth** (games that grow first): from launch to the busiest point.

**Falloff**: from the busiest point to where the decline levels off, i.e. where the
curve bends from falling fast to nearly flat. Shown as the share of players lost and
how many days it took.

**Settled**: where the game ends up: the average daily peak over days 150–179,
compared with launch week (days 0–6).

*Players* are each day's peak number of people playing at the same time. The thick
line is a 7-day average, which evens out weekends.
"""

if "games" not in st.session_state:
    st.session_state["games"] = game_ids[:1]

# ---- Sidebar: game search first, then the filters ----
shown = game_ids
with st.sidebar:
    st.markdown('<div class="op-side">Filter games</div>', unsafe_allow_html=True)
    if len(info):
        state = st.session_state
        for key, default in DEFAULTS.items():      # first load
            state.setdefault(key, default)
        args = filter_args(state["f_genre"], state["f_mode"], state["f_year"], state["f_price"],
                           year_span, price_span)
        changed = {"f_price": args["prices"] is not None, "f_year": args["years"] is not None,
                   "f_genre": bool(args["genres"]), "f_mode": bool(args["modes"])}
        if any(changed.values()):
            allowed = filter_games(sub, **args)
            shown = [i for i in game_ids if i in allowed]

    # Picked games stay on the chart even if they don't match the filters:
    # filters only narrow the list of games you can add.
    picked = st.session_state.get("games") or []
    options = [i for i in game_ids if i in shown or i in picked]
    count = (f"{len(shown)} of {len(game_ids)} match the filters" if len(shown) < len(game_ids)
             else f"All {len(game_ids)} games")
    st.markdown(f'<div class="op-pick">Select games</div><div class="op-picksub">{count}</div>',
                unsafe_allow_html=True)
    ids = st.multiselect("Select games", options=options, key="games", format_func=names.get,
                         label_visibility="collapsed",
                         placeholder="Search for a game to add…" if shown
                         else "No games match the filters")

    if len(info):
        chips = {}
        if changed["f_genre"]:
            chips[state["f_genre"]] = "f_genre"
        if changed["f_mode"]:
            chips[state["f_mode"]] = "f_mode"
        if changed["f_price"]:
            chips[f"${state['f_price'][0]}–${state['f_price'][1]}"] = "f_price"
        if changed["f_year"]:
            chips[f"{state['f_year'][0]}–{state['f_year'][1]}"] = "f_year"
        if chips:
            state["chip_keys"] = chips
            st.pills("Active filters", list(chips), key="chips", on_change=remove_chip,
                     format_func=lambda c: f"{c}  ✕", help="Click a filter to remove it.")
            st.button("Clear all filters", key="clear_filters", on_click=clear_filters,
                      width="stretch")
        st.divider()
        st.slider(mark("Launch price", changed["f_price"]), *price_span, step=5, key="f_price",
                  format="$%d", help="Price on Steam at release; $0 = free to play.")
        st.slider(mark("Release year", changed["f_year"]), *year_span, key="f_year")
        genre_counts = option_counts(sub, args, "genres", all_genres)
        st.selectbox(mark("Genre", changed["f_genre"]), [ANY_GENRE] + all_genres, key="f_genre",
                     format_func=lambda g: g if g == ANY_GENRE else f"{g} ({genre_counts[g]})")
        type_counts = option_counts(sub, args, "modes", GAME_TYPES[1:])
        st.selectbox(mark("Game type", changed["f_mode"]), GAME_TYPES, key="f_mode",
                     format_func=lambda t: t if t == ANY_TYPE else f"{t} ({type_counts[t]})",
                     help="The game's main way to play, checked by the team. "
                          "Games that are both count in either.")
        unreviewed = int((~sub["reviewed"]).sum())
        if unreviewed:
            st.caption(f"⚠️ Launch prices and game types are drafts: the team has "
                       f"checked {len(sub) - unreviewed} of {len(sub)} games so far.")

# ---- Chart controls ----
with st.container(border=True):
    days_col, units_col, axis_col = st.columns([2.4, 2, 2.4], gap="large",
                                               vertical_alignment="bottom")
    with days_col:
        days = st.slider("Days after release", min_value=MIN_DAYS, max_value=MAX_DAYS,
                         value=MAX_DAYS, step=1, key="days")
    with units_col:
        units = st.segmented_control(
            "Values as", [PCT, PLAYERS], default=PCT, required=True, key="units", width="stretch",
            help="% of peak puts games of different sizes on the same scale: "
                 "100% is the game's busiest day.")
    with axis_col:
        axis = st.segmented_control(
            "Time axis", [SINCE_RELEASE, CALENDAR], default=SINCE_RELEASE, required=True,
            key="axis", width="stretch",
            format_func=lambda a: "Since release" if a == SINCE_RELEASE else "Calendar")

if not ids:
    st.info("Pick a game in the sidebar to start." if shown
            else "No games match these filters. Remove one, or click “Clear all filters”.")
    st.stop()

ids = [i for i in game_ids if i in ids]  # keep a stable order and colors

if len(ids) == 1:
    cards_col, info_col = st.columns([11, 1], vertical_alignment="center", gap="small")
    cards_col.markdown(stage_cards(stages[ids[0]], colors[ids[0]]), unsafe_allow_html=True)
    with info_col.popover("Info", icon=":material/info:", type="tertiary",
                          help="What do these numbers mean?"):
        st.markdown(DEFINITIONS)

# The line chart reaches down to the bottom of the window, whatever its size: its
# height is the window height minus what sits above and below it (title, controls,
# the stage cards for one game, the stage bars for several), with a minimum so it
# never gets cramped. Plotly fills that space ("responsive" also refits it when the
# window is resized).
layout = st.session_state.get("layout", "One chart") if len(ids) > 1 else "One chart"
strip = len(ids) > 1 and layout == "One chart" and axis == SINCE_RELEASE
if layout == "One chart":
    above = (CHART_TOP_ONE_GAME if len(ids) == 1
             else CHART_TOP_SEVERAL + (STRIP_ROW * len(ids) + 58 if strip else 0))
    height = f"max({CHART_MIN_HEIGHT}px, calc(100vh - {above}px))"
    st.html("<style>"
            # Doubled class names: more specific than Streamlit's own height rules.
            # Streamlit gives the chart a fixed "flex: 0 0 450px" slot; replace its size.
            f".st-key-chart_box .st-key-trend.st-key-trend{{flex:0 0 {height}!important;"
            f"height:{height}!important}}"
            ".st-key-chart_box .st-key-trend .stFullScreenFrame.stFullScreenFrame,"
            ".st-key-chart_box .st-key-trend .stPlotlyChart.stPlotlyChart,"
            ".st-key-chart_box .st-key-trend .js-plotly-plot.js-plotly-plot,"
            ".st-key-chart_box .st-key-trend .plot-container.plot-container"
            "{height:100%!important}</style>")

with st.container(border=True, key="chart_box"):
    if len(ids) > 1:
        toggle_col, info_col = st.columns([11, 1], vertical_alignment="center", gap="small")
        with toggle_col:
            layout = st.segmented_control("Layout", ["One chart", "Side by side"],
                                          default="One chart", required=True, key="layout",
                                          label_visibility="collapsed")
        with info_col.popover("Info", icon=":material/info:", type="tertiary",
                              help="What do the stages mean?"):
            st.markdown(DEFINITIONS)
    config = {"displaylogo": False, "responsive": True}
    if layout == "One chart":
        left = strip_left_margin(ids, names) if strip else None
        st.plotly_chart(trend_chart(data, ids, stages, colors, names, units, axis, days, left),
                        width="stretch", config=config, key="trend")
        if strip:
            st.plotly_chart(stage_strip(data, ids, stages, names, colors, days, left),
                            width="stretch", config=config, key="strip")
    else:
        st.plotly_chart(small_multiples(data, ids, stages, colors, names, units, axis, days),
                        width="stretch", config=config, key="trend")
