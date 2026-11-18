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
    initial_sidebar_state="collapsed",
)

DATA_PATH = Path(__file__).resolve().parent / "processed" / "launch_daily.parquet"

MAX_DAYS = 180           # data loaded and plotted; the slider zooms within it
MIN_DAYS = 14
SMOOTH_DAYS = 7          # rolling average that evens out weekend bumps
TAIL_DAYS = 30           # the settled level is the average of the last 30 days

INK = "#253249"
MUTED = "#7B8698"
GRID = "#E8ECF2"
PALETTE = ["#5578B7", "#C88077", "#539885", "#8876AF", "#D0A45D",
           "#4C8B9D", "#AB789E", "#7A8D5F", "#AE8970", "#64768F"]
FALLOFF_TINT = "rgba(200,128,119,.09)"
FALLOFF_INK = "#A4655D"
SETTLED_TINT = "rgba(83,152,133,.09)"
SETTLED_INK = "#3F7A69"
PCT = "% of peak"
PLAYERS = "Players"
SINCE_RELEASE = "Days since release"
CALENDAR = "Calendar date"

st.markdown(
    """
    <style>
    [data-testid="stApp"], [data-testid="stAppViewContainer"]{background:#F9FAFB;color:#253249}
    [data-testid="stHeader"]{background:transparent}
    [data-testid="stSidebarCollapsedControl"]{display:none}
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


def game_stages(game: pd.DataFrame, tail: int = TAIL_DAYS) -> dict:
    """Launch, falloff and settled stages for one game, from its full data.

    Falloff runs from the top of the 7-day average to the leveling-off day.
    Settled starts there; its level is the average daily peak over the last
    `tail` days, i.e. where the game ends up (the curve keeps sliding slowly
    after the leveling-off day, so an average over the whole stage would sit
    above most of it).

    Computed once on all MAX_DAYS days, so the stages don't move when the
    window slider changes.
    """
    game = game.sort_values("day")
    smooth = game.set_index("day")["smooth"]
    top = game["players"].max()
    peak_row = game.loc[game["players"].idxmax()]
    start = int(smooth.idxmax())                  # falloff starts at the smoothed peak
    knee = leveling_off_day(smooth)
    out = {
        "peak": top,
        "peak_day": int(peak_row["day"]),
        "peak_date": peak_row["date"],
        "falloff_start": start,
        "knee": knee,
        "lost_pct": None,
        "falloff_days": None,
        "half_life_days": None,
        "settled": None,
        "settled_pct": None,
        "last_day": int(game["day"].max()),
        "tail_start": int(game["day"].max()) - tail + 1,
    }
    if knee is None or knee <= start:
        return out
    kept = smooth[knee] / smooth[start]
    settled = game.loc[game["day"] >= max(knee, out["tail_start"]), "players"].mean()
    out.update(
        lost_pct=100 * (1 - kept),
        falloff_days=knee - start,
        # Kept for the "halves every N days" wording we may switch to later.
        half_life_days=(knee - start) * math.log(0.5) / math.log(kept) if 0 < kept < 1 else None,
        settled=settled,
        settled_pct=100 * settled / top,
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


def falloff_text(s: dict) -> str:
    return f"lost {s['lost_pct']:.0f}% in {s['falloff_days']} days"


def settled_text(s: dict) -> str:
    return f"≈ {compact(s['settled'])} players ({s['settled_pct']:.0f}% of peak)"


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
    """Shade the falloff and settled stages, with a dashed line at the leveling-off day.

    Labels sit at the top of each band, where the curve never is: every curve
    starts high on the left and ends low on the right.
    """
    if s["knee"] is None:
        return
    end = s["last_day"] + 0.5
    start, knee = s["falloff_start"], s["knee"]
    paper = f"{yref} domain"     # 0 = bottom of this plot, 1 = top
    for x0, x1, tint in ((start, knee, FALLOFF_TINT), (knee, end, SETTLED_TINT)):
        fig.add_shape(type="rect", xref=xref, yref=paper, x0=to_x(game, x0, axis),
                      x1=to_x(game, x1, axis), y0=0, y1=1, fillcolor=tint,
                      line_width=0, layer="below")
    fig.add_shape(type="line", xref=xref, yref=paper, x0=to_x(game, knee, axis),
                  x1=to_x(game, knee, axis), y0=0, y1=1,
                  line=dict(color=SETTLED_INK, width=1.3, dash="dash"))
    level = s["settled_pct"] if units == PCT else s["settled"]
    fig.add_shape(type="line", xref=xref, yref=yref,
                  x0=to_x(game, max(knee, s["tail_start"]) - 0.5, axis),
                  x1=to_x(game, end, axis), y0=level, y1=level,
                  line=dict(color=SETTLED_INK, width=2, dash="dot"))
    if not labels:
        return
    # Keep the falloff label inside the visible window when it's zoomed in.
    falloff_x = min(knee, days - 1)
    fig.add_annotation(
        x=to_x(game, falloff_x, axis), y=0.97, xref=xref, yref=paper,
        xanchor="right", yanchor="top", showarrow=False, align="right",
        text=f"<b>FALLOFF</b><br>{falloff_text(s)}",
        font=dict(size=12, color=FALLOFF_INK), xshift=-8,
    )
    if days - 1 - knee >= 25:   # only when the settled band has room in the window
        fig.add_annotation(
            x=to_x(game, knee, axis), y=0.97, xref=xref, yref=paper,
            xanchor="left", yanchor="top", showarrow=False, align="left",
            text=f"<b>SETTLED</b> from day {knee}<br>{settled_text(s)}",
            font=dict(size=12, color=SETTLED_INK), xshift=8,
        )


def add_stage_strip(fig: go.Figure, ids: list[int], stages: dict, names: dict,
                    colors: dict, row: int) -> None:
    """One bar per game, split into launch / falloff / settled, on the days axis."""
    for appid in reversed(ids):          # first game on top
        s, name, color = stages[appid], names[appid], colors[appid]
        end = s["last_day"] + 1
        if s["knee"] is None:
            segments = [(0, end, "", .15, "No clear leveling-off within 180 days")]
        else:
            start, knee = s["falloff_start"], s["knee"]
            segments = [
                (0, start, "", .15, f"Launch: peak {s['peak']:,.0f} players on day {s['peak_day']}"),
                (start, knee, f"−{s['lost_pct']:.0f}% in {s['falloff_days']}d", .85,
                 f"Falloff: {falloff_text(s)} (days {start}–{knee})"),
                (knee, end, settled_text(s), .25, f"Settled from day {knee}: {settled_text(s)}"),
            ]
        for x0, x1, text, alpha, hover in segments:
            if x1 <= x0:
                continue
            fig.add_trace(go.Bar(
                x=[x1 - x0], base=[x0], y=[name], orientation="h",
                marker=dict(color=color, opacity=alpha, line_width=0),
                text=[text], textposition="inside", insidetextanchor="middle",
                textfont=dict(size=11, color="#FFFFFF" if alpha > .5 else INK),
                hovertemplate=f"<b>{escape(name)}</b><br>{escape(hover)}<extra></extra>",
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
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            row_heights=[0.74, 0.26])
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
    titles = []
    for i in ids:
        s = stages[i]
        extra = (f"<br><span style='font-size:11px;color:{MUTED}'>{falloff_text(s)} · "
                 f"settles {settled_text(s)}</span>" if s["knee"] is not None else "")
        titles.append(f"{escape(names[i])}{extra}")
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

def stage_cards(s: dict, accent: str) -> str:
    """Three cards in reading order: launch → falloff → settled."""
    def card(step: str, big: str, small: str, color: str) -> str:
        return (f'<div class="op-card" style="--accent:{color}"><div class="op-step">{step}</div>'
                f'<div class="op-big">{big}</div><div class="op-small">{small}</div></div>')

    d = s["peak_date"]
    launch = card("1 · Launch", f"{s['peak']:,.0f} players",
                  f"most at once, on {day_label(s['peak_day']).lower()} "
                  f"({d:%b} {d.day}, {d.year})", accent)
    if s["knee"] is None:
        rest = card("2 · Falloff", "—", "No clear leveling-off within 180 days", FALLOFF_INK)
        return f'<div class="op-stages">{launch}<div class="op-arrow">→</div>{rest}</div>'
    falloff = card("2 · Falloff", f"Lost {s['lost_pct']:.0f}%",
                   f"in {s['falloff_days']} days, leveling off on day {s['knee']}",
                   FALLOFF_INK)
    settled = card("3 · Settled", f"≈ {s['settled']:,.0f} players",
                   f"{s['settled_pct']:.0f}% of peak, average of days "
                   f"{s['tail_start']}–{s['last_day']}", SETTLED_INK)
    arrow = '<div class="op-arrow">→</div>'
    return f'<div class="op-stages">{launch}{arrow}{falloff}{arrow}{settled}</div>'


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

with st.container(border=True):
    ids = st.multiselect(
        "Games", options=game_ids, default=game_ids[:1], format_func=names.get,
        placeholder="Add a game to compare…",
    )
    left, mid, right = st.columns([2, 1, 1], gap="large")
    with left:
        days = st.slider("Days after release", min_value=MIN_DAYS, max_value=MAX_DAYS,
                         value=MAX_DAYS, step=1)
    with mid:
        units = st.radio("Show", [PCT, PLAYERS], horizontal=True,
                         help="% of peak puts games of different sizes on the same scale: "
                              "100% is the game's busiest day.")
    with right:
        axis = st.radio("Time axis", [SINCE_RELEASE, CALENDAR], horizontal=True,
                        format_func=lambda a: "Since release" if a == SINCE_RELEASE else "Calendar")

if not ids:
    st.info("Pick at least one game above.")
    st.stop()

ids = [i for i in game_ids if i in ids]  # keep a stable order and colors

if len(ids) == 1:
    st.markdown(stage_cards(stages[ids[0]], colors[ids[0]]), unsafe_allow_html=True)

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
        notes.append("Shading: falloff, then settled; the dashed line is where the decline levels off.")
    elif axis == SINCE_RELEASE:
        notes.append("Bars below: each game's launch, falloff and settled stages.")
    else:
        notes.append("Switch the time axis to “Since release” to compare stages.")
    st.caption(" ".join(notes))

with st.expander("How to read this", expanded=False):
    st.markdown(
        "- **Players** is the highest number of people playing the game *at the same time* "
        "on that day (peak concurrent players). It is not the number of different people "
        "who played, and not retention.\n"
        "- **Day 0** is launch day. Days are counted in UTC, so day 0 can be the day after "
        "the date on the store page.\n"
        "- **The 7-day average** smooths out the weekly rhythm: more people play on weekends.\n"
        "- **% of peak**: 100% is the game's busiest day in its first six months.\n"
        "- **Stages.** *Falloff* runs from the top of the 7-day average to the "
        "**leveling-off point**, where the decline flattens out: the day the curve sits "
        "furthest below a straight line drawn from its peak to day 180. *Settled* starts "
        "there; its level (the dotted line) is the average daily peak over the last 30 days, "
        "where the game ends up. Stages always use all 180 days, so they don't move with "
        "the slider."
    )

st.markdown('<div class="op-rule"></div>', unsafe_allow_html=True)
st.caption(
    "DATA 511 · Independent academic visualization. "
    "Player counts: SteamDB (steamdb.info), daily peak concurrent players. "
    "Not affiliated with Valve or SteamDB."
)
