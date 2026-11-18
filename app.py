"""Portfolio dashboard for trading strategies that publish the daily-PnL results contract.

The page shows the record since go-live: paper trading now, live trading later. Backtests are hindsight and
appear only in the reference table at the bottom.

    streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import os
import pathlib

import polars as pl
import streamlit as st

from portfolio import charts, loader, metrics

CONFIG = os.environ.get("PORTFOLIO_CONFIG", str(pathlib.Path(__file__).parent / "strategies.toml"))
RECORDS = {"Paper trading": "paper", "Live trading": "live"}
STALE_AFTER_DAYS = 3
MIN_DAYS_FOR_RATIOS = 30          # a Sharpe from a handful of days is noise, so it is not shown

st.set_page_config(page_title="Trading strategies", page_icon="📈", layout="wide")


def gcs_token():
    try:
        return dict(st.secrets["gcp_service_account"])
    except Exception:  # noqa: BLE001 - no secrets file, or no such section: public sources still work
        return None


@st.cache_data(ttl=900, show_spinner="Loading strategy results…")
def load(config_path: str) -> loader.Portfolio:
    return loader.load_all(config_path, gcs_token())


def money(x: float | None) -> str:
    if x is None:
        return "–"
    if abs(x) < 0.5:
        return "$0"                                       # no "-$0" for a rounding-size or negative-zero value
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def ratio(x: float | None) -> str:
    return "–" if x is None else f"{x:.2f}"


def apply_quick(quick_key: str, range_key: str, presets: dict, dmin: dt.date, dmax: dt.date) -> None:
    """A quick range is an action, not a state: set the date range, then clear the pill."""
    choice = st.session_state.get(quick_key)
    if choice in presets:
        st.session_state[range_key] = (max(presets[choice], dmin), dmax)
    st.session_state[quick_key] = None


def reference(data: loader.Portfolio, all_ids: list[str]) -> None:
    """Backtest results as published, and where the data comes from. Shown under every state of the page."""
    st.subheader("Backtest results (before go-live)")
    if data.summaries is None or data.summaries.height == 0:
        st.info("No source publishes a backtest summary.")
    else:
        t = data.summaries
        if "strategy_id" in t.columns:
            show_all = st.toggle("Include tested variants that are not in the portfolio", value=True)
            if not show_all:
                t = t.filter(pl.col("strategy_id").is_in(all_ids))
        labels = {"name": "Strategy", "role": "Role", "paper_traded": "Paper traded", "buys": "Trades", "pnl": "PnL", "return_pct": "Return %",
                  "sharpe": "Sharpe", "sortino": "Sortino", "max_dd": "Max drawdown", "sharpe_per_dollar": "Sharpe per $ at risk",
                  "placebo_p": "Placebo p", "put_in": "Money in", "start": "From", "end": "To", "source": "Source"}
        if "max_dd" in t.columns:
            t = t.with_columns(-pl.col("max_dd"))
        if "role" in t.columns:
            t = t.with_columns(pl.col("role").replace({"portfolio": "Portfolio", "monitor": "Monitoring", "tested": "Tested only"}))
        t = t.with_columns([pl.col(c).round(0).cast(pl.Int64) for c in ("pnl", "max_dd", "put_in") if c in t.columns])
        t = t.select([c for c in labels if c in t.columns] + [c for c in t.columns if c not in labels and c != "strategy_id"]).rename(
            {k: v for k, v in labels.items() if k in t.columns})
        whole = st.column_config.NumberColumn(format="localized")
        two = st.column_config.NumberColumn(format="%.2f")
        day = st.column_config.DateColumn(format="YYYY-MM-DD")
        st.dataframe(t.to_pandas(), hide_index=True, width="stretch", column_config={
            "PnL": whole, "Money in": whole, "Max drawdown": whole, "Trades": whole, "From": day, "To": day,
            "Return %": st.column_config.NumberColumn(format="%.0f%%"), "Sharpe": two, "Sortino": two, "Sharpe per $ at risk": two,
            "Placebo p": st.column_config.NumberColumn(format="%.2f", help="Share of random date-shifts of the trades that did at least as well. Small is good; above 0.05 is not evidence of an edge."),
        })
        st.caption("Hindsight, as published by each strategy for its full backtest period. Nothing above this table uses these figures.")
    with st.expander("Sources and data freshness"):
        for src in data.sources:
            st.write(f"**{src['name']}** · {len(src.get('strategies', []))} strategies · data through {src.get('data_through', '?')} · "
                     f"updated {src.get('updated_utc', '?')} · `{src['root']}`")
        for e in data.errors:
            st.warning(e)


data = load(CONFIG)
daily = data.daily
st.title("Trading strategies")

if daily.height == 0:
    st.error("No strategy results could be loaded. Check strategies.toml and the sources below.")
    for e in data.errors:
        st.write("•", e)
    st.stop()

names = {sid: m.get("name", sid) for sid, m in data.manifests.items()}
all_ids = list(names)
# role "monitor": published for comparison only. It is shown, and it never enters the portfolio figures.
roles = {sid: ("monitor" if m.get("role") == "monitor" else "portfolio") for sid, m in data.manifests.items()}
portfolio_ids = [x for x in all_ids if roles[x] == "portfolio"]
monitor_ids = [x for x in all_ids if roles[x] == "monitor"]
colours = charts.colour_map([names[s] for s in all_ids])
present = [label for label, m in RECORDS.items() if (daily["mode"] == m).any()] or ["Paper trading"]

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Filters")
    mode_label = present[-1] if len(present) == 1 else st.radio(
        "Record", present, index=len(present) - 1, help="Paper trading and live trading are separate records and are never mixed in one curve.")
    mode = RECORDS[mode_label]
    selected = st.multiselect("Portfolio strategies", options=portfolio_ids, default=portfolio_ids, format_func=lambda s: names[s])
    watched = []
    if monitor_ids:
        watched = st.multiselect("Monitoring only", options=monitor_ids, default=monitor_ids, format_func=lambda s: names[s],
                                 help="Shown for comparison as dashed lines. Never included in the portfolio figures.")
    in_mode = daily.filter(pl.col("mode") == mode)
    if in_mode.height:
        dmin, dmax = in_mode["date"].min(), in_mode["date"].max()
        key = f"range_{mode}"
        cur = st.session_state.get(key)
        if cur is None:
            st.session_state[key] = (dmin, dmax)
        elif len(cur) == 2:                               # keep a stored range inside the data after a reload
            lo, hi = max(cur[0], dmin), min(cur[1], dmax)
            st.session_state[key] = (lo, hi) if lo <= hi else (dmin, dmax)
        picked = st.date_input("Date range", min_value=dmin, max_value=dmax, key=key, format="YYYY-MM-DD")
        presets = {"1M": dmax - dt.timedelta(days=30), "3M": dmax - dt.timedelta(days=91), "YTD": dt.date(dmax.year, 1, 1),
                   "1Y": dmax - dt.timedelta(days=365), "All": dmin}
        st.pills("Quick ranges", list(presets), key=f"quick_{mode}", on_change=apply_quick, args=(f"quick_{mode}", key, presets, dmin, dmax))
    if st.button("Reload data", width="stretch"):
        load.clear()
        st.rerun()

# ---------------------------------------------------------------- states without a record to show
if in_mode.height == 0:
    starts = sorted({str(m.get("paper_start")) for m in data.manifests.values() if m.get("paper_start")})
    st.info(f"No {mode_label.lower()} days have been published yet."
            + (f" Paper trading starts on {', '.join(starts)}; each day appears after its UTC close, once the strategy's job has run." if mode == "paper" and starts else ""))
    reference(data, all_ids)
    st.stop()
shown = selected + watched
if not shown:
    st.info("Select at least one strategy in the sidebar.")
    reference(data, all_ids)
    st.stop()
if not isinstance(picked, (tuple, list)) or len(picked) != 2:
    st.info("Pick the end date of the range.")
    st.stop()
start, end = picked

wide = metrics.wide_daily(daily, shown, mode, start, end)
if wide.height == 0:
    st.info("The selected strategies have no rows in this date range.")
    reference(data, all_ids)
    st.stop()

# ---------------------------------------------------------------- headline
currencies = sorted({data.manifests[s].get("currency", "USD") for s in shown})
days_per_year = max(int(data.manifests[s].get("days_per_year", 365)) for s in shown)
# the portfolio figures use the portfolio strategies only
total = wide.select(pl.sum_horizontal(selected)).to_series().to_numpy() if selected else []
s = metrics.stats(total, days_per_year)
enough = wide.height >= MIN_DAYS_FOR_RATIOS
st.caption(f"{mode_label} since {dmin} · {len(selected)} of {len(portfolio_ids)} portfolio strategies · {wide['date'].min()} to {wide['date'].max()} · {wide.height:,} day{'' if wide.height == 1 else 's'}"
           + (f" · {len(watched)} monitored, not counted in these figures" if watched else ""))
if not selected:
    st.info("No portfolio strategy is selected, so there are no portfolio figures. Monitored strategies are shown below.")
if len(currencies) > 1:
    st.warning(f"The selection mixes currencies ({', '.join(currencies)}). Totals add them without conversion.")
too_few = f" Shown once the range has {MIN_DAYS_FOR_RATIOS} days."
c1, c2, c3, c4 = st.columns(4)
c1.metric("PnL", money(s["pnl"]))
c2.metric("Sharpe", ratio(s["sharpe"] if enough else None), help=f"Mean ÷ standard deviation of daily PnL × √{days_per_year}." + too_few)
c3.metric("Sortino", ratio(s["sortino"] if enough else None), help=f"Mean ÷ downside deviation of daily PnL × √{days_per_year}." + too_few)
c4.metric("Max drawdown", money(-s["max_dd"]), help="Largest peak-to-trough fall of cumulative PnL in the period.")
if not enough:
    st.caption(f"Sharpe and Sortino appear once the range covers {MIN_DAYS_FOR_RATIOS} days; this one covers {wide.height}.")

# ---------------------------------------------------------------- chart
st.subheader("Cumulative PnL")
long = metrics.cumulative_long(wide.select("date", *selected), names) if selected else None
long_watch = metrics.cumulative_long(wide.select("date", *watched), names) if watched else None
st.altair_chart(charts.stacked_cumulative(long, colours, currencies[0], monitors=long_watch), width="stretch")
st.caption("Areas are stacked by portfolio strategy and start at zero on the first day of the range. The black line is the portfolio total. "
           "A strategy with a loss is drawn below zero."
           + (" Dashed lines are monitored strategies: shown for comparison, not part of the total." if watched else ""))

# ---------------------------------------------------------------- per strategy
st.subheader("By strategy")
today = dt.datetime.now(dt.timezone.utc).date()
cur = currencies[0]
rows = []
for sid in shown:
    part = daily.filter((pl.col("strategy_id") == sid) & (pl.col("mode") == mode) & pl.col("date").is_between(start, end)).sort("date")
    dpy = int(data.manifests[sid].get("days_per_year", 365))
    r = metrics.stats(wide[sid].to_numpy(), dpy)
    last = part["date"].max() if part.height else None
    rows.append({
        "Strategy": names[sid], "Role": "Monitoring" if roles[sid] == "monitor" else "Portfolio", f"PnL ({cur})": round(r["pnl"]),
        "Sharpe": r["sharpe"] if enough else None, "Sortino": r["sortino"] if enough else None,
        f"Max drawdown ({cur})": -round(r["max_dd"]),
        "Sharpe per $ at risk": metrics.sharpe_per_dollar(part["pnl"].to_numpy(), part["capital"].to_numpy(), dpy) if part.height and enough else None,
        f"At risk now ({cur})": round(part["capital"].fill_null(0)[-1]) if part.height else 0,
        "Trades": int(part["n_trades"].fill_null(0).sum()) if part.height else 0,
        "Data through": last,
        "Status": "no rows" if last is None else ("stale" if (today - last).days > STALE_AFTER_DAYS else "ok"),
        "Source": data.manifests[sid].get("source", ""),
    })
whole = st.column_config.NumberColumn(format="localized")
two = st.column_config.NumberColumn(format="%.2f")
table = pl.DataFrame(rows, infer_schema_length=None)
if not enough:                                           # no ratios yet: leave the columns out instead of showing blanks
    table = table.drop("Sharpe", "Sortino", "Sharpe per $ at risk")
st.dataframe(table.to_pandas(), hide_index=True, width="stretch", column_config={
    f"PnL ({cur})": whole, f"Max drawdown ({cur})": whole, f"At risk now ({cur})": whole, "Sharpe": two, "Sortino": two, "Trades": whole,
    "Data through": st.column_config.DateColumn(format="YYYY-MM-DD"),
    "Sharpe per $ at risk": st.column_config.NumberColumn(format="%.2f", help="Sharpe of daily PnL ÷ money at risk the day before. Fairer for rules that keep adding to a position."),
})
st.caption("Each row uses the same calendar as the chart, with zero PnL on days a strategy has no row. Monitoring rows are not part of the portfolio figures.")

reference(data, all_ids)
