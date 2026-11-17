"""Portfolio dashboard for trading strategies that publish the daily-PnL results contract.

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
MODES = {"Paper trading": "paper", "Backtest": "backtest", "Live": "live"}
STALE_AFTER_DAYS = 3

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
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def ratio(x: float | None) -> str:
    return "–" if x is None else f"{x:.2f}"


def apply_quick(quick_key: str, range_key: str, presets: dict, dmin: dt.date, dmax: dt.date) -> None:
    """A quick range is an action, not a state: set the date range, then clear the pill."""
    choice = st.session_state.get(quick_key)
    if choice in presets:
        st.session_state[range_key] = (max(presets[choice], dmin), dmax)
    st.session_state[quick_key] = None


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
colours = charts.colour_map([names[s] for s in all_ids])
present = [label for label, m in MODES.items() if (daily["mode"] == m).any()]
if "Paper trading" not in present:
    present = ["Paper trading"] + present            # always offered, so an empty paper book is visible as empty

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Filters")
    default_mode = "Paper trading" if (daily["mode"] == "paper").any() else "Backtest"
    mode_label = st.radio("Record", present, index=present.index(default_mode),
                          help="Backtest is hindsight. Paper trading is what the rules did on days nobody had seen. They are never mixed in one curve.")
    mode = MODES[mode_label]
    selected = st.multiselect("Strategies", options=all_ids, default=all_ids, format_func=lambda s: names[s])
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

# ---------------------------------------------------------------- empty states
if in_mode.height == 0:
    starts = sorted({str(m.get("paper_start")) for m in data.manifests.values() if m.get("paper_start")})
    st.info(f"No {mode_label.lower()} days have been published yet."
            + (f" Paper books start on {', '.join(starts)}." if mode == "paper" and starts else "")
            + " Switch the record to Backtest in the sidebar to see the history.")
    st.stop()
if not selected:
    st.info("Select at least one strategy in the sidebar.")
    st.stop()
if not isinstance(picked, (tuple, list)) or len(picked) != 2:
    st.info("Pick the end date of the range.")
    st.stop()
start, end = picked

wide = metrics.wide_daily(daily, selected, mode, start, end)
if wide.height == 0:
    st.info("The selected strategies have no rows in this date range.")
    st.stop()

# ---------------------------------------------------------------- headline
currencies = sorted({data.manifests[s].get("currency", "USD") for s in selected})
days_per_year = max(int(data.manifests[s].get("days_per_year", 365)) for s in selected)
total = wide.select(pl.sum_horizontal(selected)).to_series().to_numpy()
s = metrics.stats(total, days_per_year)
st.caption(f"{mode_label} · {len(selected)} of {len(all_ids)} strategies · {wide['date'].min()} to {wide['date'].max()} · {s['days']:,} days")
if len(currencies) > 1:
    st.warning(f"The selection mixes currencies ({', '.join(currencies)}). Totals add them without conversion.")
c1, c2, c3, c4 = st.columns(4)
c1.metric("PnL", money(s["pnl"]))
c2.metric("Sharpe", ratio(s["sharpe"]), help=f"Mean ÷ standard deviation of daily PnL × √{days_per_year}.")
c3.metric("Sortino", ratio(s["sortino"]), help=f"Mean ÷ downside deviation of daily PnL × √{days_per_year}.")
c4.metric("Max drawdown", money(-s["max_dd"]), help="Largest peak-to-trough fall of cumulative PnL in the period.")

# ---------------------------------------------------------------- chart
st.subheader("Cumulative PnL")
long = metrics.cumulative_long(wide, names)
st.altair_chart(charts.stacked_cumulative(long, colours, currencies[0]), width="stretch")
st.caption("Areas are stacked by strategy and start at zero on the first day of the range. The black line is the portfolio total. "
           "A strategy with a loss is drawn below zero.")

# ---------------------------------------------------------------- per strategy
st.subheader("By strategy")
today = dt.datetime.now(dt.timezone.utc).date()
cur = currencies[0]
rows = []
for sid in selected:
    part = daily.filter((pl.col("strategy_id") == sid) & (pl.col("mode") == mode) & pl.col("date").is_between(start, end)).sort("date")
    dpy = int(data.manifests[sid].get("days_per_year", 365))
    r = metrics.stats(wide[sid].to_numpy(), dpy)
    last = part["date"].max() if part.height else None
    rows.append({
        "Strategy": names[sid], f"PnL ({cur})": round(r["pnl"]), "Sharpe": r["sharpe"], "Sortino": r["sortino"],
        f"Max drawdown ({cur})": -round(r["max_dd"]),
        "Sharpe per $ at risk": metrics.sharpe_per_dollar(part["pnl"].to_numpy(), part["capital"].to_numpy(), dpy) if part.height else None,
        "Trades": int(part["n_trades"].fill_null(0).sum()) if part.height else 0,
        "Data through": last,
        "Status": "no rows" if last is None else ("stale" if mode != "backtest" and (today - last).days > STALE_AFTER_DAYS else "ok"),
        "Source": data.manifests[sid].get("source", ""),
    })
whole = st.column_config.NumberColumn(format="localized")
two = st.column_config.NumberColumn(format="%.2f")
st.dataframe(pl.DataFrame(rows).to_pandas(), hide_index=True, width="stretch", column_config={
    f"PnL ({cur})": whole, f"Max drawdown ({cur})": whole, "Sharpe": two, "Sortino": two, "Trades": whole,
    "Data through": st.column_config.DateColumn(format="YYYY-MM-DD"),
    "Sharpe per $ at risk": st.column_config.NumberColumn(format="%.2f", help="Sharpe of daily PnL ÷ money at risk the day before. Fairer for rules that keep adding to a position."),
})
st.caption("Each row uses the same calendar as the portfolio above, with zero PnL on days a strategy has no row.")

# ---------------------------------------------------------------- backtest results
st.subheader("Backtest results")
if data.summaries is None or data.summaries.height == 0:
    st.info("No source publishes a backtest summary.")
else:
    t = data.summaries
    if "strategy_id" in t.columns:
        show_all = st.toggle("Include tested variants that are not in the portfolio", value=True)
        if not show_all:
            t = t.filter(pl.col("strategy_id").is_in(all_ids))
    labels = {"name": "Strategy", "paper_traded": "In portfolio", "buys": "Trades", "pnl": "PnL", "return_pct": "Return %", "sharpe": "Sharpe",
              "sortino": "Sortino", "max_dd": "Max drawdown", "sharpe_per_dollar": "Sharpe per $ at risk", "placebo_p": "Placebo p",
              "put_in": "Money in", "start": "From", "end": "To", "source": "Source"}
    if "max_dd" in t.columns:
        t = t.with_columns(-pl.col("max_dd"))
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
    st.caption("As published by each strategy for its full backtest period. These figures do not change with the date range above.")

# ---------------------------------------------------------------- sources
with st.expander("Sources and data freshness"):
    for src in data.sources:
        st.write(f"**{src['name']}** · {len(src.get('strategies', []))} strategies · data through {src.get('data_through', '?')} · "
                 f"updated {src.get('updated_utc', '?')} · `{src['root']}`")
    for e in data.errors:
        st.warning(e)
