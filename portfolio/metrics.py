"""Statistics computed from daily PnL. The same definitions for every strategy and for the combined portfolio.

- PnL          sum of daily PnL in the period
- Sharpe       mean / standard deviation of daily PnL, times sqrt(days per year); risk-free rate zero
- Sortino      mean / downside deviation (root mean square of the losing days), times sqrt(days per year)
- Max drawdown largest peak-to-trough fall of cumulative PnL in the period, in currency; the curve starts at zero
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl


def stats(daily_pnl, days_per_year: int = 365) -> dict[str, float | None]:
    d = np.asarray(daily_pnl, dtype=float)
    if len(d) == 0:
        return {"pnl": 0.0, "sharpe": None, "sortino": None, "max_dd": 0.0, "days": 0}
    cum = np.concatenate([[0.0], np.cumsum(d)])
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    down = float(np.sqrt(np.mean(np.minimum(d, 0.0) ** 2)))
    return {
        "pnl": float(cum[-1]),
        "sharpe": float(d.mean() / sd * np.sqrt(days_per_year)) if sd > 0 else None,
        "sortino": float(d.mean() / down * np.sqrt(days_per_year)) if down > 0 else None,
        "max_dd": float((np.maximum.accumulate(cum) - cum).max()),
        "days": int(len(d)),
    }


def sharpe_per_dollar(daily_pnl, capital, days_per_year: int = 365) -> float | None:
    """Sharpe of the daily return on money at risk: pnl[t] / capital at the close of t-1. None without capital data."""
    p = np.asarray(daily_pnl, dtype=float)
    c = np.asarray(capital, dtype=float)
    if len(p) < 3 or np.isnan(c).all():
        return None
    prev = np.concatenate([[0.0], np.nan_to_num(c[:-1])])
    r = np.where(prev > 0, p / np.where(prev > 0, prev, 1.0), 0.0)[1:]
    if not (prev[1:] > 0).any() or r.std(ddof=1) == 0:
        return None
    return float(r.mean() / r.std(ddof=1) * np.sqrt(days_per_year))


def active_days(daily: pl.DataFrame, strategy_ids: list[str], mode: str, start: dt.date, end: dt.date) -> list[dt.date]:
    """Days in [start, end] on which at least one of the strategies published a row."""
    sel = daily.filter(pl.col("strategy_id").is_in(strategy_ids) & (pl.col("mode") == mode) & pl.col("date").is_between(start, end))
    return sel["date"].unique().sort().to_list()


def is_stale(last: dt.date, today: dt.date, days_per_year: int = 365, limit: int = 3) -> bool:
    """A feed is stale when its last day is more than `limit` days old. A weekday-only calendar gets two more days:
    Friday's row is still the latest on Tuesday morning, and on Wednesday morning after a Monday holiday."""
    return (today - last).days > (limit if days_per_year >= 300 else limit + 2)


def wide_daily(daily: pl.DataFrame, strategy_ids: list[str], mode: str, start: dt.date, end: dt.date) -> pl.DataFrame:
    """One row per calendar day, one PnL column per strategy, zeros where a strategy has no row.

    The calendar runs from the first day any selected strategy has data inside [start, end] to the last one.
    """
    sel = daily.filter(pl.col("strategy_id").is_in(strategy_ids) & (pl.col("mode") == mode) & pl.col("date").is_between(start, end))
    if sel.height == 0:
        return pl.DataFrame({"date": []}, schema={"date": pl.Date})
    cal = pl.DataFrame({"date": pl.date_range(sel["date"].min(), sel["date"].max(), "1d", eager=True)})
    wide = sel.pivot(on="strategy_id", index="date", values="pnl", aggregate_function="sum")
    out = cal.join(wide, on="date", how="left").fill_null(0.0)
    for sid in strategy_ids:
        if sid not in out.columns:
            out = out.with_columns(pl.lit(0.0).alias(sid))
    return out.select("date", *strategy_ids).sort("date")


def cumulative_long(wide: pl.DataFrame, names: dict[str, str], max_points: int = 1500) -> pl.DataFrame:
    """Cumulative PnL per strategy, re-based to zero at the start of the period, thinned for plotting."""
    ids = [c for c in wide.columns if c != "date"]
    cum = wide.with_columns([pl.col(c).cum_sum() for c in ids])
    if cum.height > max_points:
        step = int(np.ceil(cum.height / max_points))
        keep = sorted(set(range(0, cum.height, step)) | {cum.height - 1})
        cum = cum[keep]
    long = cum.unpivot(index="date", on=ids, variable_name="strategy_id", value_name="cum_pnl")
    return long.with_columns(pl.col("strategy_id").replace_strict(names, default=pl.col("strategy_id")).alias("strategy"))
