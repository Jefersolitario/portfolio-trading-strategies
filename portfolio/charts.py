"""Stacked cumulative PnL by strategy with the portfolio total on top."""
from __future__ import annotations

import altair as alt
import polars as pl

# Fixed categorical order. A strategy keeps its colour whatever else is selected.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER = "#8a8983"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
MAX_SERIES = len(PALETTE)


def colour_map(all_names: list[str]) -> dict[str, str]:
    """Colour by position in the full registry, not by rank in the current selection."""
    return {n: PALETTE[i] if i < MAX_SERIES else OTHER for i, n in enumerate(all_names)}


def fold_other(long: pl.DataFrame, colours: dict[str, str]) -> tuple[pl.DataFrame, dict[str, str]]:
    """More strategies than colours: keep the first seven, add the rest up as 'Other'."""
    names = [n for n in colours if n in set(long["strategy"].unique())]
    if len(names) <= MAX_SERIES:
        return long, {n: colours[n] for n in names}
    keep = names[: MAX_SERIES - 1]
    label = f"Other ({len(names) - len(keep)})"
    rest = (long.filter(~pl.col("strategy").is_in(keep)).group_by("date").agg(pl.col("cum_pnl").sum())
            .with_columns(pl.lit(label).alias("strategy"), pl.lit("other").alias("strategy_id")))
    long = pl.concat([long.filter(pl.col("strategy").is_in(keep)), rest.select(long.columns)])
    return long, {**{n: colours[n] for n in keep}, label: OTHER}


def stacked_cumulative(long: pl.DataFrame, colours: dict[str, str], currency: str = "USD") -> alt.Chart:
    alt.data_transformers.disable_max_rows()
    long, colours = fold_other(long, colours)
    order = {n: i for i, n in enumerate(colours)}
    df = long.with_columns(pl.col("strategy").replace_strict(order, default=99).alias("rank")).to_pandas()
    total = long.group_by("date").agg(pl.col("cum_pnl").sum().alias("total")).sort("date").to_pandas()
    money = alt.Axis(format="$~s", title=f"Cumulative PnL ({currency})", gridColor=GRID, domain=False, tickSize=0, labelColor=INK2, titleColor=INK2)
    span = (long["date"].max() - long["date"].min()).days
    fmt, ticks = ("%Y", "year") if span > 800 else (("%b %Y", "month") if span > 100 else ("%d %b", "week"))
    time = alt.Axis(title=None, grid=False, domainColor=GRID, tickColor=GRID, labelColor=INK2, format=fmt, tickCount=ticks, labelOverlap=True)
    # Vega-Lite stacks positive and negative values on separate sides of zero, so a losing strategy
    # sits below the axis instead of eating into the winners.
    area = alt.Chart(df).mark_area(opacity=0.88, line={"strokeWidth": 1, "color": "#fcfcfb"}).encode(
        x=alt.X("date:T", axis=time),
        y=alt.Y("cum_pnl:Q", stack="zero", axis=money),
        color=alt.Color("strategy:N", scale=alt.Scale(domain=list(colours), range=list(colours.values())),
                        legend=alt.Legend(title=None, orient="top", labelColor=INK, labelLimit=400, symbolType="square")),
        order=alt.Order("rank:Q"),
        tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("strategy:N", title="Strategy"),
                 alt.Tooltip("cum_pnl:Q", title="Cumulative PnL", format="$,.0f")],
    )
    line = alt.Chart(total).mark_line(color=INK, strokeWidth=2).encode(
        x="date:T", y="total:Q",
        tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("total:Q", title="Portfolio total", format="$,.0f")],
    )
    zero = alt.Chart(pl.DataFrame({"y": [0.0]}).to_pandas()).mark_rule(color=INK2, strokeWidth=1).encode(y="y:Q")
    return (area + zero + line).properties(height=420).configure_view(strokeWidth=0)
