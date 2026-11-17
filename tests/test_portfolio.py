import datetime as dt
import json
import pathlib

import numpy as np
import polars as pl
import pytest

from portfolio import charts, loader, metrics

D0 = dt.date(2026, 1, 1)


def _write_source(root: pathlib.Path, strategies: dict[str, list[float]], mode: str = "backtest", start: dt.date = D0) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "index.json").write_text(json.dumps({"contract_version": 1, "strategies": list(strategies), "data_through": "2026-01-10"}))
    for sid, pnl in strategies.items():
        (root / sid).mkdir(exist_ok=True)
        (root / sid / "manifest.json").write_text(json.dumps({"id": sid, "name": sid.upper(), "currency": "USD", "days_per_year": 365}))
        pl.DataFrame({"date": [start + dt.timedelta(days=i) for i in range(len(pnl))], "mode": [mode] * len(pnl), "pnl": pnl,
                      "capital": [1000.0] * len(pnl), "n_trades": [1] * len(pnl)},
                     schema_overrides={"n_trades": pl.Int32}).write_parquet(root / sid / "daily.parquet")
    pl.DataFrame({"strategy_id": list(strategies), "name": [s.upper() for s in strategies], "pnl": [sum(v) for v in strategies.values()],
                  "max_dd": [1.0] * len(strategies)}).write_parquet(root / "backtest_summary.parquet")


@pytest.fixture
def config(tmp_path):
    _write_source(tmp_path / "a", {"alpha": [10, -5, 20, -40, 15], "beta": [-1, -2, -3, 4, 5]})
    _write_source(tmp_path / "b", {"gamma": [100, 100]}, start=D0 + dt.timedelta(days=3))
    cfg = tmp_path / "strategies.toml"
    cfg.write_text('[[source]]\nname = "a"\nroot = "a"\n\n[[source]]\nname = "b"\nroot = "b"\n\n[[source]]\nname = "missing"\nroot = "nowhere"\n')
    return cfg


def test_stats_known_series():
    s = metrics.stats([10, -5, 20, -40, 15])
    assert s["pnl"] == 0
    assert s["max_dd"] == 40                      # cumulative 10, 5, 25, -15, 0: peak 25 to trough -15
    d = np.array([10, -5, 20, -40, 15], dtype=float)
    assert s["sharpe"] == pytest.approx(d.mean() / d.std(ddof=1) * np.sqrt(365))
    assert s["sortino"] == pytest.approx(d.mean() / np.sqrt(np.mean(np.minimum(d, 0) ** 2)) * np.sqrt(365))


def test_drawdown_counts_a_loss_from_the_start():
    assert metrics.stats([-30, 10])["max_dd"] == 30          # the curve starts at zero
    assert metrics.stats([5, 5, 5])["max_dd"] == 0
    assert metrics.stats([])["sharpe"] is None


def test_loader_reads_every_source_and_reports_broken_ones(config):
    p = loader.load_all(config)
    assert set(p.manifests) == {"alpha", "beta", "gamma"}
    assert p.daily.height == 12
    assert p.summaries.height == 3 and "source" in p.summaries.columns
    assert len(p.errors) == 1 and "missing" in p.errors[0]


def test_wide_daily_fills_zeros_and_respects_the_range(config):
    p = loader.load_all(config)
    w = metrics.wide_daily(p.daily, ["alpha", "gamma"], "backtest", D0, D0 + dt.timedelta(days=30))
    assert w.columns == ["date", "alpha", "gamma"]
    assert w.height == 5
    assert w["gamma"].to_list() == [0, 0, 0, 100, 100]        # no row before its first day -> zero PnL
    total = w.select(pl.sum_horizontal("alpha", "gamma")).to_series()
    assert total.sum() == pytest.approx(200)
    narrow = metrics.wide_daily(p.daily, ["alpha"], "backtest", D0 + dt.timedelta(days=1), D0 + dt.timedelta(days=2))
    assert narrow["alpha"].to_list() == [-5, 20]
    assert metrics.wide_daily(p.daily, ["alpha"], "paper", D0, D0 + dt.timedelta(days=30)).height == 0


def test_portfolio_pnl_is_the_sum_of_the_strategies(config):
    p = loader.load_all(config)
    ids = ["alpha", "beta", "gamma"]
    w = metrics.wide_daily(p.daily, ids, "backtest", D0, D0 + dt.timedelta(days=30))
    total = metrics.stats(w.select(pl.sum_horizontal(ids)).to_series().to_numpy())["pnl"]
    assert total == pytest.approx(sum(metrics.stats(w[i].to_numpy())["pnl"] for i in ids))
    long = metrics.cumulative_long(w, {i: i.upper() for i in ids})
    last = long.filter(pl.col("date") == long["date"].max())
    assert last["cum_pnl"].sum() == pytest.approx(total)     # the stack ends at the portfolio total


def test_colours_follow_the_strategy_not_the_selection():
    names = [f"S{i}" for i in range(10)]
    cm = charts.colour_map(names)
    assert cm["S0"] == charts.PALETTE[0] and cm["S9"] == charts.OTHER
    long = pl.DataFrame({"date": [D0] * 10, "strategy_id": names, "strategy": names, "cum_pnl": [1.0] * 10})
    folded, used = charts.fold_other(long, cm)
    assert len(used) == 8 and "Other (3)" in used
    assert folded["cum_pnl"].sum() == pytest.approx(10)
    assert charts.stacked_cumulative(long.filter(pl.col("strategy").is_in(["S1", "S4"])), cm).to_dict()   # builds a valid spec


def test_app_renders(config, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("PORTFOLIO_CONFIG", str(config))
    at = AppTest.from_file(str(pathlib.Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
    assert not at.exception
    assert [m.label for m in at.metric] == ["PnL", "Sharpe", "Sortino", "Max drawdown"]
    assert at.metric[0].value == "$203"                      # alpha 0 + beta 3 + gamma 200
    at.multiselect[0].set_value(["gamma"]).run()
    assert not at.exception
    assert at.metric[0].value == "$200"
