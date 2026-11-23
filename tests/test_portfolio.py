import datetime as dt
import json
import pathlib

import numpy as np
import polars as pl
import pytest

from portfolio import charts, loader, metrics

D0 = dt.date(2026, 1, 1)


def _write_source(root: pathlib.Path, strategies: dict[str, list[float]], mode: str = "paper", start: dt.date = D0,
                  roles: dict[str, str] | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "index.json").write_text(json.dumps({"contract_version": 1, "strategies": list(strategies), "data_through": "2026-01-10"}))
    for sid, pnl in strategies.items():
        (root / sid).mkdir(exist_ok=True)
        manifest = {"id": sid, "name": sid.upper(), "currency": "USD", "days_per_year": 365}
        if roles and sid in roles:
            manifest["role"] = roles[sid]
        (root / sid / "manifest.json").write_text(json.dumps(manifest))
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
    w = metrics.wide_daily(p.daily, ["alpha", "gamma"], "paper", D0, D0 + dt.timedelta(days=30))
    assert w.columns == ["date", "alpha", "gamma"]
    assert w.height == 5
    assert w["gamma"].to_list() == [0, 0, 0, 100, 100]        # no row before its first day -> zero PnL
    total = w.select(pl.sum_horizontal("alpha", "gamma")).to_series()
    assert total.sum() == pytest.approx(200)
    narrow = metrics.wide_daily(p.daily, ["alpha"], "paper", D0 + dt.timedelta(days=1), D0 + dt.timedelta(days=2))
    assert narrow["alpha"].to_list() == [-5, 20]
    assert metrics.wide_daily(p.daily, ["alpha"], "backtest", D0, D0 + dt.timedelta(days=30)).height == 0


def test_portfolio_pnl_is_the_sum_of_the_strategies(config):
    p = loader.load_all(config)
    ids = ["alpha", "beta", "gamma"]
    w = metrics.wide_daily(p.daily, ids, "paper", D0, D0 + dt.timedelta(days=30))
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


def test_monitored_strategies_are_shown_but_stay_out_of_the_portfolio_figures(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    _write_source(tmp_path / "a", {"alpha": [10, -5, 20, -40, 15], "beta": [-1, -2, -3, 4, 5], "watch": [100, 100, 100, 100, 100]}, roles={"watch": "monitor"})
    cfg = tmp_path / "strategies.toml"
    cfg.write_text('[[source]]\nname = "a"\nroot = "a"\n')
    monkeypatch.setenv("PORTFOLIO_CONFIG", str(cfg))
    at = AppTest.from_file(str(pathlib.Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
    assert not at.exception
    assert [m.label for m in at.multiselect] == ["Portfolio strategies", "Monitoring only"]
    assert at.multiselect[0].options == ["ALPHA", "BETA"] and at.multiselect[1].value == ["watch"]
    assert at.metric[0].value == "$3"                         # alpha 0 + beta 3; the monitored 500 is not counted
    table = at.dataframe[0].value
    assert dict(zip(table["Strategy"], table["Role"])) == {"ALPHA": "Portfolio", "BETA": "Portfolio", "WATCH": "Monitoring"}
    at.multiselect[0].set_value([]).run()                     # monitoring alone still renders, with no portfolio figures
    assert not at.exception
    assert at.metric[0].value == "$0"


def test_chart_builds_with_monitor_lines_and_without_a_portfolio():
    names = ["A", "B"]
    cm = charts.colour_map(names)
    port = pl.DataFrame({"date": [D0, D0 + dt.timedelta(days=1)], "strategy_id": ["a", "a"], "strategy": ["A", "A"], "cum_pnl": [1.0, 2.0]})
    mon = pl.DataFrame({"date": [D0, D0 + dt.timedelta(days=1)], "strategy_id": ["b", "b"], "strategy": ["B", "B"], "cum_pnl": [3.0, 1.0]})
    spec = charts.stacked_cumulative(port, cm, monitors=mon).to_dict()
    assert len(spec["layer"]) == 4                            # area, zero line, total, dashed monitor line
    last = charts.stacked_cumulative(None, cm, monitors=mon).to_dict()["layer"][-1]["mark"]
    assert last["strokeDash"] == [6, 4] and last["point"]     # a two-day record is drawn with visible points


def test_paper_and_backtest_live_in_separate_tabs(tmp_path, monkeypatch):
    """Backtest rows never reach the paper tab; ratios wait for 30 days of record."""
    from streamlit.testing.v1 import AppTest

    root = tmp_path / "a"
    _write_source(root, {"alpha": [1.0] * 400}, mode="backtest", start=dt.date(2025, 1, 1))      # history: +400
    paper = pl.DataFrame({"date": [D0 + dt.timedelta(days=i) for i in range(5)], "mode": ["paper"] * 5, "pnl": [0.0, 0.0, -1.0, 6.0, 5.0],
                          "capital": [0.0, 0.0, 999.0, 1005.0, 1010.0], "n_trades": [0, 0, 1, 0, 0]}, schema_overrides={"n_trades": pl.Int32})
    both = pl.concat([pl.read_parquet(root / "alpha" / "daily.parquet"), paper.select(pl.read_parquet(root / "alpha" / "daily.parquet").columns)])
    both.write_parquet(root / "alpha" / "daily.parquet")
    cfg = tmp_path / "strategies.toml"
    cfg.write_text('[[source]]\nname = "a"\nroot = "a"\n')
    monkeypatch.setenv("PORTFOLIO_CONFIG", str(cfg))
    at = AppTest.from_file(str(pathlib.Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
    assert not at.exception
    assert [tab.label for tab in at.tabs] == ["Paper trading", "Backtest"]
    paper, backtest = at.tabs
    assert paper.metric[0].value == "$10"                     # paper days only, not the +400 of history
    assert paper.metric[1].value == "–" and paper.metric[2].value == "–"   # five days: no Sharpe or Sortino yet
    assert paper.metric[3].value == "-$1"
    assert any("since 2026-01-01" in c.value for c in paper.caption)
    table = paper.dataframe[0].value
    assert table["Trades"].tolist() == [1] and table["At risk now (USD)"].tolist() == [1010]
    assert backtest.metric[0].value == "$400"                 # history only, none of the paper days
    assert backtest.metric[1].value == "–"                    # a constant +1 a day has no spread, so no Sharpe
    assert "At risk now (USD)" not in backtest.dataframe[0].value.columns
    assert len(backtest.dataframe) == 2                       # per strategy, then every tested variant


def test_before_the_first_paper_day_the_page_says_so_and_still_shows_the_backtest_table(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    _write_source(tmp_path / "a", {"alpha": [1.0, 2.0, 3.0]}, mode="backtest")
    cfg = tmp_path / "strategies.toml"
    cfg.write_text('[[source]]\nname = "a"\nroot = "a"\n')
    monkeypatch.setenv("PORTFOLIO_CONFIG", str(cfg))
    at = AppTest.from_file(str(pathlib.Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
    assert not at.exception
    paper, backtest = at.tabs
    assert len(paper.metric) == 0 and any("No paper trading days" in i.value for i in paper.info)
    assert backtest.metric[0].value == "$6" and len(backtest.dataframe) == 2


def test_ratios_appear_after_thirty_days(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    _write_source(tmp_path / "a", {"alpha": [3.0, -1.0] * 20})
    cfg = tmp_path / "strategies.toml"
    cfg.write_text('[[source]]\nname = "a"\nroot = "a"\n')
    monkeypatch.setenv("PORTFOLIO_CONFIG", str(cfg))
    at = AppTest.from_file(str(pathlib.Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
    assert not at.exception
    assert at.metric[1].value not in ("–", "") and float(at.metric[1].value) > 0


def test_each_strategy_is_measured_on_its_own_calendar(tmp_path, monkeypatch):
    """A weekday-only strategy that starts later is not diluted by the days it never traded."""
    from streamlit.testing.v1 import AppTest

    every_day = [3.0, -1.0, 2.0] * 40
    _write_source(tmp_path / "a", {"crypto": every_day})                                   # 365-day calendar from 1 Jan 2026
    start = dt.date(2026, 2, 2)                                                            # a Monday
    days = [d for d in (start + dt.timedelta(days=i) for i in range(80)) if d.weekday() < 5]
    shares = ([5.0, -2.0, 4.0, -1.0, 3.0] * 12)[:len(days)]
    root = tmp_path / "b"; (root / "shares").mkdir(parents=True)
    (root / "index.json").write_text(json.dumps({"contract_version": 1, "strategies": ["shares"], "data_through": str(days[-1])}))
    (root / "shares" / "manifest.json").write_text(json.dumps({"id": "shares", "name": "SHARES", "currency": "USD", "days_per_year": 252}))
    pl.DataFrame({"date": days, "mode": ["paper"] * len(days), "pnl": shares, "capital": [1000.0] * len(days), "n_trades": [0] * len(days)},
                 schema_overrides={"n_trades": pl.Int32}).write_parquet(root / "shares" / "daily.parquet")
    cfg = tmp_path / "strategies.toml"
    cfg.write_text('[[source]]\nname = "a"\nroot = "a"\n\n[[source]]\nname = "b"\nroot = "b"\n')
    monkeypatch.setenv("PORTFOLIO_CONFIG", str(cfg))
    at = AppTest.from_file(str(pathlib.Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
    assert not at.exception
    table = at.tabs[0].dataframe[0].value.set_index("Strategy")
    assert table.loc["SHARES", "Sharpe"] == pytest.approx(metrics.stats(shares, 252)["sharpe"])
    assert table.loc["CRYPTO", "Sharpe"] == pytest.approx(metrics.stats(every_day, 365)["sharpe"])
    assert table.loc["SHARES", "PnL (USD)"] == round(sum(shares))
    # the portfolio total still adds both on the shared calendar, where the crypto strategy has a row every day
    both = np.array(every_day); both[[(d - D0).days for d in days]] += np.array(shares)
    assert float(at.tabs[0].metric[1].value) == pytest.approx(metrics.stats(both, 365)["sharpe"], abs=0.006)

    # with only the weekday strategy selected, weekends are not part of the record
    at.multiselect[0].set_value(["shares"]).run()
    assert not at.exception
    assert float(at.tabs[0].metric[1].value) == pytest.approx(metrics.stats(shares, 252)["sharpe"], abs=0.006)


def test_a_weekday_feed_is_not_called_stale_over_a_weekend():
    friday = dt.date(2026, 11, 20)
    assert not metrics.is_stale(friday, dt.date(2026, 11, 24), days_per_year=252)          # Tuesday morning, before Monday's row lands
    assert not metrics.is_stale(friday, dt.date(2026, 11, 25), days_per_year=252)          # after a Monday holiday
    assert metrics.is_stale(friday, dt.date(2026, 11, 26), days_per_year=252)
    assert metrics.is_stale(friday, dt.date(2026, 11, 24), days_per_year=365)              # a daily feed four days old is stale
    assert not metrics.is_stale(friday, dt.date(2026, 11, 23), days_per_year=365)


def test_a_weekly_feed_is_stale_only_after_a_missed_week():
    monday = dt.date(2026, 11, 23)                                                          # one row a week, dated Monday
    assert not metrics.is_stale(monday, dt.date(2026, 11, 29), days_per_year=52)            # the Sunday after
    assert not metrics.is_stale(monday, dt.date(2026, 12, 1), days_per_year=52)             # next Monday's row is a day late
    assert metrics.is_stale(monday, dt.date(2026, 12, 2), days_per_year=52)                 # a week was missed


def test_ratios_wait_for_enough_days_of_the_series_they_are_computed_from(tmp_path, monkeypatch):
    """A young portfolio strategy beside a long-running monitored one: no headline ratio from three days, and each row judged on its own."""
    from streamlit.testing.v1 import AppTest

    young = [5.0, -2.0, 4.0]
    watch = [3.0, -1.0, 2.0, 1.0] * 10
    _write_source(tmp_path / "a", {"watch": watch}, roles={"watch": "monitor"})
    _write_source(tmp_path / "b", {"young": young}, start=D0 + dt.timedelta(days=37))
    cfg = tmp_path / "strategies.toml"
    cfg.write_text('[[source]]\nname = "a"\nroot = "a"\n\n[[source]]\nname = "b"\nroot = "b"\n')
    monkeypatch.setenv("PORTFOLIO_CONFIG", str(cfg))
    at = AppTest.from_file(str(pathlib.Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
    assert not at.exception
    paper = at.tabs[0]
    assert paper.metric[0].value == "$7" and paper.metric[1].value == "–" and paper.metric[2].value == "–"
    assert any("it has 3" in c.value for c in paper.caption)
    table = paper.dataframe[0].value.set_index("Strategy")
    assert table.loc["WATCH", "Sharpe"] == pytest.approx(metrics.stats(watch, 365)["sharpe"])
    assert table.loc["YOUNG", "Sharpe"] != table.loc["YOUNG", "Sharpe"]            # blank: three days are not enough
