# portfolio-trading-strategies

One dashboard for every trading strategy that publishes the daily-PnL results contract.

**The page shows the record since go-live only**: paper trading now, live trading later. Backtests are hindsight;
they appear in one reference table at the bottom and never in the chart or the figures.

- **Filters:** portfolio strategies, monitored strategies, date range with quick ranges.
- **Headline:** PnL, Sharpe, Sortino and max drawdown of the portfolio strategies combined, for the range.
  Sharpe and Sortino appear once the range covers 30 days; before that they would be noise.
- **Chart:** cumulative PnL stacked by strategy, re-based to zero at the start of the range, with the
  portfolio total as a line and monitored strategies as dashed lines.
- **Tables:** the same statistics per strategy with money at risk and data freshness, then each strategy's
  backtest results for reference.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

To host it, deploy this repository on Streamlit Community Cloud with `app.py` as the entry point. Public
sources need no secrets.

## Portfolio and monitoring

A strategy's manifest can carry `"role": "monitor"`. A monitored strategy is published and paper traded like any
other, and the dashboard shows it for comparison: a dashed line on the chart, its own row in the tables and its own
selector in the sidebar. It is never added to the portfolio figures at the top or to the total line. A manifest without
a role, or with `"role": "portfolio"`, is a portfolio strategy.

## Add a strategy

1. The strategy repository publishes the contract below (see
   [fear-greed-strategy](https://github.com/Jefersolitario/fear-greed-strategy) for a working example).
2. Add one block to `strategies.toml`:

```toml
[[source]]
name = "gb-pair-trading"
root = "gs://my-bucket/gb-pair-trading"      # or an https URL, or a local folder
```

Strategy ids must be unique across sources.

## The results contract

```
<root>/index.json                   {"contract_version": 1, "strategies": ["id", ...], "data_through": ..., "updated_utc": ...}
<root>/backtest_summary.parquet     optional; one row per tested variant, shown as published
<root>/<strategy_id>/manifest.json  id, name, market, currency, days_per_year, paper_start, role, ...
<root>/<strategy_id>/daily.parquet  one row per UTC day
```

| `daily.parquet` column | Type | Meaning |
|---|---|---|
| `date` | Date | UTC day |
| `mode` | String | `backtest`, `paper` or `live` |
| `pnl` | Float64 | That day's PnL in the manifest currency. Not cumulative, not a return. |
| `capital` | Float64, optional | Market value of open positions at the close (money at risk) |
| `n_trades` | Int32, optional | Fills that day |

Daily PnL is the unit because it adds up across strategies and re-bases to any date range. A strategy never
reports its own Sharpe: every statistic on the page is computed here, the same way for everyone.

- **Sharpe:** mean ÷ standard deviation of daily PnL × √(days per year), zero risk-free rate.
- **Sortino:** mean ÷ downside deviation (root mean square of losing days) × √(days per year).
- **Max drawdown:** largest peak-to-trough fall of cumulative PnL in the range; the curve starts at zero.
- **Sharpe per $ at risk:** Sharpe of daily PnL ÷ the previous day's `capital`. Shown when `capital` is
  published. It is the fairer number for rules that keep adding to a position.
- A day without a row counts as zero PnL. The combined figures use the calendar from the first to the last
  day any selected strategy has data in the range.

Only `paper` and `live` rows reach the chart and the figures. `backtest` rows are ignored by the page: a backtest is
hindsight, and its summary is shown once, in the reference table.

## Reading from Google Cloud Storage

Add a read-only service account key to the app's secrets (`.streamlit/secrets.toml` locally, the Secrets box
on Streamlit Community Cloud), then use `gs://` roots in `strategies.toml`:

```toml
[gcp_service_account]
type = "service_account"
project_id = "..."
private_key = "..."
client_email = "..."
# the remaining fields of the JSON key
```

## Limits

- **One currency.** Totals add currencies without conversion and the page warns when a selection mixes them.
  EUR and GBP strategies need an FX step before they are combined with USD ones.
- **Eight colours.** With more than eight strategies the chart keeps the first seven and adds the rest up as
  "Other". The tables always list every strategy.
- **Daily grain.** Intraday strategies publish their daily total here; order-level analysis belongs in the
  strategy's own tools.
- Results are cached for 15 minutes. "Reload data" in the sidebar clears the cache.

## Tests

```bash
pip install pytest
python -m pytest -q
```
