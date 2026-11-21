# portfolio-trading-strategies

One dashboard for every trading strategy that publishes the daily-PnL results contract.

The page has two tabs that never mix:

- **Paper trading:** the record since go-live. This is the default tab. A **Live trading** tab appears when a
  strategy publishes live days.
- **Backtest:** hindsight. The same views over the backtest period, plus the table of every tested variant.

Each tab has:

- **Date range** with quick ranges, set per tab.
- **Headline:** PnL, Sharpe, Sortino and max drawdown of the portfolio strategies combined, for the range.
  Sharpe and Sortino appear once the range covers 30 days; before that they would be noise.
- **Chart:** cumulative PnL stacked by strategy, re-based to zero at the start of the range, with the
  portfolio total as a line and monitored strategies as dashed lines.
- **Table:** the same statistics per strategy; in the paper and live tabs also money at risk and data freshness.

The sidebar chooses the portfolio strategies and the monitored strategies for all tabs.

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
- **Calendars differ.** A crypto strategy has a row every day, a share strategy only on sessions. Each strategy's
  row in the table is computed on its own published days with its own `days_per_year`. The combined figures use
  every day on which at least one selected strategy published a row, and a strategy without a row that day
  counts as zero.
- A feed is marked stale after 3 days without a new row, or 5 for a weekday-only strategy.

Each tab reads one `mode` only, so `backtest` rows never reach the paper or live record and the other way round.

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
