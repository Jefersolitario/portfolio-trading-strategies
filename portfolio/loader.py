"""Read the results contract from every source listed in strategies.toml.

A source root can be an https URL, a local folder or a gs:// prefix. Under it:

    index.json                      {"contract_version": 1, "strategies": [ids], ...}
    backtest_summary.parquet        optional, one row per tested variant
    <strategy_id>/manifest.json     id, name, market, currency, days_per_year, ...
    <strategy_id>/daily.parquet     date, mode, pnl, capital, n_trades

Daily PnL is the only thing the dashboard trusts; every statistic is computed here from it.
"""
from __future__ import annotations

import io
import json
import pathlib
import tomllib
from dataclasses import dataclass, field

import polars as pl

SUPPORTED_CONTRACT = 1
REQUIRED = {"date": pl.Date, "mode": pl.String, "pnl": pl.Float64}
OPTIONAL = {"capital": pl.Float64, "n_trades": pl.Int32}
LONG_SCHEMA = {"date": pl.Date, "mode": pl.String, "pnl": pl.Float64, "capital": pl.Float64, "n_trades": pl.Int32,
               "strategy_id": pl.String, "strategy": pl.String, "source": pl.String}


@dataclass
class Portfolio:
    daily: pl.DataFrame                                   # long: one row per strategy and day
    manifests: dict[str, dict] = field(default_factory=dict)   # key: strategy_id
    summaries: pl.DataFrame | None = None                 # backtest results, all sources
    sources: list[dict] = field(default_factory=list)     # index.json of each source, plus name and root
    errors: list[str] = field(default_factory=list)       # sources or strategies that could not be read


def read_bytes(uri: str, gcs_token=None) -> bytes:
    if uri.startswith(("http://", "https://")):
        import requests

        r = requests.get(uri, timeout=30)
        r.raise_for_status()
        return r.content
    if uri.startswith("gs://"):
        import gcsfs

        return gcsfs.GCSFileSystem(token=gcs_token).cat_file(uri)
    return pathlib.Path(uri).read_bytes()


def _join(root: str, *parts: str) -> str:
    return "/".join([root.rstrip("/"), *parts])


def _daily(raw: bytes, sid: str, name: str, source: str) -> pl.DataFrame:
    df = pl.read_parquet(io.BytesIO(raw))
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"daily.parquet is missing columns {missing}")
    cols = [pl.col(c).cast(t) for c, t in REQUIRED.items()]
    cols += [(pl.col(c).cast(t) if c in df.columns else pl.lit(None, dtype=t).alias(c)) for c, t in OPTIONAL.items()]
    df = df.select(cols).with_columns(pl.lit(sid).alias("strategy_id"), pl.lit(name).alias("strategy"), pl.lit(source).alias("source"))
    if df.select(pl.struct("date", "mode").is_duplicated().any()).item():
        raise ValueError("more than one row for a date")
    if df["pnl"].is_null().any():
        raise ValueError("missing pnl values")
    return df.sort("date")


def load_source(name: str, root: str, gcs_token=None) -> Portfolio:
    out = Portfolio(daily=pl.DataFrame(schema=LONG_SCHEMA))
    try:
        index = json.loads(read_bytes(_join(root, "index.json"), gcs_token))
    except Exception as e:  # noqa: BLE001 - one broken source must not take the dashboard down
        out.errors.append(f"{name}: cannot read index.json ({e})")
        return out
    if index.get("contract_version") != SUPPORTED_CONTRACT:
        out.errors.append(f"{name}: contract version {index.get('contract_version')} is not supported")
        return out
    out.sources.append(dict(index, name=name, root=root))
    frames = []
    for sid in index.get("strategies", []):
        try:
            manifest = json.loads(read_bytes(_join(root, sid, "manifest.json"), gcs_token))
            frames.append(_daily(read_bytes(_join(root, sid, "daily.parquet"), gcs_token), sid, manifest.get("name", sid), name))
            out.manifests[sid] = dict(manifest, source=name)
        except Exception as e:  # noqa: BLE001
            out.errors.append(f"{name}/{sid}: {e}")
    if frames:
        out.daily = pl.concat(frames)
    try:
        s = pl.read_parquet(io.BytesIO(read_bytes(_join(root, "backtest_summary.parquet"), gcs_token)))
        out.summaries = s.with_columns(pl.lit(name).alias("source"))
    except Exception:  # noqa: BLE001 - the summary is optional
        pass
    return out


def load_config(path: str | pathlib.Path) -> list[dict]:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    return cfg.get("source", [])


def load_all(config_path: str | pathlib.Path, gcs_token=None) -> Portfolio:
    out = Portfolio(daily=pl.DataFrame(schema=LONG_SCHEMA))
    frames, summaries = [], []
    base = pathlib.Path(config_path).resolve().parent
    for src in load_config(config_path):
        root = src["root"]
        if "://" not in root:
            root = str((base / root).resolve())            # local folders are relative to the config file
        p = load_source(src["name"], root, gcs_token)
        dup = set(p.manifests) & set(out.manifests)
        if dup:
            out.errors.append(f"{src['name']}: strategy ids already used by another source: {sorted(dup)}")
            continue
        frames.append(p.daily)
        out.manifests.update(p.manifests)
        out.sources += p.sources
        out.errors += p.errors
        if p.summaries is not None:
            summaries.append(p.summaries)
    if frames:
        out.daily = pl.concat(frames)
    if summaries:
        out.summaries = pl.concat(summaries, how="diagonal_relaxed")
    return out
