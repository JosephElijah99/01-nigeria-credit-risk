"""Data ingestion from multiple sources.

Sources supported:
  * World Bank Open Data REST API  (macro indicators, with local cache + fallback)
  * CSV files and SQL databases     (generic loaders)
"""
from __future__ import annotations
import logging
from pathlib import Path

import pandas as pd
import requests

from .config import ROOT, load_config

log = logging.getLogger(__name__)
WB_URL = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"


def fetch_world_bank(country: str, indicator: str, start: int, end: int,
                     timeout: int = 20) -> pd.DataFrame:
    """Fetch one World Bank indicator -> tidy DataFrame [year, value]."""
    params = {"format": "json", "date": f"{start}:{end}", "per_page": 500}
    r = requests.get(WB_URL.format(country=country, indicator=indicator),
                     params=params, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    if len(payload) < 2 or payload[1] is None:
        raise ValueError(f"No data returned for {indicator}")
    rows = [{"year": int(d["date"]), "value": d["value"]} for d in payload[1]]
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def load_macro(cfg: dict | None = None, refresh: bool = False) -> pd.DataFrame:
    """Return annual macro table [year, inflation_rate, gdp_growth, unemployment_rate].

    Order of preference: live API (if refresh or no cache) -> cached CSV -> bundled fallback.
    """
    cfg = cfg or load_config()
    m = cfg["macro"]
    cache = ROOT / cfg["paths"]["external"] / "macro_cache.csv"
    fallback = ROOT / cfg["paths"]["external"] / "macro_fallback.csv"

    if refresh:
        try:
            frames = []
            for name, code in m["indicators"].items():
                df = fetch_world_bank(m["country"], code, m["start_year"], m["end_year"])
                frames.append(df.rename(columns={"value": name}).set_index("year"))
            macro = pd.concat(frames, axis=1).reset_index()
            cache.parent.mkdir(parents=True, exist_ok=True)
            macro.to_csv(cache, index=False)
            log.info("Macro data refreshed from World Bank API -> %s", cache)
            return _fill_latest(macro)
        except Exception as exc:  # network down, API change, etc.
            log.warning("World Bank API unavailable (%s); using local data.", exc)

    path = cache if cache.exists() else fallback
    log.info("Loading macro data from %s", path)
    return _fill_latest(pd.read_csv(path))


def _fill_latest(macro: pd.DataFrame) -> pd.DataFrame:
    """World Bank lags by a year; carry last value forward so joins don't produce NaN."""
    return macro.sort_values("year").ffill().reset_index(drop=True)


def load_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def load_sql(query: str, connection_string: str) -> pd.DataFrame:
    """Load from any SQLAlchemy-supported database (Postgres, MySQL, SQLite...)."""
    from sqlalchemy import create_engine  # optional dependency
    return pd.read_sql(query, create_engine(connection_string))
