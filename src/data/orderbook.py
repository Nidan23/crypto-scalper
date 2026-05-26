"""CCXT-based order book fetcher with local parquet caching.

Snapshots are collected at a configurable interval (default 10 s) and
stored per symbol per day so that historical lookups are fast.
"""

import hashlib
import logging
import time
from pathlib import Path
from typing import Optional

import ccxt
import pandas as pd

from src.config import config

logger = logging.getLogger(__name__)


def _ob_cache_dir() -> Path:
    """Resolve and create the order-book cache directory."""
    path = Path(config.orderbook_cache_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ob_cache_path(symbol: str, date_str: str) -> Path:
    """Deterministic parquet path for a symbol + day.

    Args:
        symbol: Trading pair, e.g. ``"BTC/USDT"``.
        date_str: ISO date string, e.g. ``"2024-06-15"``.

    Returns:
        Absolute path to the ``.parquet`` file for that day.
    """
    safe = symbol.replace("/", "_")
    key = f"{safe}_{date_str}"
    digest = hashlib.md5(key.encode()).hexdigest()
    return _ob_cache_dir() / f"{digest}.parquet"


def _snapshot_to_rows(
    timestamp: pd.Timestamp, bids: list, asks: list, depth: int
) -> dict:
    """Flatten a single OB snapshot into a flat dict of price/vol columns.

    Args:
        timestamp: Timestamp of this snapshot.
        bids: List of ``[price, volume]`` pairs (highest bid first).
        asks: List of ``[price, volume]`` pairs (lowest ask first).
        depth: Number of levels to keep.

    Returns:
        Dict with ``timestamp``, ``bid_0_price``, ``bid_0_vol``, ...,
        ``ask_N_price``, ``ask_N_vol``.
    """
    row = {"timestamp": timestamp}
    for i in range(min(len(bids), depth)):
        row[f"bid_{i}_price"] = float(bids[i][0])
        row[f"bid_{i}_vol"] = float(bids[i][1])
    for i in range(min(len(asks), depth)):
        row[f"ask_{i}_price"] = float(asks[i][0])
        row[f"ask_{i}_vol"] = float(asks[i][1])
    return row


class OrderBookFetcher:
    """Fetch and cache order-book snapshots from a CCXT exchange.

    Snapshots are stored per symbol per day as parquet files under
    ``config.orderbook_cache_dir``.  Each row is one snapshot with
    ``depth`` price/volume levels on each side.

    Parameters
    ----------
    exchange_id: CCXT exchange id (default from config).
    depth: Number of price levels to fetch (default from config).
    """

    def __init__(
        self,
        exchange_id: Optional[str] = None,
        depth: Optional[int] = None,
    ) -> None:
        self.exchange_id = exchange_id or config.exchange_id
        self.depth = depth or config.orderbook_depth
        self._exchange: Optional[ccxt.Exchange] = None

    @property
    def exchange(self) -> ccxt.Exchange:
        """Lazily initialised CCXT exchange instance."""
        if self._exchange is None:
            if not hasattr(ccxt, self.exchange_id):
                raise ValueError(
                    f"Unknown exchange '{self.exchange_id}'."
                )
            self._exchange = getattr(ccxt, self.exchange_id)(
                {"enableRateLimit": False}
            )
            self._exchange.load_markets()
        return self._exchange

    # ------------------------------------------------------------------
    # Live snapshot
    # ------------------------------------------------------------------

    def fetch_snapshot(self, symbol: str) -> Optional[dict]:
        """Fetch a single live order-book snapshot and cache it.

        Args:
            symbol: Trading pair, e.g. ``"BTC/USDT"``.

        Returns:
            Flattened snapshot dict, or ``None`` on failure.
        """
        try:
            ob = self.exchange.fetch_order_book(symbol, limit=self.depth)
        except Exception as exc:
            logger.warning("Order book fetch failed for %s: %s", symbol, exc)
            return None

        ts = pd.Timestamp.now("UTC")
        row = _snapshot_to_rows(ts, ob["bids"], ob["asks"], self.depth)
        self._append_to_cache(symbol, row)
        return row

    def _append_to_cache(self, symbol: str, row: dict) -> None:
        """Append a single snapshot row to the appropriate daily cache file."""
        ts = row["timestamp"]
        date_str = ts.strftime("%Y-%m-%d")
        cache_path = _ob_cache_path(symbol, date_str)

        new_row = pd.DataFrame([row])
        new_row.set_index("timestamp", inplace=True)

        if cache_path.exists():
            try:
                existing = pd.read_parquet(cache_path)
                combined = pd.concat([existing, new_row])
                combined.to_parquet(cache_path)
            except Exception:
                new_row.to_parquet(cache_path)
        else:
            new_row.to_parquet(cache_path)

    # ------------------------------------------------------------------
    # Historical lookup
    # ------------------------------------------------------------------

    def get_snapshots(
        self,
        symbol: str,
        start_dt: pd.Timestamp,
        end_dt: pd.Timestamp,
    ) -> pd.DataFrame:
        """Return all cached snapshots for *symbol* in *[start_dt, end_dt]*.

        Only cached parquet files are consulted — no live data is fetched.

        Args:
            symbol: Trading pair.
            start_dt: Inclusive start of the window (timezone-aware or UTC).
            end_dt: Inclusive end of the window.

        Returns:
            DataFrame with a DatetimeIndex and columns like
            ``bid_0_price``, ``bid_0_vol``, ..., ``ask_N_price``, ``ask_N_vol``.
            Empty DataFrame if no cached data exists for the window.
        """
        if start_dt.tz is None:
            start_dt = start_dt.tz_localize("UTC")
        else:
            start_dt = start_dt.tz_convert("UTC")
        if end_dt.tz is None:
            end_dt = end_dt.tz_localize("UTC")
        else:
            end_dt = end_dt.tz_convert("UTC")

        dates = pd.date_range(start_dt.date(), end_dt.date(), freq="D")
        frames: list[pd.DataFrame] = []

        for dt in dates:
            path = _ob_cache_path(symbol, dt.strftime("%Y-%m-%d"))
            if path.exists():
                try:
                    df = pd.read_parquet(path)
                    frames.append(df)
                except Exception:
                    logger.debug("Corrupt OB cache file: %s", path)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames).sort_index()
        combined = combined.loc[
            (combined.index >= start_dt) & (combined.index <= end_dt)
        ]
        return combined
