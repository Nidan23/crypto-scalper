"""CCXT-based OHLCV data fetcher with local parquet caching."""

import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional

import ccxt
import pandas as pd

from src.config import config


def _init_exchange(exchange_id: str) -> ccxt.Exchange:
    """Initialize a CCXT exchange instance by name.

    Args:
        exchange_id: Lowercase exchange identifier (e.g. 'binance', 'coinbase').

    Returns:
        A configured ccxt Exchange instance.

    Raises:
        ValueError: If the exchange_id does not correspond to a known CCXT exchange.
    """
    if not hasattr(ccxt, exchange_id):
        raise ValueError(
            f"Unknown exchange '{exchange_id}'. "
            f"Available exchanges: {len(ccxt.exchanges)} loaded."
        )
    exchange_class = getattr(ccxt, exchange_id)
    return exchange_class({"enableRateLimit": False})


def _cache_path(exchange_id: str, symbol: str, timeframe: str, limit: int) -> Path:
    """Generate a deterministic cache file path for a given request.

    Cache key is an MD5 hash of ``exchange_id:symbol:timeframe:limit``.

    Args:
        exchange_id: Exchange identifier.
        symbol: Trading pair symbol.
        timeframe: OHLCV candle interval.
        limit: Number of candles.

    Returns:
        Absolute Path to the parquet cache file.
    """
    cache_dir = Path(config.data_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{exchange_id}:{symbol}:{timeframe}:{limit}"
    digest = hashlib.md5(key.encode()).hexdigest()
    return cache_dir / f"{digest}.parquet"


def fetch_ohlcv(
    symbol: str,
    exchange_id: Optional[str] = None,
    timeframe: Optional[str] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch OHLCV candles for a trading pair from a CCXT exchange.

    Results are cached as parquet files under ``config.data_dir`` to avoid
    re-fetching identical requests.  Cache invalidation is manual (delete the
    cache directory or individual parquet files).

    Retry behaviour: up to 3 attempts on network-level errors with
    exponential backoff (1 s, 2 s, 4 s).  A 1-second pause is always applied
    before each exchange call for basic rate limiting.

    Args:
        symbol: Trading pair, e.g. ``"BTC/USDT"``, ``"ETH/GBP"``, ``"SOL/EUR"``.
        exchange_id: CCXT exchange name.  Defaults to ``config.exchange_id``.
        timeframe: Candle interval.  Defaults to ``config.timeframe``.
        limit: Number of candles to fetch.  Defaults to ``config.lookback_candles``.

    Returns:
        DataFrame with a datetime index (``timestamp``) and columns
        ``['open', 'high', 'low', 'close', 'volume']``.

    Raises:
        ValueError: If the symbol is not listed on the exchange.
        RuntimeError: If all retry attempts fail.
    """
    exchange_id = exchange_id or config.exchange_id
    timeframe = timeframe or config.timeframe
    limit = limit or config.lookback_candles

    # ------------------------------------------------------------------
    # Cache check
    # ------------------------------------------------------------------
    cache = _cache_path(exchange_id, symbol, timeframe, limit)
    if cache.exists():
        try:
            df: pd.DataFrame = pd.read_parquet(cache)
            return df
        except Exception:
            # Corrupt cache file; fall through to re-fetch.
            pass

    # ------------------------------------------------------------------
    # Exchange initialisation
    # ------------------------------------------------------------------
    exchange = _init_exchange(exchange_id)
    exchange.load_markets()

    if symbol not in exchange.markets:
        raise ValueError(
            f"Symbol '{symbol}' is not listed on {exchange_id}. "
            f"Available symbols: {len(exchange.markets)} loaded."
        )

    # ------------------------------------------------------------------
    # Fetch with retry
    # ------------------------------------------------------------------
    max_retries = 3
    last_exc: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            time.sleep(1.0)  # basic rate limiting

            ohlcv = exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, limit=limit
            )

            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            # Cache (non-fatal on failure)
            try:
                df.to_parquet(cache)
            except Exception:
                pass

            return df

        except ccxt.NetworkError as e:
            last_exc = e
            if attempt < max_retries - 1:
                backoff = 2.0 ** attempt  # 1, 2, 4 seconds
                time.sleep(backoff)

    raise RuntimeError(
        f"Failed to fetch OHLCV for '{symbol}' on {exchange_id} "
        f"after {max_retries} retries: {last_exc}"
    )


def fetch_multiple(
    symbols: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV data for several symbols.

    Each symbol is fetched sequentially (respecting the rate-limit pause in
    :func:`fetch_ohlcv`).

    Args:
        symbols: List of trading pairs.  Defaults to ``config.symbols``.
        **kwargs: Additional keyword arguments forwarded to :func:`fetch_ohlcv`
            (e.g. ``exchange_id``, ``timeframe``, ``limit``).

    Returns:
        Dict mapping each symbol string to its OHLCV DataFrame.
    """
    if symbols is None:
        symbols = config.symbols

    result: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        result[symbol] = fetch_ohlcv(symbol, **kwargs)

    return result
