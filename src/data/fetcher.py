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

    Paginates automatically when *limit* exceeds the exchange's per-request
    cap (typically 1000 for Binance).

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
    # Paginated fetch
    # ------------------------------------------------------------------
    max_retries = 3
    per_request = 1000  # Binance/CCXT max
    all_candles: List[list] = []

    # Start from `limit` minutes ago, fetch forward in 1000-candle chunks.
    start_ms = int(pd.Timestamp.now().timestamp() * 1000) - (limit * 60 * 1000)
    since_ms = start_ms

    while len(all_candles) < limit:
        last_exc: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                time.sleep(0.5)  # rate limit
                batch = exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=since_ms,
                    limit=per_request,
                )
                break
            except ccxt.NetworkError as e:
                last_exc = e
                if attempt < max_retries - 1:
                    time.sleep(2.0 ** attempt)
            except Exception as e:
                last_exc = e
                if attempt < max_retries - 1:
                    time.sleep(2.0 ** attempt)
        else:
            if all_candles:
                break  # partial data is better than nothing
            raise RuntimeError(
                f"Failed to fetch OHLCV for '{symbol}' on {exchange_id} "
                f"after {max_retries} retries: {last_exc}"
            )

        if not batch:
            break  # no more data available

        all_candles.extend(batch)

        if len(batch) < per_request:
            break  # reached present (exchange returned partial page)

        # Advance past the newest candle in this batch.
        since_ms = batch[-1][0] + 1

    # ------------------------------------------------------------------
    # Build DataFrame
    # ------------------------------------------------------------------
    if not all_candles:
        raise RuntimeError(
            f"No OHLCV data returned for '{symbol}' on {exchange_id}."
        )

    df = pd.DataFrame(
        all_candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="timestamp").set_index("timestamp")
    df = df.sort_index()

    # Cache
    try:
        df.to_parquet(cache)
    except Exception:
        pass

    return df


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
