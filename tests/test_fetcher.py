"""Tests for src.data.fetcher."""

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.config import config
from src.data.fetcher import _cache_path, _init_exchange, fetch_multiple, fetch_ohlcv


# ---------------------------------------------------------------------------
# _init_exchange
# ---------------------------------------------------------------------------

class TestInitExchange:
    def test_known_exchange(self):
        """Known exchange id returns a ccxt Exchange instance."""
        exchange = _init_exchange("binance")
        assert exchange is not None
        assert exchange.id == "binance"

    def test_unknown_exchange_raises(self):
        """Unknown exchange id raises ValueError."""
        with pytest.raises(ValueError, match="Unknown exchange"):
            _init_exchange("nonexistent_exchange_12345")


# ---------------------------------------------------------------------------
# _cache_path
# ---------------------------------------------------------------------------

class TestCachePath:
    def test_returns_parquet_path(self):
        """Cache path ends in .parquet and lives under config.data_dir."""
        path = _cache_path("binance", "BTC/USDT", "5m", 500)
        assert isinstance(path, Path)
        assert path.suffix == ".parquet"
        assert str(config.data_dir) in str(path)

    def test_deterministic_hash(self):
        """Same inputs produce the same path."""
        p1 = _cache_path("binance", "BTC/USDT", "5m", 500)
        p2 = _cache_path("binance", "BTC/USDT", "5m", 500)
        assert p1 == p2

    def test_different_inputs_different_paths(self):
        """Different inputs produce different paths."""
        p1 = _cache_path("binance", "BTC/USDT", "5m", 500)
        p2 = _cache_path("coinbase", "BTC/USDT", "5m", 500)
        assert p1 != p2

    def test_creates_cache_directory(self, tmp_path):
        """Cache directory is created if it does not exist."""
        test_dir = tmp_path / "custom_cache"
        with patch.object(config, "data_dir", str(test_dir)):
            path = _cache_path("binance", "BTC/USDT", "5m", 500)
            assert test_dir.exists()
            assert path.parent == test_dir.resolve()


# ---------------------------------------------------------------------------
# fetch_ohlcv
# ---------------------------------------------------------------------------

def _make_mock_ohlcv(n_candles=100, start_price=50000.0, volatility=0.01):
    """Generate a realistic-looking OHLCV list as returned by ccxt."""
    np.random.seed(42)
    records = []
    price = start_price
    base_ts = 1700000000000  # milliseconds
    for i in range(n_candles):
        ret = np.random.normal(0, volatility)
        close = price * (1 + ret)
        high = max(price, close) * (1 + abs(np.random.normal(0, volatility * 0.5)))
        low = min(price, close) * (1 - abs(np.random.normal(0, volatility * 0.5)))
        open_ = price
        volume = np.random.uniform(100, 1000)
        records.append([base_ts + i * 300_000, open_, high, low, close, volume])
        price = close
    return records


class TestFetchOHLCV:
    def test_cached_return(self, tmp_path):
        """When cache exists and is valid, return cached data without fetching."""
        cache_dir = tmp_path / "data_cache"
        cache_dir.mkdir(parents=True)
        key = hashlib.md5(b"binance:BTC/USDT:1m:100").hexdigest()
        cache_file = cache_dir / f"{key}.parquet"

        expected_df = pd.DataFrame(
            {"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [100]},
            index=pd.DatetimeIndex([pd.Timestamp("2023-01-01")], name="timestamp"),
        )
        expected_df.to_parquet(cache_file)

        with patch.object(config, "data_dir", str(cache_dir)):
            with patch("src.data.fetcher._init_exchange") as mock_init:
                result = fetch_ohlcv("BTC/USDT", limit=100)

        pd.testing.assert_frame_equal(result, expected_df)
        # Exchange should NOT be initialised (cache hit)
        mock_init.assert_not_called()

    @patch("src.data.fetcher._init_exchange")
    def test_corrupted_cache_re_fetches(self, mock_init_exchange, tmp_path):
        """Corrupt cache file triggers a re-fetch."""
        cache_dir = tmp_path / "data_cache"
        cache_dir.mkdir(parents=True)
        key = hashlib.md5(b"binance:BTC/USDT:1m:100").hexdigest()
        cache_file = cache_dir / f"{key}.parquet"
        cache_file.write_text("not a parquet file")

        mock_exchange = MagicMock()
        mock_exchange.markets = {"BTC/USDT": {}}
        ohlcv_data = _make_mock_ohlcv(n_candles=100)
        mock_exchange.fetch_ohlcv.return_value = ohlcv_data
        mock_init_exchange.return_value = mock_exchange

        with patch.object(config, "data_dir", str(cache_dir)):
            result = fetch_ohlcv("BTC/USDT", limit=100)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]
        assert len(result) == 100
        mock_exchange.fetch_ohlcv.assert_called_once()

    @patch("src.data.fetcher.time.sleep", return_value=None)
    @patch("src.data.fetcher._init_exchange")
    def test_successful_fetch(self, mock_init_exchange, mock_sleep, tmp_path):
        """Successful fetch returns a correctly formatted DataFrame."""
        mock_exchange = MagicMock()
        mock_exchange.markets = {"BTC/USDT": {}}
        ohlcv_data = _make_mock_ohlcv(n_candles=50)
        mock_exchange.fetch_ohlcv.return_value = ohlcv_data
        mock_init_exchange.return_value = mock_exchange

        cache_dir = tmp_path / "data_cache"
        with patch.object(config, "data_dir", str(cache_dir)):
            result = fetch_ohlcv("BTC/USDT", exchange_id="binance", timeframe="5m", limit=50)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]
        assert len(result) == 50
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.name == "timestamp"

    @patch("src.data.fetcher.time.sleep", return_value=None)
    @patch("src.data.fetcher._init_exchange")
    def test_invalid_symbol_raises(self, mock_init_exchange, mock_sleep):
        """Fetching an unlisted symbol raises ValueError."""
        mock_exchange = MagicMock()
        mock_exchange.markets = {"BTC/USDT": {}}  # does NOT include SOL/BAD
        mock_init_exchange.return_value = mock_exchange

        with pytest.raises(ValueError, match="not listed"):
            fetch_ohlcv("SOL/BAD", exchange_id="binance", limit=10)

    @patch("src.data.fetcher.time.sleep", return_value=None)
    @patch("src.data.fetcher._init_exchange")
    def test_retry_on_network_error_then_succeeds(self, mock_init_exchange, mock_sleep, tmp_path):
        """Fetch retries on NetworkError and succeeds on the second attempt."""
        from ccxt import NetworkError

        mock_exchange = MagicMock()
        mock_exchange.markets = {"BTC/USDT": {}}
        ohlcv_data = _make_mock_ohlcv(n_candles=30)

        mock_exchange.fetch_ohlcv.side_effect = [NetworkError("timeout"), ohlcv_data]
        mock_init_exchange.return_value = mock_exchange

        cache_dir = tmp_path / "data_cache"
        with patch.object(config, "data_dir", str(cache_dir)):
            result = fetch_ohlcv("BTC/USDT", exchange_id="binance", limit=30)

        assert len(result) == 30
        assert mock_exchange.fetch_ohlcv.call_count == 2

    @patch("src.data.fetcher.time.sleep", return_value=None)
    @patch("src.data.fetcher._init_exchange")
    def test_all_retries_exhausted_raises(self, mock_init_exchange, mock_sleep):
        """Fetch raises RuntimeError after all retries are exhausted."""
        from ccxt import NetworkError

        mock_exchange = MagicMock()
        mock_exchange.markets = {"BTC/USDT": {}}
        mock_exchange.fetch_ohlcv.side_effect = NetworkError("down")
        mock_init_exchange.return_value = mock_exchange

        with pytest.raises(RuntimeError, match="Failed to fetch"):
            fetch_ohlcv("BTC/USDT", exchange_id="binance", limit=10)

        assert mock_exchange.fetch_ohlcv.call_count == 3

    @patch("src.data.fetcher.time.sleep", return_value=None)
    @patch("src.data.fetcher._init_exchange")
    def test_uses_defaults_from_config(self, mock_init_exchange, mock_sleep, tmp_path):
        """When optional args are omitted, config defaults are used."""
        mock_exchange = MagicMock()
        mock_exchange.markets = {"BTC/USDT": {}}
        ohlcv_data = _make_mock_ohlcv(n_candles=config.lookback_candles)
        mock_exchange.fetch_ohlcv.return_value = ohlcv_data
        mock_init_exchange.return_value = mock_exchange

        cache_dir = tmp_path / "data_cache"
        with patch.object(config, "data_dir", str(cache_dir)):
            result = fetch_ohlcv("BTC/USDT")

        assert len(result) == config.lookback_candles
        mock_exchange.fetch_ohlcv.assert_called_once_with(
            "BTC/USDT", timeframe=config.timeframe, limit=config.lookback_candles
        )


# ---------------------------------------------------------------------------
# fetch_multiple
# ---------------------------------------------------------------------------

class TestFetchMultiple:
    @patch("src.data.fetcher.fetch_ohlcv")
    def test_fetches_all_symbols(self, mock_fetch):
        """fetch_multiple calls fetch_ohlcv for each symbol and returns a dict."""
        mock_fetch.side_effect = lambda s, **kw: pd.DataFrame(
            {"open": [s]}, index=pd.DatetimeIndex([pd.Timestamp("2023-01-01")])
        )
        symbols = ["BTC/USDT", "ETH/USDT"]
        result = fetch_multiple(symbols)
        assert set(result.keys()) == set(symbols)
        assert mock_fetch.call_count == 2

    @patch("src.data.fetcher.fetch_ohlcv")
    def test_defaults_to_config_symbols(self, mock_fetch):
        """When symbols is None, default to config.symbols."""
        mock_fetch.return_value = pd.DataFrame(
            {"open": [1.0]}, index=pd.DatetimeIndex([pd.Timestamp("2023-01-01")])
        )
        result = fetch_multiple()
        assert set(result.keys()) == set(config.symbols)
        assert mock_fetch.call_count == len(config.symbols)

    @patch("src.data.fetcher.fetch_ohlcv")
    def test_passes_kwargs(self, mock_fetch):
        """Extra kwargs are forwarded to fetch_ohlcv."""
        mock_fetch.return_value = pd.DataFrame(
            {"open": [1.0]}, index=pd.DatetimeIndex([pd.Timestamp("2023-01-01")])
        )
        fetch_multiple(["BTC/USDT"], exchange_id="coinbase", limit=100)
        mock_fetch.assert_called_with("BTC/USDT", exchange_id="coinbase", limit=100)

    @patch("src.data.fetcher.fetch_ohlcv")
    def test_handles_empty_symbols_list(self, mock_fetch):
        """Empty symbols list returns an empty dict."""
        result = fetch_multiple([])
        assert result == {}
        mock_fetch.assert_not_called()
