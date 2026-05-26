"""Tests for src.data.orderbook."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.config import config
from src.data.orderbook import (
    OrderBookFetcher,
    _ob_cache_dir,
    _ob_cache_path,
    _snapshot_to_rows,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_order_book(depth: int = 20, base_price: float = 65000.0):
    """Generate synthetic CCXT-style order book (bids + asks)."""
    bids = []
    asks = []
    for i in range(depth):
        bids.append([base_price - (i + 1) * 10.0, 10.0 + i * 2.0])
        asks.append([base_price + (i + 1) * 10.0, 10.0 + i * 2.0])
    return {"bids": bids, "asks": asks}


# ---------------------------------------------------------------------------
# _cache_path
# ---------------------------------------------------------------------------

class TestCachePath:
    def test_returns_parquet_path(self):
        path = _ob_cache_path("BTC/USDT", "2024-06-15")
        assert isinstance(path, Path)
        assert path.suffix == ".parquet"

    def test_deterministic(self):
        p1 = _ob_cache_path("BTC/USDT", "2024-06-15")
        p2 = _ob_cache_path("BTC/USDT", "2024-06-15")
        assert p1 == p2

    def test_different_symbol_different_path(self):
        p1 = _ob_cache_path("BTC/USDT", "2024-06-15")
        p2 = _ob_cache_path("ETH/USDT", "2024-06-15")
        assert p1 != p2

    def test_creates_cache_directory(self, tmp_path: Path):
        with patch.object(config, "orderbook_cache_dir", str(tmp_path / "ob")):
            _ob_cache_path("BTC/USDT", "2024-06-15")
            assert (tmp_path / "ob").exists()


# ---------------------------------------------------------------------------
# _snapshot_to_rows
# ---------------------------------------------------------------------------

class TestSnapshotToRows:
    def test_basic_structure(self):
        ts = pd.Timestamp("2024-06-15T12:00:00")
        bids = [[65000.0, 1.5], [64990.0, 2.0]]
        asks = [[65010.0, 1.0], [65020.0, 3.0]]
        row = _snapshot_to_rows(ts, bids, asks, depth=2)

        assert row["timestamp"] == ts
        assert row["bid_0_price"] == 65000.0
        assert row["bid_0_vol"] == 1.5
        assert row["bid_1_price"] == 64990.0
        assert row["bid_1_vol"] == 2.0
        assert row["ask_0_price"] == 65010.0
        assert row["ask_0_vol"] == 1.0
        assert row["ask_1_price"] == 65020.0
        assert row["ask_1_vol"] == 3.0

    def test_fewer_levels_than_depth(self):
        ts = pd.Timestamp("2024-06-15T12:00:00")
        bids = [[65000.0, 1.0]]
        asks = [[65010.0, 0.5]]
        row = _snapshot_to_rows(ts, bids, asks, depth=5)
        # Should not raise; only available columns present.
        assert "bid_0_price" in row
        assert "bid_4_price" not in row


# ---------------------------------------------------------------------------
# OrderBookFetcher
# ---------------------------------------------------------------------------

class TestOrderBookFetcher:
    @patch("src.data.orderbook.ccxt")
    def test_fetch_snapshot_caches(self, mock_ccxt, tmp_path: Path):
        """fetch_snapshot caches a snapshot to a parquet file."""
        mock_exchange = MagicMock()
        mock_exchange.markets = {"BTC/USDT": {}}
        mock_exchange.fetch_order_book.return_value = _make_mock_order_book(depth=5)
        mock_ccxt.binance = MagicMock(return_value=mock_exchange)

        cache_path = tmp_path / "ob_cache"
        with patch.object(config, "orderbook_cache_dir", str(cache_path)):
            fetcher = OrderBookFetcher(exchange_id="binance", depth=5)
            fetcher._exchange = mock_exchange  # skip lazy init
            result = fetcher.fetch_snapshot("BTC/USDT")

        assert result is not None
        assert "bid_0_price" in result
        assert "ask_0_price" in result
        assert cache_path.exists()
        # Should have at least one parquet file.
        parquets = list(cache_path.glob("*.parquet"))
        assert len(parquets) > 0

    @patch("src.data.orderbook.ccxt")
    def test_fetch_snapshot_handles_error(self, mock_ccxt, tmp_path: Path):
        """fetch_snapshot returns None on exchange error."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_order_book.side_effect = RuntimeError("fail")
        mock_ccxt.binance = MagicMock(return_value=mock_exchange)

        cache_path = tmp_path / "ob_cache"
        with patch.object(config, "orderbook_cache_dir", str(cache_path)):
            fetcher = OrderBookFetcher(exchange_id="binance", depth=5)
            fetcher._exchange = mock_exchange
            result = fetcher.fetch_snapshot("BTC/USDT")

        assert result is None

    def test_get_snapshots_empty(self, tmp_path: Path):
        """get_snapshots returns empty DataFrame when no cache exists."""
        cache_path = tmp_path / "ob_empty"
        cache_path.mkdir(parents=True, exist_ok=True)
        with patch.object(config, "orderbook_cache_dir", str(cache_path)):
            fetcher = OrderBookFetcher(exchange_id="binance", depth=5)
            result = fetcher.get_snapshots(
                "BTC/USDT",
                pd.Timestamp("2024-06-01", tz="UTC"),
                pd.Timestamp("2024-06-02", tz="UTC"),
            )
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_get_snapshots_returns_cached(self, tmp_path: Path):
        """get_snapshots returns cached data within the time window."""
        cache_path = tmp_path / "ob_cache"
        cache_path.mkdir(parents=True, exist_ok=True)

        # Pre-populate a cache file.
        ts1 = pd.Timestamp("2024-06-15T12:00:00", tz="UTC")
        ts2 = pd.Timestamp("2024-06-15T12:00:10", tz="UTC")
        df = pd.DataFrame(
            {
                "bid_0_price": [65000.0, 65001.0],
                "bid_0_vol": [1.0, 1.5],
                "ask_0_price": [65010.0, 65011.0],
                "ask_0_vol": [2.0, 2.5],
            },
            index=pd.DatetimeIndex([ts1, ts2], name="timestamp"),
        )

        with patch.object(config, "orderbook_cache_dir", str(cache_path)):
            cache_file = _ob_cache_path("BTC/USDT", "2024-06-15")
            df.to_parquet(cache_file)

            fetcher = OrderBookFetcher(exchange_id="binance", depth=1)
            result = fetcher.get_snapshots(
                "BTC/USDT",
                pd.Timestamp("2024-06-15T11:00:00"),
                pd.Timestamp("2024-06-15T13:00:00"),
            )

        assert len(result) == 2
        assert result.iloc[0]["bid_0_price"] == 65000.0

    def test_get_snapshots_filters_time_window(self, tmp_path: Path):
        """get_snapshots only returns rows within [start_dt, end_dt]."""
        cache_path = tmp_path / "ob_cache"
        cache_path.mkdir(parents=True, exist_ok=True)

        ts = pd.date_range(
            "2024-06-15T12:00:00", periods=20, freq="10s", tz="UTC"
        )
        df = pd.DataFrame(
            {
                "bid_0_price": np.full(20, 65000.0),
                "bid_0_vol": np.ones(20),
                "ask_0_price": np.full(20, 65010.0),
                "ask_0_vol": np.ones(20) * 2,
            },
            index=pd.DatetimeIndex(ts, name="timestamp"),
        )

        with patch.object(config, "orderbook_cache_dir", str(cache_path)):
            cache_file = _ob_cache_path("BTC/USDT", "2024-06-15")
            df.to_parquet(cache_file)

            fetcher = OrderBookFetcher(exchange_id="binance", depth=1)
            result = fetcher.get_snapshots(
                "BTC/USDT",
                pd.Timestamp("2024-06-15T12:00:30"),
                pd.Timestamp("2024-06-15T12:01:00"),
            )

        # Only snapshots within the window should be returned.
        assert len(result) > 0
        assert result.index.min() >= pd.Timestamp("2024-06-15T12:00:30", tz="UTC")
        assert result.index.max() <= pd.Timestamp("2024-06-15T12:01:00", tz="UTC")

    def test_unknown_exchange_raises(self):
        """Unknown exchange raises ValueError on lazy init."""
        fetcher = OrderBookFetcher(exchange_id="nonexistent_123", depth=5)
        with patch("src.data.orderbook.hasattr", return_value=False):
            with pytest.raises(ValueError, match="Unknown exchange"):
                _ = fetcher.exchange
