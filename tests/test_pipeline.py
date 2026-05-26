"""Tests for src.data.pipeline."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.config import config
from src.data.pipeline import run_pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_synthetic_ohlcv(n: int, start_price: float = 50000.0, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame for testing."""
    rng = np.random.default_rng(seed)
    prices = start_price + np.cumsum(rng.normal(0, 50, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", name="timestamp")
    return pd.DataFrame(
        {
            "open": prices * (1 + rng.normal(0, 0.001, n)),
            "high": prices * (1 + np.abs(rng.normal(0, 0.002, n))),
            "low": prices * (1 - np.abs(rng.normal(0, 0.002, n))),
            "close": prices,
            "volume": rng.uniform(100, 1000, n),
        },
        index=idx,
    )


@pytest.fixture
def mock_fetch_multiple():
    """Patches fetch_multiple to return synthetic data."""
    with patch("src.data.pipeline.fetch_multiple") as mock:
        yield mock


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------

class TestRunPipeline:
    def test_basic_pipeline(self, mock_fetch_multiple):
        """End-to-end pipeline returns all expected keys with correct shapes."""
        n_candles = 500
        df = _make_synthetic_ohlcv(n_candles)
        mock_fetch_multiple.return_value = {"BTC/USDT": df}

        result = run_pipeline(symbols=["BTC/USDT"])

        expected_keys = {
            "X_train", "y_train", "X_val", "y_val", "X_test", "y_test",
            "scaler", "feature_names",
        }
        assert set(result.keys()) == expected_keys

        # Check shapes
        assert result["X_train"].ndim == 3
        assert result["X_train"].shape[1] == config.seq_len  # seq_len
        assert result["X_train"].shape[0] > 0
        assert result["y_train"].shape[0] == result["X_train"].shape[0]

        # All splits should have sequence data
        assert result["X_val"].shape[0] > 0
        assert result["X_test"].shape[0] > 0

        # Feature names should be a list of strings
        assert isinstance(result["feature_names"], list)
        assert len(result["feature_names"]) > 0
        assert all(isinstance(n, str) for n in result["feature_names"])

        # Scaler should be fitted
        assert hasattr(result["scaler"], "mean_")
        assert len(result["scaler"].mean_) == len(result["feature_names"])

    def test_split_ratios(self, mock_fetch_multiple):
        """Splits roughly respect train/val/test ratios.

        Each split loses ``seq_len`` samples to sequence creation, so the
        post-sequence ratio is shifted from the raw row-split ratio.
        Tolerance is widened accordingly.
        """
        n_candles = 1500  # more data = smaller proportional overhead
        df = _make_synthetic_ohlcv(n_candles, seed=1)
        mock_fetch_multiple.return_value = {"BTC/USDT": df}

        result = run_pipeline(["BTC/USDT"])

        n_train = result["X_train"].shape[0]
        n_val = result["X_val"].shape[0]
        n_test = result["X_test"].shape[0]
        total = n_train + n_val + n_test

        assert abs(n_train / total - config.train_split) < 0.08
        assert abs(n_val / total - config.val_split) < 0.08

    def test_multiple_symbols(self, mock_fetch_multiple):
        """Pipeline handles multiple symbols."""
        df_btc = _make_synthetic_ohlcv(500, start_price=50000, seed=1)
        df_eth = _make_synthetic_ohlcv(500, start_price=3000, seed=2)
        mock_fetch_multiple.return_value = {
            "BTC/USDT": df_btc,
            "ETH/USDT": df_eth,
        }

        result = run_pipeline(symbols=["BTC/USDT", "ETH/USDT"])
        assert result["X_train"].shape[0] > 0
        assert result["X_val"].shape[0] > 0
        assert result["X_test"].shape[0] > 0

    def test_single_symbol_works(self, mock_fetch_multiple):
        """Pipeline works with a single symbol (not in a list)."""
        df = _make_synthetic_ohlcv(500)
        mock_fetch_multiple.return_value = {"BTC/USDT": df}

        result = run_pipeline(symbols=["BTC/USDT"])
        assert result["X_train"].shape[0] > 0

    def test_insufficient_data_raises(self, mock_fetch_multiple):
        """Too few candles after preprocessing raises RuntimeError."""
        n_candles = config.seq_len + 5  # not enough after NaN dropping + seq creation
        df = _make_synthetic_ohlcv(n_candles, seed=3)
        mock_fetch_multiple.return_value = {"BTC/USDT": df}

        with pytest.raises(RuntimeError, match="at least"):
            run_pipeline(symbols=["BTC/USDT"])

    def test_empty_data_raises(self, mock_fetch_multiple):
        """Empty DataFrame from fetch_multiple raises RuntimeError."""
        empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        mock_fetch_multiple.return_value = {"BTC/USDT": empty_df}

        with pytest.raises(RuntimeError, match="No OHLCV data"):
            run_pipeline(symbols=["BTC/USDT"])

    def test_no_symbols_provided(self, mock_fetch_multiple):
        """When symbols is None, uses config.symbols."""
        df = _make_synthetic_ohlcv(500)
        # Return data for all config symbols
        mock_fetch_multiple.return_value = {s: df for s in config.symbols}

        result = run_pipeline()
        assert result["X_train"].shape[0] > 0
        # fetch_multiple was called with no explicit symbols arg
        # (run_pipeline passes None, fetch_multiple defaults to config.symbols)
        assert mock_fetch_multiple.call_count == 1

    def test_target_values_are_binary(self, mock_fetch_multiple):
        """All target values are 0 or 1."""
        df = _make_synthetic_ohlcv(500, seed=4)
        mock_fetch_multiple.return_value = {"BTC/USDT": df}

        result = run_pipeline(["BTC/USDT"])

        for key in ["y_train", "y_val", "y_test"]:
            arr = result[key]
            if arr.size > 0:
                assert np.all((arr == 0) | (arr == 1)), f"{key} has non-binary values"

    def test_feature_names_are_consistent(self, mock_fetch_multiple):
        """Feature names match the columns returned by build_features."""
        df = _make_synthetic_ohlcv(500, seed=5)
        mock_fetch_multiple.return_value = {"BTC/USDT": df}

        result = run_pipeline(["BTC/USDT"])
        n_features = result["X_train"].shape[2]
        assert len(result["feature_names"]) == n_features

    @patch("src.data.pipeline.config")
    def test_respects_custom_config(self, mock_config, mock_fetch_multiple):
        """Pipeline uses config values for split ratios and seq_len."""
        # Set up a mock config with non-default values
        mock_config.exchange_id = "binance"
        mock_config.symbols = ["BTC/USDT"]
        mock_config.timeframe = "5m"
        mock_config.lookback_candles = 500
        mock_config.seq_len = 30
        mock_config.train_split = 0.8
        mock_config.val_split = 0.1
        mock_config.rsi_period = 14
        mock_config.macd_fast = 12
        mock_config.macd_slow = 26
        mock_config.macd_signal = 9
        mock_config.bb_period = 20
        mock_config.bb_std = 2.0

        df = _make_synthetic_ohlcv(1000, seed=6)
        mock_fetch_multiple.return_value = {"BTC/USDT": df}

        # We need to patch at the import level in pipeline too
        with patch("src.data.pipeline.config.seq_len", 30), \
             patch("src.data.pipeline.config.train_split", 0.8), \
             patch("src.data.pipeline.config.val_split", 0.1):
            result = run_pipeline(["BTC/USDT"])

        assert result["X_train"].shape[1] == 30  # custom seq_len
        # Check split ratios (seq_len overhead shifts ratios, so use wide tolerance)
        n_train = result["X_train"].shape[0]
        n_val = result["X_val"].shape[0]
        n_test = result["X_test"].shape[0]
        total = n_train + n_val + n_test
        assert abs(n_train / total - 0.8) < 0.08
        assert abs(n_val / total - 0.1) < 0.05
