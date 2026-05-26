"""Shared fixtures for crypto-scalper tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """Generate 200 rows of synthetic random-walk OHLCV data.

    Seeds with ``42`` for reproducibility.  OHLC integrity is enforced:
    ``high >= max(open, close)`` and ``low <= min(open, close)``.
    """
    n = 200
    rng = np.random.default_rng(42)

    # Random-walk close prices.
    log_returns = rng.normal(0, 0.02, n)
    close = 50000.0 * np.exp(log_returns.cumsum())

    # Open drifts from previous close.
    open_ = close * (1.0 + rng.normal(0, 0.005, n))

    # Ensure high >= max(open, close) and low <= min(open, close).
    bar_max = np.maximum(open_, close)
    bar_min = np.minimum(open_, close)
    high = bar_max * (1.0 + np.abs(rng.normal(0, 0.01, n)))
    low = bar_min * (1.0 - np.abs(rng.normal(0, 0.01, n)))

    volume = rng.integers(1000, 10000, n)

    dates = pd.date_range(start="2024-01-01", periods=n, freq="5min")
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume.astype(float),
        },
        index=dates,
    )


@pytest.fixture
def mock_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set config values to known test defaults.

    Use this fixture to ensure deterministic behaviour when tests depend
    on configuration values.
    """
    monkeypatch.setattr("src.config.config.symbols", ["BTC/USDT", "ETH/USDT"])
    monkeypatch.setattr("src.config.config.timeframe", "5m")
    monkeypatch.setattr("src.config.config.lookback_candles", 500)
    monkeypatch.setattr("src.config.config.seq_len", 5)
    monkeypatch.setattr("src.config.config.train_split", 0.7)
    monkeypatch.setattr("src.config.config.val_split", 0.15)

    # Model — small values for fast tests.
    monkeypatch.setattr("src.config.config.hidden_dim", 16)
    monkeypatch.setattr("src.config.config.num_layers", 1)
    monkeypatch.setattr("src.config.config.dropout", 0.0)
    monkeypatch.setattr("src.config.config.batch_size", 8)
    monkeypatch.setattr("src.config.config.learning_rate", 0.01)
    monkeypatch.setattr("src.config.config.num_epochs", 3)
    monkeypatch.setattr("src.config.config.early_stopping_patience", 2)

    # Trading params.
    monkeypatch.setattr(
        "src.config.config.confidence_threshold_long", 0.55
    )
    monkeypatch.setattr(
        "src.config.config.confidence_threshold_short", 0.45
    )
    monkeypatch.setattr("src.config.config.position_size_pct", 0.02)
    monkeypatch.setattr("src.config.config.stop_loss_atr_mult", 2.0)
    monkeypatch.setattr("src.config.config.take_profit_atr_mult", 3.0)
    monkeypatch.setattr("src.config.config.max_hold_candles", 3)

    # Paths (set in tests that need them, using tmp_path).
    monkeypatch.setattr("src.config.config.model_dir", "/tmp/test_models")
    monkeypatch.setattr("src.config.config.data_dir", "/tmp/test_data")
    monkeypatch.setattr("src.config.config.plot_dir", "/tmp/test_plots")
    monkeypatch.setattr("src.config.config.exchange_id", "binance")
