"""Tests for the strategy signals module."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.strategy.signals import (
    apply_risk_management,
    calculate_atr,
    calculate_position_size,
    generate_signals,
)


class TestGenerateSignals:
    """Tests for :func:`src.strategy.signals.generate_signals`."""

    def test_long_signal_above_threshold(self, mock_config: None) -> None:
        """When prob_up >= 0.55, a LONG signal is produced."""
        predictions = {"BTC/USDT": (0.75, 0.25)}
        ohlcv_dict = {
            "BTC/USDT": pd.DataFrame(
                {"close": [50000.0]},
                index=pd.to_datetime(["2024-01-01 12:00"]),
            )
        }
        signals = generate_signals(predictions, ohlcv_dict)
        assert len(signals) == 1
        assert signals[0]["direction"] == "long"
        assert signals[0]["confidence"] == 0.75
        assert signals[0]["entry_price"] == 50000.0

    def test_short_signal_below_threshold(self, mock_config: None) -> None:
        """When prob_up <= 0.45, a SHORT signal is produced."""
        predictions = {"BTC/USDT": (0.30, 0.70)}
        ohlcv_dict = {
            "BTC/USDT": pd.DataFrame(
                {"close": [49000.0]},
                index=pd.to_datetime(["2024-01-01 12:00"]),
            )
        }
        signals = generate_signals(predictions, ohlcv_dict)
        assert len(signals) == 1
        assert signals[0]["direction"] == "short"
        assert signals[0]["confidence"] == 0.70
        assert signals[0]["entry_price"] == 49000.0

    def test_no_signal_in_middle(self, mock_config: None) -> None:
        """When 0.45 < prob_up < 0.55, no signal is produced."""
        predictions = {"BTC/USDT": (0.50, 0.50)}
        ohlcv_dict = {
            "BTC/USDT": pd.DataFrame(
                {"close": [50000.0]},
                index=pd.to_datetime(["2024-01-01 12:00"]),
            )
        }
        signals = generate_signals(predictions, ohlcv_dict)
        assert len(signals) == 0

    def test_handles_multiple_symbols(self, mock_config: None) -> None:
        """Multiple symbols are each processed independently."""
        predictions = {
            "BTC/USDT": (0.80, 0.20),
            "ETH/USDT": (0.30, 0.70),
            "SOL/USDT": (0.50, 0.50),
        }
        ohlcv_dict = {
            "BTC/USDT": pd.DataFrame(
                {"close": [50000.0]},
                index=pd.to_datetime(["2024-01-01 12:00"]),
            ),
            "ETH/USDT": pd.DataFrame(
                {"close": [3000.0]},
                index=pd.to_datetime(["2024-01-01 12:00"]),
            ),
            "SOL/USDT": pd.DataFrame(
                {"close": [150.0]},
                index=pd.to_datetime(["2024-01-01 12:00"]),
            ),
        }
        signals = generate_signals(predictions, ohlcv_dict)
        assert len(signals) == 2
        assert signals[0]["symbol"] == "BTC/USDT"
        assert signals[0]["direction"] == "long"
        assert signals[1]["symbol"] == "ETH/USDT"
        assert signals[1]["direction"] == "short"

    def test_skips_missing_symbol(self, mock_config: None) -> None:
        """Symbols without OHLCV data are skipped."""
        predictions = {"BTC/USDT": (0.80, 0.20)}
        signals = generate_signals(predictions, {})
        assert len(signals) == 0

    def test_entry_price_is_last_close(self, mock_config: None) -> None:
        """entry_price uses the last close in the OHLCV DataFrame."""
        predictions = {"BTC/USDT": (0.80, 0.20)}
        ohlcv_dict = {
            "BTC/USDT": pd.DataFrame(
                {"close": [49000.0, 49500.0, 50000.0]},
                index=pd.date_range(
                    "2024-01-01 11:50", periods=3, freq="5min"
                ),
            )
        }
        signals = generate_signals(predictions, ohlcv_dict)
        assert signals[0]["entry_price"] == 50000.0
        assert signals[0]["timestamp"] == ohlcv_dict["BTC/USDT"].index[-1]


class TestCalculateATR:
    """Tests for :func:`src.strategy.signals.calculate_atr`."""

    def test_matches_manual_calculation(self) -> None:
        """ATR computed by calculate_atr matches a manual computation on
        a small dataset."""
        df = pd.DataFrame(
            {
                "high": [102.0, 103.0, 101.0, 104.0],
                "low": [98.0, 99.0, 97.0, 100.0],
                "close": [100.0, 101.0, 99.0, 102.0],
            },
            index=pd.date_range("2024-01-01", periods=4, freq="5min"),
        )

        atr = calculate_atr(df, period=3)

        # Manual True Range for each row.
        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [
                (df["high"] - df["low"]).abs(),
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        # Manual EWM (span=3, adjust=False).
        expected_atr = tr.ewm(span=3, adjust=False).mean()

        pd.testing.assert_series_equal(atr, expected_atr)

    def test_returns_series_with_same_index(self) -> None:
        """ATR Series preserves the original DataFrame's index."""
        df = pd.DataFrame(
            {
                "high": np.random.randn(20) + 100,
                "low": np.random.randn(20) + 99,
                "close": np.random.randn(20) + 100,
            },
            index=pd.date_range("2024-01-01", periods=20, freq="5min"),
        )
        atr = calculate_atr(df, period=14)
        assert isinstance(atr, pd.Series)
        assert list(atr.index) == list(df.index)


class TestCalculatePositionSize:
    """Tests for :func:`src.strategy.signals.calculate_position_size`."""

    def test_long_params(self, mock_config: None) -> None:
        """Position sizing for long: stop below entry, take profit above."""
        result = calculate_position_size(
            capital=10000.0, price=50000.0, atr=1000.0, direction="long"
        )
        assert result["size"] == pytest.approx(
            10000 * 0.02
        )
        assert result["stop_loss"] == 50000.0 - 2.0 * 1000.0  # = 48000
        assert result["take_profit"] == 50000.0 + 3.0 * 1000.0  # = 53000

    def test_short_params(self, mock_config: None) -> None:
        """Position sizing for short: stop above entry, take profit below."""
        result = calculate_position_size(
            capital=10000.0, price=50000.0, atr=1000.0, direction="short"
        )
        expected_size = 10000 * 0.02
        assert result["size"] == pytest.approx(expected_size)
        assert result["stop_loss"] == 50000.0 + 2.0 * 1000.0  # = 52000
        assert result["take_profit"] == 50000.0 - 3.0 * 1000.0  # = 47000

    def test_zero_atr(self, mock_config: None) -> None:
        """A zero ATR produces stop == take_profit == entry_price."""
        result = calculate_position_size(
            capital=10000.0, price=50000.0, atr=0.0, direction="long"
        )
        assert result["stop_loss"] == 50000.0
        assert result["take_profit"] == 50000.0

    def test_position_size_scales_with_capital(
        self, mock_config: None,
    ) -> None:
        """Doubling capital doubles position size."""
        r1 = calculate_position_size(10000, 50000, 1000, "long")
        r2 = calculate_position_size(20000, 50000, 1000, "long")
        assert r2["size"] == pytest.approx(2 * r1["size"])


class TestApplyRiskManagement:
    """Tests for :func:`src.strategy.signals.apply_risk_management`."""

    def test_enriches_signals(self, mock_config: None) -> None:
        """apply_risk_management adds size, stop_loss, take_profit to valid
        signals."""
        signals = [
            {
                "symbol": "BTC/USDT",
                "direction": "long",
                "confidence": 0.75,
                "entry_price": 50000.0,
                "timestamp": pd.Timestamp("2024-01-01 12:00"),
            }
        ]
        ohlcv_dict = {
            "BTC/USDT": pd.DataFrame(
                {
                    "high": [50500.0, 51000.0, 50800.0],
                    "low": [49500.0, 49800.0, 50200.0],
                    "close": [50000.0, 50500.0, 50600.0],
                    "open": [50000.0, 50500.0, 50600.0],
                },
                index=pd.date_range(
                    "2024-01-01 11:50", periods=3, freq="5min"
                ),
            )
        }

        enriched = apply_risk_management(signals, capital=10000.0, ohlcv_dict=ohlcv_dict)

        assert len(enriched) == 1
        enriched_signal = enriched[0]
        assert "size" in enriched_signal
        assert "stop_loss" in enriched_signal
        assert "take_profit" in enriched_signal
        assert enriched_signal["size"] > 0
        assert enriched_signal["stop_loss"] < enriched_signal["entry_price"]
        assert enriched_signal["take_profit"] > enriched_signal["entry_price"]

    def test_skips_signal_without_ohlcv(
        self, mock_config: None,
    ) -> None:
        """Signals for symbols without OHLCV data are skipped."""
        signals = [
            {
                "symbol": "BTC/USDT",
                "direction": "long",
                "confidence": 0.75,
                "entry_price": 50000.0,
                "timestamp": pd.Timestamp("2024-01-01 12:00"),
            }
        ]
        enriched = apply_risk_management(signals, 10000.0, {})
        assert len(enriched) == 0
