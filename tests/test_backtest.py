"""Tests for the strategy backtest module."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.strategy.backtest import compute_metrics, plot_equity_curve, run_backtest


class TestRunBacktest:
    """Tests for :func:`src.strategy.backtest.run_backtest`."""

    def test_empty_predictions_raises(self) -> None:
        """run_backtest raises ValueError when predictions dict is empty."""
        with pytest.raises(ValueError, match="predictions dict is empty"):
            run_backtest({}, np.array([]), {})

    def test_backtest_on_simple_data(
        self, mock_config: None,
    ) -> None:
        """run_backtest returns expected structure on a minimal dataset with
        a known signal."""
        dates = pd.date_range("2024-01-01", periods=10, freq="5min")
        # All OHLCV data at roughly the same price.
        ohlcv = pd.DataFrame(
            {
                "open": [100.0] * 10,
                "high": [101.0] * 10,
                "low": [99.0] * 10,
                "close": [100.0] * 10,
                "volume": [1000.0] * 10,
            },
            index=dates,
        )

        predictions = {
            "BTC/USDT": np.array([0.8, 0.8, 0.8, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),
        }
        y_test = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        ohlcv_test = {"BTC/USDT": ohlcv}

        result = run_backtest(
            predictions, y_test, ohlcv_test, initial_capital=10000.0,
        )

        # Result should contain the expected keys.
        assert "metrics" in result
        assert "equity_curve" in result
        assert "trades" in result

        # Equity curve must have the same number of steps as the common index.
        # There are 10 common index values, so 10 equity curve entries.
        assert len(result["equity_curve"]) == 10

        # Metrics dict should have all keys.
        expected_metric_keys = {
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
            "profit_factor",
            "total_trades",
        }
        assert expected_metric_keys.issubset(result["metrics"].keys())

    def test_backtest_multi_symbol(
        self, mock_config: None,
    ) -> None:
        """run_backtest works with multiple symbols sharing a common index."""
        dates = pd.date_range("2024-01-01", periods=6, freq="5min")
        base_ohlcv = pd.DataFrame(
            {
                "open": [100.0] * 6,
                "high": [101.0] * 6,
                "low": [99.0] * 6,
                "close": [100.0] * 6,
                "volume": [1000.0] * 6,
            },
            index=dates,
        )

        predictions = {
            "BTC/USDT": np.array([0.8, 0.3, 0.8, 0.3, 0.5, 0.5]),
            "ETH/USDT": np.array([0.3, 0.8, 0.3, 0.8, 0.5, 0.5]),
        }
        y_test = np.array([1, 0, 1, 0, 1, 0])
        ohlcv_test = {
            "BTC/USDT": base_ohlcv.copy(),
            "ETH/USDT": base_ohlcv.copy(),
        }

        result = run_backtest(
            predictions, y_test, ohlcv_test, initial_capital=10000.0,
        )
        assert len(result["equity_curve"]) == 6
        assert isinstance(result["metrics"]["total_trades"], int)

    @pytest.mark.filterwarnings("ignore:invalid value")
    def test_very_short_data_raises(
        self, mock_config: None,
    ) -> None:
        """Fewer than 2 common time steps raises."""
        dates = pd.date_range("2024-01-01", periods=1, freq="5min")
        ohlcv = pd.DataFrame(
            {
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
            },
            index=dates,
        )
        predictions = {"BTC/USDT": np.array([0.5])}
        y_test = np.array([0])
        ohlcv_test = {"BTC/USDT": ohlcv}

        with pytest.raises(ValueError, match="at least 2"):
            run_backtest(predictions, y_test, ohlcv_test)


class TestComputeMetrics:
    """Tests for :func:`src.strategy.backtest.compute_metrics`."""

    def test_zero_return_on_flat_equity(self) -> None:
        """A flat equity curve produces zero return and zero drawdown."""
        equity = np.ones(100) * 10000.0
        metrics = compute_metrics(equity, [], 10000.0)
        assert metrics["total_return"] == 0.0
        assert metrics["max_drawdown"] == 0.0
        assert metrics["total_trades"] == 0
        assert metrics["win_rate"] == 0.0
        # Sharpe should be 0 (no volatility, no return).
        assert metrics["sharpe_ratio"] == 0.0

    def test_all_winning_trades(self) -> None:
        """All-winning trades produce 100% win rate."""
        trades = [
            {
                "symbol": "BTC/USDT",
                "direction": "long",
                "entry_price": 100.0,
                "exit_price": 110.0,
                "size": 10.0,
                "pnl": 1.0,
                "pnl_pct": 10.0,
            },
            {
                "symbol": "BTC/USDT",
                "direction": "long",
                "entry_price": 100.0,
                "exit_price": 105.0,
                "size": 10.0,
                "pnl": 0.5,
                "pnl_pct": 5.0,
            },
        ]
        equity = np.linspace(10000, 11000, 5)
        metrics = compute_metrics(equity, trades, 10000.0)
        assert metrics["win_rate"] == 100.0
        assert metrics["total_trades"] == 2
        assert metrics["total_return"] > 0

    def test_mixed_trades_profit_factor(self) -> None:
        """Profit factor is correctly computed for mixed trades."""
        trades = [
            {"pnl": 100.0, "pnl_pct": 1.0},
            {"pnl": -50.0, "pnl_pct": -0.5},
            {"pnl": 200.0, "pnl_pct": 2.0},
        ]
        equity = np.array([10000.0, 10100.0, 10050.0, 10250.0])
        metrics = compute_metrics(equity, trades, 10000.0)
        assert metrics["profit_factor"] == pytest.approx(300.0 / 50.0)
        assert metrics["total_trades"] == 3
        assert metrics["win_rate"] == pytest.approx(2 / 3 * 100, rel=1e-4)

    def test_infinite_profit_factor_when_no_losses(self) -> None:
        """No losing trades gives infinite profit factor."""
        trades = [{"pnl": 100.0, "pnl_pct": 1.0}]
        equity = np.array([10000.0, 10100.0])
        metrics = compute_metrics(equity, trades, 10000.0)
        assert metrics["profit_factor"] == float("inf")

    def test_sharpe_with_no_volatility(self) -> None:
        """Sharpe ratio is 0 when there is no return variation."""
        # Flat equity curve that doesn't move at all.
        equity = np.ones(50) * 10000.0
        equity[0] = 10000.0
        metrics = compute_metrics(equity, [], 10000.0)
        assert metrics["sharpe_ratio"] == 0.0


class TestPlotEquityCurve:
    """Tests for :func:`src.strategy.backtest.plot_equity_curve`."""

    @pytest.fixture(autouse=True)
    def _use_agg_backend(self) -> None:
        """Use non-interactive Agg backend to prevent plt.show() from
        blocking."""
        import matplotlib
        matplotlib.use("Agg")

    def test_plot_saves_to_path(self, tmp_path) -> None:
        """plot_equity_curve saves a PNG file when ``save_path`` is given."""
        equity = np.linspace(10000.0, 10500.0, 50)
        metrics = {
            "total_return": 5.0,
            "sharpe_ratio": 1.5,
            "max_drawdown": -2.0,
            "win_rate": 60.0,
            "profit_factor": 2.0,
            "total_trades": 10,
        }
        save_path = str(tmp_path / "equity_curve.png")
        plot_equity_curve(equity, metrics, save_path=save_path)

        # Verify the file was created.
        import os
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0

    def test_plot_does_not_raise_with_empty_data(self) -> None:
        """plot_equity_curve handles a single-point equity curve without
        raising."""
        equity = np.array([10000.0])
        metrics = {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_trades": 0,
        }
        # Should not raise.
        plot_equity_curve(equity, metrics)
