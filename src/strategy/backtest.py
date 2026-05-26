"""Walk-forward backtesting for LSTM-based trading strategies."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import config
from src.strategy.signals import (
    apply_risk_management,
    generate_signals,
)


def run_backtest(
    predictions: Dict[str, np.ndarray],
    y_test: np.ndarray,
    ohlcv_test: Dict[str, pd.DataFrame],
    initial_capital: float = 10000.0,
    regime: str = "off",
) -> dict:
    """Run a walk-forward backtest simulation.

    For each time step the function:
      1. Generates signals from model predictions.
      2. Checks open positions for stop-loss, take-profit, max-hold, or
         signal-reversal exits.
      3. Enters new positions at the signal bar's close.
      4. Computes portfolio value.

    Args:
        predictions: Dict mapping symbol -> 1-D array of predicted
            probabilities of upward movement (aligned to *ohlcv_test* bars).
        y_test: Ground-truth binary labels for the test period (used for
            alignment length).
        ohlcv_test: Dict mapping symbol -> OHLCV DataFrame for the test
            period.  Each DataFrame must have at least as many rows as
            ``len(predictions[symbol])`` for that symbol.
        initial_capital: Starting capital in quote currency.
        regime: Regime gate mode string passed through to signals.

    Returns:
        Dict with keys:
            ``metrics`` (dict from ``compute_metrics``),
            ``equity_curve`` (np.ndarray),
            ``trades`` (list of trade dicts).

    Raises:
        ValueError: If *predictions* is empty or fewer than 2 common time
            steps exist across symbols.
    """
    if not predictions:
        raise ValueError("predictions dict is empty")

    symbols = list(predictions.keys())

    # ------------------------------------------------------------------
    # Build a common datetime index across all symbols in ohlcv_test.
    # ------------------------------------------------------------------
    common_index: Optional[pd.Index] = None
    for sym in symbols:
        idx = ohlcv_test[sym].index
        if common_index is None:
            common_index = idx
        else:
            common_index = common_index.intersection(idx)

    if common_index is None or len(common_index) < 2:
        raise ValueError("Need at least 2 common time steps across symbols")

    common_index = common_index.sort_values()
    n_steps = len(common_index)

    cash = initial_capital
    positions: Dict[str, dict] = {}
    equity_curve: List[float] = []
    trades: List[dict] = []

    # Map each symbol's predictions index to the common index.
    # For symbols with fewer predictions, extrapolate the last value.
    pred_step_map: Dict[str, Dict[pd.Timestamp, float]] = {}
    for sym in symbols:
        pred_arr = predictions[sym]
        sym_idx = ohlcv_test[sym].index
        mapping: Dict[pd.Timestamp, float] = {}
        for j, ts in enumerate(sym_idx):
            if ts in common_index:
                prob = float(pred_arr[j]) if j < len(pred_arr) else float(pred_arr[-1])
                mapping[ts] = prob
        pred_step_map[sym] = mapping

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------
    for step_idx in range(n_steps):
        current_time = common_index[step_idx]

        # Build current predictions dict (prob_up, prob_down).
        current_predictions: Dict[str, tuple[float, float]] = {}
        for sym in symbols:
            prob_up = pred_step_map[sym].get(current_time, 0.5)
            current_predictions[sym] = (prob_up, 1.0 - prob_up)

        # Build current OHLCV dict (single-row DataFrame per symbol).
        current_ohlcv: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            df = ohlcv_test[sym]
            if current_time in df.index:
                current_ohlcv[sym] = df.loc[[current_time]]

        if not current_ohlcv:
            equity_curve.append(cash)
            continue

        # Generate and enrich signals.
        signals = generate_signals(current_predictions, current_ohlcv, regime=regime)
        signals = apply_risk_management(signals, cash, current_ohlcv)

        # ------------------------------------------------------------------
        # Exit existing positions
        # ------------------------------------------------------------------
        closed: List[str] = []
        for sym, pos in list(positions.items()):
            if sym not in current_ohlcv:
                continue

            row = current_ohlcv[sym].iloc[0]
            exit_price: Optional[float] = None
            exit_reason: Optional[str] = None

            if pos["direction"] == "long":
                # Stop-loss.
                if row["low"] <= pos["stop_loss"]:
                    exit_price = pos["stop_loss"]
                    exit_reason = "stop_loss"
                # Take-profit.
                elif row["high"] >= pos["take_profit"]:
                    exit_price = pos["take_profit"]
                    exit_reason = "take_profit"
            else:  # short
                if row["high"] >= pos["stop_loss"]:
                    exit_price = pos["stop_loss"]
                    exit_reason = "stop_loss"
                elif row["low"] <= pos["take_profit"]:
                    exit_price = pos["take_profit"]
                    exit_reason = "take_profit"

            # Max-hold.
            if exit_reason is None and (
                step_idx - pos["entry_idx"]
            ) >= config.max_hold_candles:
                exit_price = row["close"]
                exit_reason = "max_hold"

            # Signal reversal (exit at open).
            if exit_reason is None:
                pos_signal = next(
                    (s for s in signals if s["symbol"] == sym), None
                )
                if pos_signal is not None and pos_signal["direction"] != pos["direction"]:
                    exit_price = row["open"]
                    exit_reason = "signal_reversal"

            if exit_reason is not None:
                trade_record, trade_pnl = _close_position(
                    pos, exit_price, current_time, exit_reason,
                )
                cash += pos["size"] + trade_pnl
                trades.append(trade_record)
                closed.append(sym)

        for sym in closed:
            del positions[sym]

        # ------------------------------------------------------------------
        # Enter new positions
        # ------------------------------------------------------------------
        for signal in signals:
            if signal["symbol"] not in positions:
                positions[signal["symbol"]] = {
                    "symbol": signal["symbol"],
                    "direction": signal["direction"],
                    "entry_price": signal["entry_price"],
                    "size": signal["size"],
                    "stop_loss": signal["stop_loss"],
                    "take_profit": signal["take_profit"],
                    "entry_time": signal["timestamp"],
                    "entry_idx": step_idx,
                }
                cash -= signal["size"]

        # ------------------------------------------------------------------
        # Portfolio valuation
        # ------------------------------------------------------------------
        portfolio_value = cash
        for sym, pos in positions.items():
            if sym in current_ohlcv:
                current_price = current_ohlcv[sym]["close"].iloc[-1]
                shares = pos["size"] / pos["entry_price"]
                portfolio_value += shares * current_price
            else:
                portfolio_value += pos["size"]

        equity_curve.append(portfolio_value)

    # ------------------------------------------------------------------
    # Close any positions still open at the end of the test period
    # ------------------------------------------------------------------
    for sym, pos in list(positions.items()):
        last_df = ohlcv_test[sym]
        exit_price = last_df["close"].iloc[-1]
        trade_record, trade_pnl = _close_position(
            pos, exit_price, common_index[-1], "end_of_test",
        )
        cash += pos["size"] + trade_pnl
        trades.append(trade_record)

    equity_curve[-1] = cash

    metrics = compute_metrics(np.array(equity_curve), trades, initial_capital)
    return {
        "metrics": metrics,
        "equity_curve": np.array(equity_curve),
        "trades": trades,
    }


def _close_position(
    pos: dict,
    exit_price: float,
    exit_time: pd.Timestamp,
    exit_reason: str,
) -> tuple:
    """Calculate PnL for a closed position and return trade record + PnL.

    Args:
        pos: Position dict with keys ``entry_price``, ``size``, ``direction``,
            ``entry_time``, ``symbol``.
        exit_price: Price at which the position is exited.
        exit_time: Timestamp of the exit.
        exit_reason: String reason for the exit.

    Returns:
        Tuple of ``(trade_record_dict, trade_pnl)``.
    """
    if pos["direction"] == "long":
        pnl_frac = (exit_price / pos["entry_price"]) - 1.0
    else:
        pnl_frac = 1.0 - (exit_price / pos["entry_price"])

    trade_pnl = pos["size"] * pnl_frac

    trade_record = {
        "symbol": pos["symbol"],
        "direction": pos["direction"],
        "entry_time": pos["entry_time"],
        "exit_time": exit_time,
        "entry_price": pos["entry_price"],
        "exit_price": exit_price,
        "size": pos["size"],
        "pnl": trade_pnl,
        "pnl_pct": pnl_frac * 100.0,
        "exit_reason": exit_reason,
    }
    return trade_record, trade_pnl


def compute_metrics(
    equity_curve: np.ndarray,
    trades: List[dict],
    initial_capital: float,
) -> dict:
    """Compute backtest performance metrics.

    Args:
        equity_curve: Array of portfolio values at each time step.
        trades: List of trade dicts with keys ``pnl`` and ``pnl_pct``.
        initial_capital: Starting capital.

    Returns:
        Dict with keys:
            ``total_return`` (float, %),
            ``sharpe_ratio`` (float, annualised),
            ``max_drawdown`` (float, %),
            ``win_rate`` (float, %),
            ``profit_factor`` (float),
            ``total_trades`` (int).
    """
    final_value = equity_curve[-1] if len(equity_curve) > 0 else initial_capital
    total_return = (final_value / initial_capital - 1.0) * 100.0

    # Per-bar returns (proxy for daily returns for Sharpe).
    bar_returns = np.diff(equity_curve) / equity_curve[:-1]
    if len(bar_returns) > 0 and np.std(bar_returns) > 1e-10:
        sharpe_ratio = float(
            np.mean(bar_returns) / np.std(bar_returns) * np.sqrt(252)
        )
    else:
        sharpe_ratio = 0.0

    # Max drawdown.
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    max_drawdown = float(np.min(drawdown)) * 100.0

    # Trade statistics.
    total_trades = len(trades)
    if total_trades > 0:
        wins = [t for t in trades if t["pnl"] > 0]
        win_rate = len(wins) / total_trades * 100.0

        gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        )
    else:
        win_rate = 0.0
        profit_factor = 0.0

    return {
        "total_return": round(total_return, 4),
        "sharpe_ratio": round(sharpe_ratio, 4),
        "max_drawdown": round(max_drawdown, 4),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "total_trades": total_trades,
    }


def plot_equity_curve(
    equity_curve: np.ndarray,
    metrics: dict,
    save_path: Optional[str] = None,
) -> None:
    """Plot the equity curve with a drawdown subplot.

    Args:
        equity_curve: Array of portfolio values.
        metrics: Dict from ``compute_metrics``.
        save_path: If provided, saves the figure to this path instead of
            displaying.
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]}
    )

    # Equity curve.
    ax1.plot(equity_curve, color="blue", linewidth=1.5)
    ax1.set_title(
        f"Equity Curve  |  "
        f"Return: {metrics.get('total_return', 0):.2f}%  |  "
        f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}  |  "
        f"Trades: {metrics.get('total_trades', 0)}"
    )
    ax1.set_ylabel("Portfolio Value")
    ax1.grid(True, alpha=0.3)

    # Drawdown.
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak * 100.0
    ax2.fill_between(
        range(len(drawdown)), drawdown, 0, color="red", alpha=0.3
    )
    ax2.set_title(
        f"Drawdown  (Max: {metrics.get('max_drawdown', 0):.2f}%)"
    )
    ax2.set_ylabel("Drawdown %")
    ax2.set_xlabel("Time Step")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
