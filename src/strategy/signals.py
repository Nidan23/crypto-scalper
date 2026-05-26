"""Convert model predictions to trade signals with risk management."""

from typing import Dict, List

import numpy as np
import pandas as pd

from src.config import config


def generate_signals(
    predictions: Dict[str, tuple[float, float]],
    ohlcv_dict: Dict[str, pd.DataFrame],
    regime: str = "off",
) -> List[dict]:
    """Convert model predictions to trade signals.

    A LONG signal is emitted when the upward probability is >=
    ``config.confidence_threshold_long`` (0.55).  A SHORT signal is emitted
    when the upward probability is <= ``config.confidence_threshold_short``
    (0.45).  Otherwise no signal is generated for that symbol.

    Args:
        predictions: Dict mapping symbol -> ``(prob_up, prob_down)`` where
            both values are in ``[0, 1]`` and sum to 1.
        ohlcv_dict: Dict mapping symbol -> OHLCV DataFrame.  The last row
            of each DataFrame provides the entry price and timestamp.
        regime: Regime gate mode string (``"strict"``, ``"loose"``, ``"off"``).

    Returns:
        List of signal dicts, each with keys:
            ``symbol``, ``direction`` (``"long"``|``"short"``),
            ``confidence`` (float), ``entry_price`` (float),
            ``timestamp`` (datetime), ``regime`` (str).
    """
    signals: List[dict] = []
    for symbol, (prob_up, prob_down) in predictions.items():
        df = ohlcv_dict.get(symbol)
        if df is None or df.empty:
            continue

        entry_price = float(df["close"].iloc[-1])
        timestamp = df.index[-1]

        if prob_up >= config.confidence_threshold_long:
            signals.append({
                "symbol": symbol,
                "direction": "long",
                "confidence": prob_up,
                "entry_price": entry_price,
                "timestamp": timestamp,
                "regime": regime,
            })
        elif prob_up <= config.confidence_threshold_short:
            signals.append({
                "symbol": symbol,
                "direction": "short",
                "confidence": prob_down,
                "entry_price": entry_price,
                "timestamp": timestamp,
                "regime": regime,
            })

    return signals


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR) using Wilder's smoothing.

    ATR is the exponentially weighted moving average of the True Range.
    True Range is the maximum of:
        - high - low
        - |high - previous close|
        - |low - previous close|

    Args:
        df: DataFrame with columns ``['open', 'high', 'low', 'close']``.
        period: Lookback period (default 14).

    Returns:
        Series of ATR values aligned to *df*'s index.
    """
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(span=period, adjust=False).mean()
    return atr


def calculate_position_size(
    capital: float,
    price: float,
    atr: float,
    direction: str = "long",
) -> dict:
    """Calculate position size, stop-loss, and take-profit levels.

    Position size is a fixed percentage of available capital.  Stop-loss
    and take-profit are offset by ATR multiples from the entry price.

    Args:
        capital: Available capital in quote currency.
        price: Entry price.
        atr: Current ATR value.
        direction: ``"long"`` or ``"short"``.

    Returns:
        Dict with keys ``size`` (notional quote-currency value), ``stop_loss``,
        ``take_profit``.
    """
    size = capital * config.position_size_pct

    if direction == "long":
        stop_loss = price - config.stop_loss_atr_mult * atr
        take_profit = price + config.take_profit_atr_mult * atr
    else:
        stop_loss = price + config.stop_loss_atr_mult * atr
        take_profit = price - config.take_profit_atr_mult * atr

    return {
        "size": size,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
    }


def apply_risk_management(
    signals: List[dict],
    capital: float,
    ohlcv_dict: Dict[str, pd.DataFrame],
) -> List[dict]:
    """Enrich trade signals with position sizing and risk parameters.

    For each signal, computes ATR for the symbol and attaches position
    size, stop-loss, and take-profit.

    Signals for which ATR cannot be computed (zero / NaN) are omitted.

    Args:
        signals: List of signal dicts from ``generate_signals``.
        capital: Available capital in quote currency.
        ohlcv_dict: Dict mapping symbol -> OHLCV DataFrame.

    Returns:
        List of signal dicts enriched with keys:
            ``size``, ``stop_loss``, ``take_profit``.
    """
    enriched: List[dict] = []
    for signal in signals:
        df = ohlcv_dict.get(signal["symbol"])
        if df is None or df.empty:
            continue

        atr_series = calculate_atr(df)
        current_atr = atr_series.iloc[-1]

        if pd.isna(current_atr) or current_atr <= 0.0:
            continue

        sizing = calculate_position_size(
            capital=capital,
            price=signal["entry_price"],
            atr=current_atr,
            direction=signal["direction"],
        )

        signal["size"] = sizing["size"]
        signal["stop_loss"] = sizing["stop_loss"]
        signal["take_profit"] = sizing["take_profit"]
        enriched.append(signal)

    return enriched
