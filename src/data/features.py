"""Feature engineering from raw OHLCV data.

All transformations are computed without lookahead bias — each feature value at
time *t* depends only on data available at or before time *t*.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing (EMA alpha = 1/period).

    Args:
        close: Closing price series.
        period: RSI lookback period (default 14).

    Returns:
        RSI values in [0, 100]; NaN for the first ``period`` rows.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder smoothing via exponential weighted moving average
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)  # avoid division by zero
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _macd(
    close: pd.Series,
    fast: int,
    slow: int,
    signal: int,
) -> Dict[str, pd.Series]:
    """MACD indicator.

    Args:
        close: Closing price series.
        fast: Fast EMA period.
        slow: Slow EMA period.
        signal: Signal-line EMA period.

    Returns:
        Dict with keys ``macd_line``, ``macd_signal``, ``macd_histogram``.
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal_line
    return {
        "macd_line": macd_line,
        "macd_signal": macd_signal_line,
        "macd_histogram": macd_hist,
    }


def _bollinger(
    close: pd.Series,
    period: int,
    std_mult: float,
) -> Dict[str, pd.Series]:
    """Bollinger Bands %B and bandwidth.

    Args:
        close: Closing price series.
        period: Rolling-window length for the middle band (SMA).
        std_mult: Number of standard deviations for the outer bands.

    Returns:
        Dict with keys ``bb_pctb`` and ``bb_bandwidth``.
    """
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = middle + std_mult * std
    lower = middle - std_mult * std

    # %B = (close - lower) / (upper - lower)
    denom = upper - lower
    pctb = (close - lower) / denom.replace(0.0, np.nan)

    # Bandwidth = (upper - lower) / middle
    bandwidth = denom / middle.replace(0.0, np.nan)

    return {"bb_pctb": pctb, "bb_bandwidth": bandwidth}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute feature columns from raw OHLCV data.

    The following features are created (all without lookahead bias):

    - **Log returns**: log(close[t] / close[t-1]).
    - **Price ratios**: close/high, close/low, close/open.
    - **Rolling stats**: mean & std of close over 5, 10, 20 periods.
    - **RSI(period)**: Wilder-smoothed RSI.
    - **MACD(fast, slow, signal)**: MACD line, signal line, histogram.
    - **Bollinger Bands %B & Bandwidth**.
    - **Volume features**: volume ratio vs 20-period MA, volume ROC.

    NaN-producing leading windows are forward-filled and any remaining NaN rows
    (which will be at the very start of the series) are dropped.

    Args:
        df: OHLCV DataFrame with columns
            ``['open', 'high', 'low', 'close', 'volume']``.

    Returns:
        DataFrame containing only derived feature columns, with leading-NaN rows
        removed.  The ``close`` column is **not** included (it is not a feature
        itself).
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required columns: {missing}"
        )
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    volume = df["volume"]

    features = pd.DataFrame(index=df.index)

    # --- Log returns -------------------------------------------------------
    features["log_return"] = np.log(close / close.shift(1))

    # --- Price ratios ------------------------------------------------------
    features["close_high_ratio"] = close / high
    features["close_low_ratio"] = close / low
    features["close_open_ratio"] = close / open_

    # --- Rolling stats -----------------------------------------------------
    for w in (5, 10, 20):
        features[f"close_mean_{w}"] = close.rolling(window=w).mean()
        features[f"close_std_{w}"] = close.rolling(window=w).std()

    # --- RSI ---------------------------------------------------------------
    rsi_period = getattr(config, "rsi_period", 14)
    features[f"rsi_{rsi_period}"] = _wilder_rsi(close, rsi_period)

    # --- MACD --------------------------------------------------------------
    macd_fast = getattr(config, "macd_fast", 12)
    macd_slow = getattr(config, "macd_slow", 26)
    macd_signal = getattr(config, "macd_signal", 9)
    macd_vals = _macd(close, macd_fast, macd_slow, macd_signal)
    features["macd_line"] = macd_vals["macd_line"]
    features["macd_signal"] = macd_vals["macd_signal"]
    features["macd_histogram"] = macd_vals["macd_histogram"]

    # --- Bollinger Bands ---------------------------------------------------
    bb_period = getattr(config, "bb_period", 20)
    bb_std = getattr(config, "bb_std", 2.0)
    bb_vals = _bollinger(close, bb_period, bb_std)
    features["bb_pctb"] = bb_vals["bb_pctb"]
    features["bb_bandwidth"] = bb_vals["bb_bandwidth"]

    # --- Volume features ---------------------------------------------------
    features["volume_ratio_20"] = volume / volume.rolling(window=20).mean()
    features["volume_roc_20"] = volume.pct_change(periods=20)

    # --- Clean NaNs --------------------------------------------------------
    # Forward-fill first (catches any internal NaNs from edge cases), then
    # drop remaining rows that still have NaN (locked-in leading windows).
    features = features.ffill()
    before = len(features)
    features = features.dropna()
    after = len(features)

    if after < 1:
        raise ValueError(
            f"All {before} feature rows were NaN after forward-fill and drop. "
            "Check that input data has enough non-NaN observations for the "
            "requested indicator periods."
        )

    return features


def build_target(df: pd.DataFrame) -> pd.Series:
    """Build binary target series from OHLCV data.

    The target at row *t* is ``1`` if the next candle's close is strictly
    greater than the current close, otherwise ``0``.  The last row of the
    returned series is always NaN (no next candle to compare against).

    Args:
        df: OHLCV DataFrame (must contain a ``'close'`` column).

    Returns:
        Series with the same index as *df* and dtype ``float64``. Values are
        ``1.0`` or ``0.0``; the last entry is ``NaN`` (no next candle to
        compare against).
    """
    if "close" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'close' column.")
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    target = (df["close"].shift(-1) > df["close"]).astype(float)
    # The last row has no "next candle", so it should be NaN.
    target.iloc[-1] = np.nan
    return target


def normalize_data(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
) -> Tuple[
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
    Tuple[StandardScaler, List[str]],
]:
    """Fit a StandardScaler on training data and transform train/val/test sets.

    The scaler is fitted **only** on the training DataFrame to avoid data
    leakage.  Validation and test sets are transformed using the fitted scaler.

    Args:
        train_df: Training features.
        val_df: Optional validation features.
        test_df: Optional test features.

    Returns:
        Tuple of ``(scaled_train, scaled_val, scaled_test, (scaler, feature_names))``.
        ``scaled_val`` and ``scaled_test`` are ``None`` when their corresponding
        input was ``None``.
    """
    if train_df.empty:
        raise ValueError("train_df is empty, cannot fit scaler.")

    feature_columns = list(train_df.columns)
    scaler = StandardScaler()

    scaled_train: np.ndarray = scaler.fit_transform(train_df.values)

    scaled_val: Optional[np.ndarray] = None
    if val_df is not None:
        scaled_val = scaler.transform(val_df.values)

    scaled_test: Optional[np.ndarray] = None
    if test_df is not None:
        scaled_test = scaler.transform(test_df.values)

    return scaled_train, scaled_val, scaled_test, (scaler, feature_columns)


def create_sequences(
    data: np.ndarray,
    targets: np.ndarray,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding-window sequences for an LSTM / RNN model.

    Each sequence consists of ``seq_len`` consecutive rows from *data* and is
    paired with the target **at the last position of the sequence** (i.e. the
    binary label for the candle immediately following the sequence window).

    Args:
        data: Array of shape ``(n_samples, n_features)``.
        targets: Array of shape ``(n_samples,)``.
        seq_len: Number of time steps per input sequence.

    Returns:
        ``(X, y)`` where:
        - ``X`` has shape ``(n_sequences, seq_len, n_features)``.
        - ``y`` has shape ``(n_sequences,)``.

    Raises:
        ValueError: If there are fewer than ``seq_len + 1`` samples (cannot
            create even one full sequence).
    """
    n_samples = data.shape[0]
    if n_samples < seq_len + 1:
        raise ValueError(
            f"Need at least {seq_len + 1} samples to create one sequence "
            f"(seq_len={seq_len}), but only {n_samples} available."
        )
    if data.shape[0] != targets.shape[0]:
        raise ValueError(
            f"data rows ({data.shape[0]}) and targets rows ({targets.shape[0]}) "
            "must match."
        )
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}.")

    n_features = data.shape[1]
    n_sequences = n_samples - seq_len

    X = np.zeros((n_sequences, seq_len, n_features), dtype=data.dtype)
    y = np.zeros((n_sequences,), dtype=targets.dtype)

    for i in range(n_sequences):
        X[i] = data[i : i + seq_len]
        y[i] = targets[i + seq_len - 1]

    return X, y
