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


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Average True Range (Wilder's smoothing), normalized by close.

    Args:
        high: High price series.
        low: Low price series.
        close: Closing price series.
        period: ATR lookback period.

    Returns:
        ATR / close — scale-invariant volatility measure.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr / close


def _stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int,
    d_period: int,
) -> Dict[str, pd.Series]:
    """Stochastic oscillator %K and %D, normalized to [0, 1].

    Args:
        high: High price series.
        low: Low price series.
        close: Closing price series.
        k_period: %K lookback window.
        d_period: %D SMA window.

    Returns:
        Dict with keys ``stoch_k`` and ``stoch_d``, each in [0, 1].
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    denom = highest_high - lowest_low
    k_raw = (close - lowest_low) / denom.replace(0.0, np.nan)
    d_raw = k_raw.rolling(window=d_period).mean()
    return {"stoch_k": k_raw, "stoch_d": d_raw}


def _obv_ratio(close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    """On-Balance Volume ratio vs its rolling mean.

    OBV is cumulative signed volume.  The ratio against its own MA makes it
    stationary and comparable across regimes.

    Args:
        close: Closing price series.
        volume: Volume series.
        period: Rolling window for the OBV mean.

    Returns:
        OBV / OBV.rolling(period).mean().
    """
    direction = close.diff().apply(np.sign)
    direction.iloc[0] = 1.0
    obv = (direction * volume).cumsum()
    obv_ma = obv.rolling(window=period).mean()
    return obv / obv_ma.replace(0.0, np.nan)


def _ema_ratio(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """EMA crossover ratio — EMA(fast) / EMA(slow).

    Values > 1 indicate short-term trend above long-term (bullish);
    values < 1 indicate bearish.

    Args:
        close: Closing price series.
        fast: Fast EMA period.
        slow: Slow EMA period.

    Returns:
        Series of EMA ratios.
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    return ema_fast / ema_slow.replace(0.0, np.nan)


def _hl_range_ratio(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Intra-candle volatility: (high - low) / close.

    Large values indicate volatile candles relative to price level.

    Args:
        high: High price series.
        low: Low price series.
        close: Closing price series.

    Returns:
        (high - low) / close.
    """
    return (high - low) / close.replace(0.0, np.nan)


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

    # --- ATR ---------------------------------------------------------------
    atr_period = getattr(config, "atr_period", 14)
    features["atr_14_norm"] = _atr(high, low, close, atr_period)

    # --- Stochastic oscillator ---------------------------------------------
    stoch_k = getattr(config, "stoch_k_period", 14)
    stoch_d = getattr(config, "stoch_d_period", 3)
    stoch = _stochastic(high, low, close, stoch_k, stoch_d)
    features["stoch_k_14"] = stoch["stoch_k"]
    features["stoch_d_3"] = stoch["stoch_d"]

    # --- OBV ratio ---------------------------------------------------------
    obv_period = getattr(config, "obv_period", 20)
    features["obv_ratio_20"] = _obv_ratio(close, volume, obv_period)

    # --- EMA crossover ratio -----------------------------------------------
    ema_fast = getattr(config, "ema_fast", 9)
    ema_slow = getattr(config, "ema_slow", 21)
    features["ema_ratio_9_21"] = _ema_ratio(close, ema_fast, ema_slow)

    # --- Price rate of change (ROC) ----------------------------------------
    roc_periods = getattr(config, "roc_periods", [5, 10, 20])
    for p in roc_periods:
        features[f"roc_{p}"] = close.pct_change(periods=p)

    # --- High-low range ratio ----------------------------------------------
    features["hl_range_ratio"] = _hl_range_ratio(high, low, close)

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


def build_target(df: pd.DataFrame, forward_periods: int = 1) -> pd.Series:
    """Build binary target series from OHLCV data.

    The target at row *t* is ``1`` if the close *forward_periods* candles
    ahead is strictly greater than the current close, otherwise ``0``.
    The last *forward_periods* rows are always NaN.

    A longer forward window (e.g. 5 for 1m candles) reduces micro-noise
    and gives the LSTM a smoother, more learnable signal.

    Args:
        df: OHLCV DataFrame (must contain a ``'close'`` column).
        forward_periods: Number of candles to look ahead (default 1).

    Returns:
        Series with the same index as *df* and dtype ``float64``. Values are
        ``1.0`` or ``0.0``; the last *forward_periods* entries are NaN.
    """
    if "close" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'close' column.")
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    target = (df["close"].shift(-forward_periods) > df["close"]).astype(float)
    # The last `forward_periods` rows have no future candle to compare against.
    target.iloc[-forward_periods:] = np.nan
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
    if val_df is not None and not val_df.empty:
        scaled_val = scaler.transform(val_df.values)

    scaled_test: Optional[np.ndarray] = None
    if test_df is not None and not test_df.empty:
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
