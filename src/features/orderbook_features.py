"""Order-book microstructure features.

12 features computed per snapshot, then aggregated (mean + std) into
24 columns aligned with 1-minute candle timestamps.

All computations vectorised over snapshots — no Python loops per level.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-snapshot feature helpers
# ---------------------------------------------------------------------------


def _spread_pct(df: pd.DataFrame) -> pd.Series:
    """(ask0 - bid0) / mid * 100."""
    mid = (df["bid_0_price"] + df["ask_0_price"]) / 2.0
    return (df["ask_0_price"] - df["bid_0_price"]) / mid.replace(0, np.nan) * 100.0


def _depth_imbalance_1(df: pd.DataFrame) -> pd.Series:
    """(bid_vol0 - ask_vol0) / (bid_vol0 + ask_vol0) — level 1."""
    bid = df["bid_0_vol"]
    ask = df["ask_0_vol"]
    denom = bid + ask
    return ((bid - ask) / denom.replace(0, np.nan)).clip(-1, 1)


def _depth_imbalance_top5(df: pd.DataFrame, depth: int) -> pd.Series:
    """Imbalance across top 5 levels (or fewer if depth < 5)."""
    n = min(5, depth)
    bid_cols = [f"bid_{i}_vol" for i in range(n)]
    ask_cols = [f"ask_{i}_vol" for i in range(n)]
    bid_sum = df[bid_cols].sum(axis=1)
    ask_sum = df[ask_cols].sum(axis=1)
    denom = bid_sum + ask_sum
    return ((bid_sum - ask_sum) / denom.replace(0, np.nan)).clip(-1, 1)


def _vol_normalized(df: pd.DataFrame, col: str, window: int = 30) -> pd.Series:
    """Volume divided by its rolling median — robust to outliers."""
    roll_med = df[col].rolling(window=window, min_periods=1).median()
    return (df[col] / roll_med.replace(0, np.nan)).clip(0, None)


def _weighted_price_deviation(df: pd.DataFrame, side: str, depth: int) -> pd.Series:
    """VWAP deviation from mid-price.

    Returns ``(VWAP / mid) - 1`` — positive when VWAP is above mid.
    """
    cols_p = [f"{side}_{i}_price" for i in range(depth)]
    cols_v = [f"{side}_{i}_vol" for i in range(depth)]
    prices = df[cols_p].values
    vols = df[cols_v].values
    vol_sum = vols.sum(axis=1)
    vwap = np.sum(prices * vols, axis=1) / np.where(vol_sum == 0, np.nan, vol_sum)
    mid = (df["bid_0_price"] + df["ask_0_price"]) / 2.0
    return pd.Series(vwap / mid.replace(0, np.nan) - 1.0, index=df.index)


def _slope_ratio(df: pd.DataFrame, depth: int) -> pd.Series:
    """Ratio of volume decay rates: bid side vs ask side.

    Fits an exponential decay constant of volume with distance from best
    price via log-linear regression.  Higher ratio → bid liquidity decays
    slower than ask (more supportive).
    """
    vols_bid = df[[f"bid_{i}_vol" for i in range(depth)]].values
    vols_ask = df[[f"ask_{i}_vol" for i in range(depth)]].values
    xs = np.arange(depth, dtype=float)
    eps = 1e-8

    def _decay_rate(vols: np.ndarray) -> np.ndarray:
        log_vols = np.log(np.maximum(vols, eps))
        x_centered = xs - xs.mean()
        denom = np.dot(x_centered, x_centered)
        if denom == 0:
            return np.zeros(vols.shape[0])
        slopes = -np.dot(log_vols - log_vols.mean(axis=1, keepdims=True), x_centered) / denom
        return slopes

    lambda_bid = _decay_rate(vols_bid)
    lambda_ask = _decay_rate(vols_ask)
    return pd.Series(
        np.where(np.abs(lambda_ask) < eps, np.nan, lambda_bid / lambda_ask),
        index=df.index,
    )


def _order_flow_delta(df: pd.DataFrame, depth: int) -> pd.Series:
    """Change in net resting liquidity (bid - ask total vol) vs prev snapshot."""
    bid_cols = [f"bid_{i}_vol" for i in range(depth)]
    ask_cols = [f"ask_{i}_vol" for i in range(depth)]
    net = df[bid_cols].sum(axis=1) - df[ask_cols].sum(axis=1)
    delta = net.diff()
    base = net.abs().shift(1)
    return (delta / base.replace(0, np.nan)).clip(-1, 1)


def _quote_imbalance_roc(df: pd.DataFrame) -> pd.Series:
    """Rate of change of depth_imbalance over 3 snapshots."""
    imb = _depth_imbalance_1(df)
    return imb.diff(periods=3) / 3.0


def _wall_density(df: pd.DataFrame, side: str, depth: int) -> pd.Series:
    """Max volume / mean volume — concentration of liquidity.

    High values indicate a large wall at one specific level.
    """
    cols = [f"{side}_{i}_vol" for i in range(depth)]
    vols = df[cols].values
    max_vol = vols.max(axis=1)
    mean_vol = vols.mean(axis=1)
    return pd.Series(
        max_vol / np.where(mean_vol == 0, np.nan, mean_vol),
        index=df.index,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Each entry is (function, needs_depth_arg).
_FEATURE_SPEC = [
    ("bid_ask_spread_pct", _spread_pct, False),
    ("depth_imbalance", _depth_imbalance_1, False),
    ("bid_vol_0", None, True),   # handled specially
    ("ask_vol_0", None, True),   # handled specially
    ("depth_imbalance_top5", _depth_imbalance_top5, True),
    ("weighted_bid_price", None, True),   # handled specially
    ("weighted_ask_price", None, True),   # handled specially
    ("depth_slope_ratio", _slope_ratio, True),
    ("order_flow_delta", _order_flow_delta, True),
    ("quote_imbalance_roc", _quote_imbalance_roc, False),
    ("wall_density_bid", None, True),     # handled specially
    ("wall_density_ask", None, True),     # handled specially
]


def compute_ob_features(
    snapshots: pd.DataFrame,
    depth: Optional[int] = None,
) -> pd.DataFrame:
    """Compute the 12 per-snapshot microstructure features.

    Args:
        snapshots: DataFrame with a DatetimeIndex and columns
            ``bid_i_price``, ``bid_i_vol``, ``ask_i_price``, ``ask_i_vol``
            for i in ``[0, depth)``.
        depth: Number of levels.  Inferred from column names if ``None``.

    Returns:
        DataFrame with the same index and 12 feature columns.
    """
    if snapshots.empty:
        return pd.DataFrame()

    if depth is None:
        bid_price_cols = [c for c in snapshots.columns if c.startswith("bid_") and c.endswith("_price")]
        depth = len(bid_price_cols)
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")

    features = pd.DataFrame(index=snapshots.index)

    for name, func, needs_depth in _FEATURE_SPEC:
        try:
            if name == "bid_vol_0":
                features[name] = _vol_normalized(snapshots, "bid_0_vol")
            elif name == "ask_vol_0":
                features[name] = _vol_normalized(snapshots, "ask_0_vol")
            elif name == "weighted_bid_price":
                features[name] = _weighted_price_deviation(snapshots, "bid", depth)
            elif name == "weighted_ask_price":
                features[name] = _weighted_price_deviation(snapshots, "ask", depth)
            elif name == "wall_density_bid":
                features[name] = _wall_density(snapshots, "bid", depth)
            elif name == "wall_density_ask":
                features[name] = _wall_density(snapshots, "ask", depth)
            elif needs_depth:
                features[name] = func(snapshots, depth)
            else:
                features[name] = func(snapshots)
        except Exception:
            logger.debug("Feature '%s' computation failed — filling NaN", name)
            features[name] = np.nan

    return features


def aggregate_to_candles(
    ob_features: pd.DataFrame,
    candle_timestamps: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Aggregate per-snapshot OB features into per-candle mean + std.

    Each snapshot is assigned to the most recent candle boundary that is
    ≤ the snapshot time.  Snapshots before the first candle are discarded.

    Args:
        ob_features: Per-snapshot feature DataFrame (output of
            :func:`compute_ob_features`).
        candle_timestamps: Sorted DatetimeIndex of candle timestamps.

    Returns:
        DataFrame indexed by *candle_timestamps* with columns
        ``<feature>_mean`` and ``<feature>_std`` (24 columns for 12 features).
    """
    if ob_features.empty:
        cols = []
        return pd.DataFrame(index=candle_timestamps, columns=cols)

    bins = candle_timestamps.sort_values()
    indices = bins.searchsorted(ob_features.index, side="right") - 1
    valid = (indices >= 0) & (indices < len(bins))
    if not valid.any():
        result = pd.DataFrame(index=candle_timestamps)
        return result

    assigned = ob_features.loc[valid].copy()
    assigned["_candle"] = bins[indices[valid]]

    means = assigned.groupby("_candle").mean()
    stds = assigned.groupby("_candle").std()

    result = pd.DataFrame(index=candle_timestamps)
    for col in means.columns:
        result[f"{col}_mean"] = means[col]
        result[f"{col}_std"] = stds[col]

    return result
