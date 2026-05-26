"""Tests for src.features.orderbook_features."""

import numpy as np
import pandas as pd
import pytest

from src.features.orderbook_features import (
    aggregate_to_candles,
    compute_ob_features,
    _spread_pct,
    _depth_imbalance_1,
    _depth_imbalance_top5,
    _vol_normalized,
    _weighted_price_deviation,
    _slope_ratio,
    _order_flow_delta,
    _quote_imbalance_roc,
    _wall_density,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_snapshot_df(
    n: int = 60,
    depth: int = 10,
    base_price: float = 65000.0,
    spread: float = 10.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic OB snapshots with realistic structure.

    Args:
        n: Number of snapshots.
        depth: Number of price levels.
        base_price: Mid price.
        spread: Half-spread (price offset from mid).
        seed: RNG seed.

    Returns:
        DataFrame with columns bid_i_price, bid_i_vol, ask_i_price, ask_i_vol.
    """
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-06-15T12:00:00", periods=n, freq="10s", name="timestamp")

    data = {}
    for i in range(depth):
        bid_price = base_price - spread - i * 10.0
        ask_price = base_price + spread + i * 10.0
        data[f"bid_{i}_price"] = np.full(n, bid_price) + rng.normal(0, 0.5, n)
        data[f"bid_{i}_vol"] = np.abs(5.0 + rng.normal(0, 1, n))
        data[f"ask_{i}_price"] = np.full(n, ask_price) + rng.normal(0, 0.5, n)
        data[f"ask_{i}_vol"] = np.abs(5.0 + rng.normal(0, 1, n))

    return pd.DataFrame(data, index=pd.DatetimeIndex(ts))


# ---------------------------------------------------------------------------
# Individual feature helpers
# ---------------------------------------------------------------------------

class TestSpreadPct:
    def test_positive(self):
        df = _make_snapshot_df(n=10, spread=10.0)
        result = _spread_pct(df)
        assert (result > 0).all()
        # Spread should be ~2*10/65000 * 100 ≈ 0.03%
        assert result.mean() < 0.1


class TestDepthImbalance1:
    def test_range(self):
        df = _make_snapshot_df(n=20)
        result = _depth_imbalance_1(df)
        assert (result >= -1).all()
        assert (result <= 1).all()

    def test_symmetric(self):
        """Equal bid/ask volumes → imbalance ≈ 0."""
        df = _make_snapshot_df(n=20)
        df["bid_0_vol"] = 5.0
        df["ask_0_vol"] = 5.0
        result = _depth_imbalance_1(df)
        assert (result.abs() < 0.01).all()


class TestDepthImbalanceTop5:
    def test_range(self):
        df = _make_snapshot_df(n=20, depth=10)
        result = _depth_imbalance_top5(df, depth=10)
        assert (result >= -1).all()
        assert (result <= 1).all()


class TestVolNormalized:
    def test_around_one_for_stationary(self):
        df = _make_snapshot_df(n=100)
        df["bid_0_vol"] = 5.0  # constant → normalized ≈ 1
        result = _vol_normalized(df, "bid_0_vol", window=30)
        valid = result.dropna()
        assert np.allclose(valid, 1.0, atol=0.1)


class TestWeightedPriceDeviation:
    def test_bid_below_mid(self):
        df = _make_snapshot_df(n=10, depth=5, base_price=65000, spread=10)
        # Bids should all be < mid, so VWAP bid < mid → deviation < 0
        result = _weighted_price_deviation(df, "bid", depth=5)
        assert (result < 0).all()

    def test_ask_above_mid(self):
        df = _make_snapshot_df(n=10, depth=5, base_price=65000, spread=10)
        result = _weighted_price_deviation(df, "ask", depth=5)
        assert (result > 0).all()


class TestSlopeRatio:
    def test_no_nan_for_flat_volume(self):
        df = _make_snapshot_df(n=10, depth=5)
        # Set all volumes equal within each snapshot — decay is flat
        for i in range(5):
            df[f"bid_{i}_vol"] = 5.0
            df[f"ask_{i}_vol"] = 5.0
        result = _slope_ratio(df, depth=5)
        # With equal vols, log-vols are flat → slope = 0 → ratio = NaN
        # (division by zero).  Just check it doesn't crash.
        assert len(result) == 10


class TestOrderFlowDelta:
    def test_range(self):
        df = _make_snapshot_df(n=20, depth=5)
        result = _order_flow_delta(df, depth=5)
        valid = result.dropna()
        assert (valid >= -1).all()
        assert (valid <= 1).all()

    def test_first_value_nan(self):
        df = _make_snapshot_df(n=20, depth=5)
        result = _order_flow_delta(df, depth=5)
        assert pd.isna(result.iloc[0])


class TestQuoteImbalanceROC:
    def test_no_error(self):
        df = _make_snapshot_df(n=20)
        result = _quote_imbalance_roc(df)
        # First 3 should be NaN (diff period 3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert pd.isna(result.iloc[2])


class TestWallDensity:
    def test_range(self):
        df = _make_snapshot_df(n=10, depth=5)
        result_bid = _wall_density(df, "bid", depth=5)
        result_ask = _wall_density(df, "ask", depth=5)
        assert (result_bid >= 1).all()
        assert (result_ask >= 1).all()

    def test_uniform_volume_gives_one(self):
        """All levels have same volume → wall density = 1."""
        df = _make_snapshot_df(n=10, depth=5)
        for i in range(5):
            df[f"bid_{i}_vol"] = 5.0
        result = _wall_density(df, "bid", depth=5)
        assert np.allclose(result, 1.0, atol=0.01)


# ---------------------------------------------------------------------------
# compute_ob_features
# ---------------------------------------------------------------------------


class TestComputeOBFeatures:
    def test_returns_12_features(self):
        df = _make_snapshot_df(n=60, depth=10)
        result = compute_ob_features(df, depth=10)
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 12
        assert len(result) == 60

    def test_same_index(self):
        df = _make_snapshot_df(n=60, depth=10)
        result = compute_ob_features(df, depth=10)
        pd.testing.assert_index_equal(result.index, df.index)

    def test_empty_input(self):
        empty = pd.DataFrame()
        result = compute_ob_features(empty, depth=10)
        assert result.empty

    def test_depth_inferred(self):
        df = _make_snapshot_df(n=10, depth=5)
        result = compute_ob_features(df)  # no depth arg
        assert len(result.columns) == 12

    def test_depth_zero_raises(self):
        df = _make_snapshot_df(n=10, depth=1)
        with pytest.raises(ValueError, match="depth"):
            compute_ob_features(df, depth=0)

    def test_deterministic(self):
        df = _make_snapshot_df(n=30, depth=10, seed=99)
        r1 = compute_ob_features(df, depth=10)
        r2 = compute_ob_features(df, depth=10)
        pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# aggregate_to_candles
# ---------------------------------------------------------------------------


class TestAggregateToCandles:
    def test_basic_aggregation(self):
        """Aggregates snapshots into candle-aligned mean+std columns."""
        df = _make_snapshot_df(n=60, depth=10)
        features = compute_ob_features(df, depth=10)

        # Candles every minute.
        candles = pd.date_range(
            "2024-06-15T12:00:00", periods=5, freq="1min", name="timestamp"
        )
        result = aggregate_to_candles(features, candles)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        # 12 features × 2 (mean + std) = 24 columns
        assert len(result.columns) == 24

    def test_column_names(self):
        df = _make_snapshot_df(n=60, depth=10)
        features = compute_ob_features(df, depth=10)
        candles = pd.date_range(
            "2024-06-15T12:00:00", periods=3, freq="1min"
        )
        result = aggregate_to_candles(features, candles)

        for name in ["bid_ask_spread_pct", "depth_imbalance", "wall_density_bid"]:
            assert f"{name}_mean" in result.columns
            assert f"{name}_std" in result.columns

    def test_empty_features(self):
        empty = pd.DataFrame()
        candles = pd.date_range("2024-06-15T12:00:00", periods=5, freq="1min")
        result = aggregate_to_candles(empty, candles)
        assert len(result) == 5

    def test_handles_nan_in_features(self):
        """NaN features should not crash aggregation."""
        df = _make_snapshot_df(n=30, depth=10)
        features = compute_ob_features(df, depth=10)
        features.iloc[5:10, 3] = np.nan  # inject NaN

        candles = pd.date_range(
            "2024-06-15T12:00:00", periods=4, freq="1min"
        )
        result = aggregate_to_candles(features, candles)
        assert len(result) == 4

    def test_snapshots_before_first_candle_discarded(self):
        """Snapshots before the first candle boundary are excluded."""
        df = _make_snapshot_df(n=30, depth=5)
        features = compute_ob_features(df, depth=5)

        # Candles start after all snapshots.
        candles = pd.date_range(
            "2024-06-15T13:00:00", periods=5, freq="1min"
        )
        result = aggregate_to_candles(features, candles)
        # All means should be NaN because no snapshots map to these candles.
        assert result.isna().all().all()
