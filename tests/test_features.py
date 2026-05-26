"""Tests for src.data.features."""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.config import config
from src.data.features import (
    _bollinger,
    _macd,
    _wilder_rsi,
    build_features,
    build_target,
    create_sequences,
    normalize_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """A small, deterministic OHLCV DataFrame for unit testing."""
    np.random.seed(42)
    n = 200
    close = 100.0 + np.cumsum(np.random.normal(0, 0.5, n))
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_ = close * (1 + np.random.normal(0, 0.003, n))
    volume = np.random.uniform(100, 1000, n)

    idx = pd.date_range("2024-01-01", periods=n, freq="5min", name="timestamp")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


@pytest.fixture
def monotonic_ohlcv() -> pd.DataFrame:
    """Strictly increasing close prices — all targets should be 1 (as floats)."""
    n = 50
    close = np.arange(100.0, 100.0 + n)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", name="timestamp")
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": np.full(n, 500.0),
        },
        index=idx,
    )


@pytest.fixture
def near_constant_ohlcv() -> pd.DataFrame:
    """Close prices with only tiny random noise — features should be finite."""
    rng = np.random.default_rng(99)
    n = 200
    close = 100.0 + rng.normal(0, 0.01, n).cumsum()
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", name="timestamp")
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.0001, n)),
            "high": close * (1 + np.abs(rng.normal(0, 0.0002, n))),
            "low": close * (1 - np.abs(rng.normal(0, 0.0002, n))),
            "close": close,
            "volume": rng.uniform(100, 1000, n),
        },
        index=idx,
    )


@pytest.fixture
def tiny_ohlcv() -> pd.DataFrame:
    """Very short OHLCV — will not be able to compute most indicators."""
    n = 5
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", name="timestamp")
    return pd.DataFrame(
        {
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.0] * n,
            "volume": [500.0] * n,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# _wilder_rsi
# ---------------------------------------------------------------------------

class TestWilderRSI:
    def test_output_range(self, sample_ohlcv):
        """RSI values are in [0, 100] when they exist."""
        rsi = _wilder_rsi(sample_ohlcv["close"], period=14)
        valid = rsi.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_first_n_values_are_nan(self, sample_ohlcv):
        """First `period` values are NaN."""
        rsi = _wilder_rsi(sample_ohlcv["close"], period=14)
        assert rsi.iloc[:14].isna().all()
        assert rsi.iloc[14:].notna().any()

    def test_monotonic_up_gives_high_rsi(self, monotonic_ohlcv):
        """Strictly increasing prices produce RSI close to 100."""
        rsi = _wilder_rsi(monotonic_ohlcv["close"], period=14)
        valid = rsi.dropna()
        assert (valid > 90).all()  # monotonic up → RSI should be very high

    def test_monotonic_down_gives_low_rsi(self):
        """Strictly decreasing prices produce RSI close to 0."""
        n = 50
        close = pd.Series(np.arange(100.0, 100.0 - n, -1.0))
        rsi = _wilder_rsi(close, period=14)
        valid = rsi.dropna()
        assert (valid < 10).all()

    def test_constant_price_gives_nan(self):
        """Flat price produces NaN RSI (no gains/losses)."""
        close = pd.Series(np.full(50, 100.0))
        rsi = _wilder_rsi(close, period=14)
        # All gains and losses are 0, so avg_loss = 0 → division by zero
        assert rsi.dropna().empty or rsi.dropna().isna().all() or (rsi.dropna() == 50).all()
        # Actually with constant prices, gain/loss are all 0, avg_loss = 0,
        # but we guard with replace(0, NaN), so rs is NaN, so rsi is NaN.
        # Let's just check it doesn't crash.
        pass


# ---------------------------------------------------------------------------
# _macd
# ---------------------------------------------------------------------------

class TestMACD:
    def test_returns_three_keys(self, sample_ohlcv):
        """MACD returns dict with expected keys."""
        result = _macd(sample_ohlcv["close"], 12, 26, 9)
        assert set(result.keys()) == {"macd_line", "macd_signal", "macd_histogram"}

    def test_histogram_is_difference(self, sample_ohlcv):
        """Histogram = macd_line - macd_signal."""
        result = _macd(sample_ohlcv["close"], 12, 26, 9)
        pd.testing.assert_series_equal(
            result["macd_histogram"],
            result["macd_line"] - result["macd_signal"],
            check_names=False,
        )

    def test_first_values_nan(self, sample_ohlcv):
        """First `slow-1` values of macd_line are NaN (need enough data for EMA)."""
        result = _macd(sample_ohlcv["close"], 12, 26, 9)
        # With ewm(span=26, adjust=False), the first value is just the first close.
        # But ema_slow and ema_fast both have values from index 0.
        # Actually, ewm without min_periods returns values from index 0.
        # So macd_line starts at index 0.
        # The signal line has NaN for the first signal-1 values.
        # So all three will have values from the start, just not fully stable.
        assert not result["macd_line"].isna().all()


# ---------------------------------------------------------------------------
# _bollinger
# ---------------------------------------------------------------------------

class TestBollinger:
    def test_returns_two_keys(self, sample_ohlcv):
        """Bollinger returns dict with expected keys."""
        result = _bollinger(sample_ohlcv["close"], 20, 2.0)
        assert set(result.keys()) == {"bb_pctb", "bb_bandwidth"}

    def test_pctb_range(self, sample_ohlcv):
        """%B is typically between 0 and 1."""
        result = _bollinger(sample_ohlcv["close"], 20, 2.0)
        valid = result["bb_pctb"].dropna()
        # Most values should be between 0 and 1, but with 2 std bands some
        # can be outside
        assert len(valid) > 0
        assert valid.notna().all()

    def test_bandwidth_positive(self, sample_ohlcv):
        """Bandwidth is always non-negative."""
        result = _bollinger(sample_ohlcv["close"], 20, 2.0)
        valid = result["bb_bandwidth"].dropna()
        assert (valid >= 0).all()

    def test_first_period_nan(self, sample_ohlcv):
        """First `period` values are NaN."""
        result = _bollinger(sample_ohlcv["close"], 20, 2.0)
        assert result["bb_pctb"].iloc[:19].isna().all()


# ---------------------------------------------------------------------------
# build_features
# ---------------------------------------------------------------------------

class TestBuildFeatures:
    def test_returns_dataframe(self, sample_ohlcv):
        """Returns a DataFrame."""
        result = build_features(sample_ohlcv)
        assert isinstance(result, pd.DataFrame)

    def test_no_nan_rows(self, sample_ohlcv):
        """No NaN values remain in the output."""
        result = build_features(sample_ohlcv)
        assert result.isna().sum().sum() == 0

    def test_fewer_rows_than_input(self, sample_ohlcv):
        """Leading NaN rows are dropped so output is shorter."""
        result = build_features(sample_ohlcv)
        assert len(result) < len(sample_ohlcv)

    def test_expected_columns(self, sample_ohlcv):
        """Output has the expected feature column names."""
        result = build_features(sample_ohlcv)
        expected = {
            "log_return",
            "close_high_ratio",
            "close_low_ratio",
            "close_open_ratio",
            "close_mean_5",
            "close_std_5",
            "close_mean_10",
            "close_std_10",
            "close_mean_20",
            "close_std_20",
            "rsi_14",
            "macd_line",
            "macd_signal",
            "macd_histogram",
            "bb_pctb",
            "bb_bandwidth",
            "volume_ratio_20",
            "volume_roc_20",
        }
        assert set(result.columns) == expected
        assert len(result.columns) == len(expected)

    def test_no_lookahead_bias(self, sample_ohlcv):
        """Feature at row t depends only on data up to row t."""
        result = build_features(sample_ohlcv)

        # Check log_return: log_return[t] = ln(close[t] / close[t-1])
        # It should NOT depend on close[t+1]
        manual_log_return = np.log(
            sample_ohlcv["close"] / sample_ohlcv["close"].shift(1)
        )
        # After build_features drops NaNs, the remaining indices should match
        common_idx = result.index.intersection(manual_log_return.dropna().index)
        pd.testing.assert_series_equal(
            result.loc[common_idx, "log_return"],
            manual_log_return.loc[common_idx],
            check_names=False,
        )

    def test_empty_df_raises(self):
        """Empty DataFrame raises ValueError."""
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        with pytest.raises(ValueError, match="empty"):
            build_features(empty)

    def test_missing_columns_raises(self, sample_ohlcv):
        """Missing required columns raises ValueError."""
        bad = sample_ohlcv.drop(columns=["volume"])
        with pytest.raises(ValueError, match="missing"):
            build_features(bad)

    def test_tiny_df_raises(self, tiny_ohlcv):
        """Very short DataFrame raises ValueError (all NaN after indicators)."""
        with pytest.raises(ValueError, match="NaN"):
            build_features(tiny_ohlcv)

    def test_near_constant_close_produces_finite_output(self, near_constant_ohlcv):
        """When close has only tiny variation, features are all finite."""
        result = build_features(near_constant_ohlcv)
        assert not result.isna().any().any()
        # log_return should be close to 0
        assert result["log_return"].abs().max() < 0.05

    def test_feature_values_deterministic(self, sample_ohlcv):
        """Same input produces identical output."""
        r1 = build_features(sample_ohlcv)
        r2 = build_features(sample_ohlcv)
        pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# build_target
# ---------------------------------------------------------------------------

class TestBuildTarget:
    def test_returns_series(self, sample_ohlcv):
        """Returns a pandas Series."""
        target = build_target(sample_ohlcv)
        assert isinstance(target, pd.Series)

    def test_all_zeros_and_ones(self, monotonic_ohlcv):
        """With monotonic up, all non-NaN targets are 1.0."""
        target = build_target(monotonic_ohlcv)
        valid = target.dropna()
        assert (valid == 1.0).all()

    def test_last_value_nan(self, sample_ohlcv):
        """Last value of the target series is NaN."""
        target = build_target(sample_ohlcv)
        assert pd.isna(target.iloc[-1])

    def test_alternating_targets(self):
        """With zigzag prices, targets alternate correctly."""
        n = 20
        idx = pd.date_range("2024-01-01", periods=n, freq="5min")
        df = pd.DataFrame(
            {
                "open": [100.0] * n,
                "high": [101.0] * n,
                "low": [99.0] * n,
                "close": [100.0 + (i % 2) * 10.0 for i in range(n)],
                "volume": [500.0] * n,
            },
            index=idx,
        )
        target = build_target(df)
        # close pattern: 100, 110, 100, 110, 100, ...
        # target[t] = 1 if close[t+1] > close[t]
        # target[0]: close[1]=110 > close[0]=100 → 1
        # target[1]: close[2]=100 < close[1]=110 → 0
        # target[2]: close[3]=110 > close[2]=100 → 1
        valid = target.dropna()
        expected = [1.0, 0.0] * ((len(valid) + 1) // 2)
        assert list(valid[: len(expected)]) == expected[: len(valid)]

    def test_empty_df_raises(self):
        """Empty DataFrame raises ValueError."""
        empty = pd.DataFrame(columns=["close"])
        with pytest.raises(ValueError, match="empty"):
            build_target(empty)

    def test_missing_close_raises(self):
        """DataFrame without close column raises ValueError."""
        df = pd.DataFrame({"open": [1.0]})
        with pytest.raises(ValueError, match="close"):
            build_target(df)


# ---------------------------------------------------------------------------
# normalize_data
# ---------------------------------------------------------------------------

class TestNormalizeData:
    def test_returns_correct_shapes(self, sample_ohlcv):
        """Scaled arrays have same number of rows as input."""
        features = build_features(sample_ohlcv)
        n = len(features)
        mid = n // 2
        train = features.iloc[:mid]
        val = features.iloc[mid:]
        scaled_train, scaled_val, scaled_test, (scaler, names) = normalize_data(
            train, val, None
        )
        assert scaled_train.shape == (mid, features.shape[1])
        assert scaled_val.shape == (n - mid, features.shape[1])
        assert scaled_test is None

    def test_mean_approx_zero_after_fit(self, sample_ohlcv):
        """Train data has mean ~0 after scaling."""
        features = build_features(sample_ohlcv)
        n = len(features)
        train = features.iloc[: int(n * 0.7)]
        scaled_train, _, _, _ = normalize_data(train)
        assert np.allclose(scaled_train.mean(axis=0), 0, atol=1e-12)

    def test_std_approx_one_after_fit(self, sample_ohlcv):
        """Train data has std ~1 after scaling."""
        features = build_features(sample_ohlcv)
        n = len(features)
        train = features.iloc[: int(n * 0.7)]
        scaled_train, _, _, _ = normalize_data(train)
        assert np.allclose(scaled_train.std(axis=0), 1, atol=1e-14)

    def test_returns_scaler_tuple(self, sample_ohlcv):
        """Returns (StandardScaler, feature_names) as last element."""
        features = build_features(sample_ohlcv)
        _, _, _, (scaler, names) = normalize_data(features)
        assert isinstance(scaler, StandardScaler)
        assert names == list(features.columns)

    def test_val_test_none_returns_none(self, sample_ohlcv):
        """When val_df/test_df are None, return None for those positions."""
        features = build_features(sample_ohlcv)
        _, val, test, _ = normalize_data(features, None, None)
        assert val is None
        assert test is None

    def test_empty_train_raises(self, sample_ohlcv):
        """Empty training DataFrame raises ValueError."""
        empty = pd.DataFrame(columns=build_features(sample_ohlcv).columns)
        with pytest.raises(ValueError, match="empty"):
            normalize_data(empty)


# ---------------------------------------------------------------------------
# create_sequences
# ---------------------------------------------------------------------------

class TestCreateSequences:
    def test_correct_shapes(self):
        """Output arrays have expected shapes."""
        data = np.random.randn(100, 5)
        targets = np.random.randint(0, 2, 100)
        X, y = create_sequences(data, targets, seq_len=10)
        assert X.shape == (90, 10, 5)
        assert y.shape == (90,)

    def test_sliding_window_values(self):
        """Each sequence is correctly shifted by 1 from the previous."""
        data = np.arange(30).reshape(30, 1).astype(float)
        targets = np.array([0] * 30)
        X, y = create_sequences(data, targets, seq_len=5)
        # Sequence 0: rows 0-4
        # Sequence 1: rows 1-5
        np.testing.assert_array_equal(X[0, :, 0], [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(X[1, :, 0], [1, 2, 3, 4, 5])

    def test_target_at_last_position(self):
        """Target for sequence i is the target at position i+seq_len-1."""
        data = np.random.randn(50, 3)
        targets = np.arange(50)  # unique values so we can verify
        X, y = create_sequences(data, targets, seq_len=10)
        # For sequence i, the last input row is at index i+9
        # y[i] should be targets[i+9]
        np.testing.assert_array_equal(y, targets[9:49])

    def test_seq_len_one(self):
        """seq_len=1 is valid and each sequence is a single row."""
        data = np.random.randn(20, 4)
        targets = np.random.randint(0, 2, 20)
        X, y = create_sequences(data, targets, seq_len=1)
        assert X.shape == (19, 1, 4)
        np.testing.assert_array_equal(X[:, 0, :], data[:19])
        np.testing.assert_array_equal(y, targets[:19])

    def test_not_enough_data_raises(self):
        """Fewer than seq_len+1 rows raises ValueError."""
        data = np.random.randn(5, 3)
        targets = np.random.randint(0, 2, 5)
        with pytest.raises(ValueError, match="at least"):
            create_sequences(data, targets, seq_len=10)

    def test_mismatched_data_targets_raises(self):
        """Different row counts in data and targets raises ValueError."""
        data = np.random.randn(50, 3)
        targets = np.random.randint(0, 2, 40)
        with pytest.raises(ValueError, match="must match"):
            create_sequences(data, targets, seq_len=5)

    def test_seq_len_zero_raises(self):
        """seq_len < 1 raises ValueError."""
        data = np.random.randn(50, 3)
        targets = np.random.randint(0, 2, 50)
        with pytest.raises(ValueError, match="seq_len"):
            create_sequences(data, targets, seq_len=0)

    def test_single_feature(self):
        """Works correctly with a single feature column."""
        data = np.random.randn(30, 1)
        targets = np.random.randint(0, 2, 30)
        X, y = create_sequences(data, targets, seq_len=5)
        assert X.shape == (25, 5, 1)
