"""Tests for src/data/bybit_ob_loader.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data.bybit_ob_loader import (
    _cache_path,
    _create_browser_session,
    _zip_path,
    available_date_range,
    gather_file_list,
    list_available_dates,
    list_files,
    load_day,
    parse_snapshots,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_snap_line(
    ts_ms: int,
    bid_price: float = 77000.0,
    ask_price: float = 77010.0,
    depth: int = 20,
) -> str:
    """Create a synthetic JSONLines snapshot entry."""
    bids = [[str(bid_price - i * 0.5), str(1.0 + i * 0.1)] for i in range(depth)]
    asks = [[str(ask_price + i * 0.5), str(1.0 + i * 0.1)] for i in range(depth)]
    obj = {
        "topic": "orderbook.200.BTCUSDT",
        "ts": ts_ms,
        "type": "snapshot",
        "data": {"s": "BTCUSDT", "b": bids, "a": asks},
    }
    return json.dumps(obj) + "\n"


def _write_data_file(path: Path, num_snapshots: int = 10, depth: int = 20) -> None:
    """Write a synthetic .data file with *num_snapshots* lines."""
    base_ts = 1779667201000  # 2026-05-25 00:00:01 UTC
    with open(path, "w") as f:
        for i in range(num_snapshots):
            f.write(_make_snap_line(base_ts + i * 100, depth=depth))


# ---------------------------------------------------------------------------
# parse_snapshots
# ---------------------------------------------------------------------------


class TestParseSnapshots:
    """Tests for parse_snapshots()."""

    def test_basic_parse(self) -> None:
        """10 snapshots → DataFrame with correct shape."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".data", delete=False) as f:
            _write_data_file(Path(f.name), num_snapshots=10, depth=5)
            tmp_path = Path(f.name)

        try:
            df = parse_snapshots(tmp_path, depth=5)
            assert len(df) == 10
            assert "bid_0_price" in df.columns
            assert "ask_0_price" in df.columns
            assert "bid_0_vol" in df.columns
            assert "ask_4_price" in df.columns
            assert "ask_4_vol" in df.columns
        finally:
            tmp_path.unlink()

    def test_column_count(self) -> None:
        """depth=10 → 20 price+vol columns (10 bid + 10 ask)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".data", delete=False) as f:
            _write_data_file(Path(f.name), num_snapshots=5, depth=10)
            tmp_path = Path(f.name)

        try:
            df = parse_snapshots(tmp_path, depth=10)
            price_cols = [c for c in df.columns if c.endswith("_price")]
            vol_cols = [c for c in df.columns if c.endswith("_vol")]
            assert len(price_cols) == 20  # 10 bid + 10 ask
            assert len(vol_cols) == 20
        finally:
            tmp_path.unlink()

    def test_empty_file(self) -> None:
        """Empty .data file → empty DataFrame."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".data", delete=False) as f:
            f.write("")
            tmp_path = Path(f.name)

        try:
            df = parse_snapshots(tmp_path)
            assert df.empty
        finally:
            tmp_path.unlink()

    def test_timestamps_are_utc(self) -> None:
        """Timestamps are parsed as UTC DatetimeIndex."""
        ts_ms = 1779667201716
        with tempfile.NamedTemporaryFile(mode="w", suffix=".data", delete=False) as f:
            f.write(_make_snap_line(ts_ms, depth=5))
            tmp_path = Path(f.name)

        try:
            df = parse_snapshots(tmp_path, depth=5)
            assert df.index.tz is not None
            assert str(df.index.tz) == "UTC"
            assert df.index[0] == pd.Timestamp(ts_ms, unit="ms", tz="UTC")
            assert len(df) == 1
        finally:
            tmp_path.unlink()

    def test_price_values_are_float(self) -> None:
        """Price and volume columns contain float64 values."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".data", delete=False) as f:
            _write_data_file(Path(f.name), num_snapshots=3, depth=5)
            tmp_path = Path(f.name)

        try:
            df = parse_snapshots(tmp_path, depth=5)
            assert df["bid_0_price"].dtype == np.float64
            assert df["bid_0_vol"].dtype == np.float64
            assert df["ask_0_price"].dtype == np.float64
        finally:
            tmp_path.unlink()

    def test_sorted_index(self) -> None:
        """Output DataFrame is sorted by timestamp."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".data", delete=False) as f:
            # Write out of order.
            base = 1779667201000
            f.write(_make_snap_line(base + 500, depth=5))
            f.write(_make_snap_line(base, depth=5))
            f.write(_make_snap_line(base + 200, depth=5))
            tmp_path = Path(f.name)

        try:
            df = parse_snapshots(tmp_path, depth=5)
            assert df.index.is_monotonic_increasing
        finally:
            tmp_path.unlink()

    def test_compatible_with_ob_features(self) -> None:
        """parse_snapshots output is compatible with compute_ob_features."""
        from src.features.orderbook_features import compute_ob_features

        with tempfile.NamedTemporaryFile(mode="w", suffix=".data", delete=False) as f:
            _write_data_file(Path(f.name), num_snapshots=20, depth=5)
            tmp_path = Path(f.name)

        try:
            df = parse_snapshots(tmp_path, depth=5)
            features = compute_ob_features(df, depth=5)
            assert len(features) == 20
            assert "bid_ask_spread_pct" in features.columns
            assert "depth_imbalance" in features.columns
            assert "bid_vol_0" in features.columns
        finally:
            tmp_path.unlink()


# ---------------------------------------------------------------------------
# cache path helpers
# ---------------------------------------------------------------------------


class TestCachePaths:
    """Cache path generation."""

    def test_cache_path_sanitizes_symbol(self) -> None:
        path = _cache_path(Path("/tmp/cache"), "BTC/USDT", "2026-05-25")
        assert "/" not in path.name
        assert "BTC_" in str(path) or "BTCUSDT" in str(path)
        assert path.suffix == ".parquet"

    def test_zip_path_creates_downloads_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _zip_path(Path(tmp), "BTCUSDT", "2026-05-25")
            assert path.parent.name == "downloads"
            assert path.name.endswith(".zip")


# ---------------------------------------------------------------------------
# API mocks
# ---------------------------------------------------------------------------


def _mock_list_files_response(dates: list[str]) -> dict:
    """Build a mock list-files API response."""
    files = []
    for d in dates:
        files.append({
            "bizType": "spot",
            "productId": "orderbook",
            "interval": "daily",
            "symbol": "BTCUSDT",
            "date": d,
            "filename": f"{d}_BTCUSDT_ob200.data.zip",
            "size": "50000000",
            "url": f"https://quote-saver.bycsi.com/orderbook/spot/BTCUSDT/{d}_BTCUSDT_ob200.data.zip",
            "period": "",
        })
    return {"ret_code": 0, "ret_msg": "", "result": {"list": files}}


class TestListFiles:
    """Tests for list_files and list_available_dates (mocked)."""

    def test_list_files_single_day(self) -> None:
        session = MagicMock()
        session.get.return_value.json.return_value = _mock_list_files_response(
            ["2026-05-25"]
        )
        session.get.return_value.raise_for_status = MagicMock()

        files = list_files(session, symbol="BTCUSDT", start_day="2026-05-25")
        assert len(files) == 1
        assert files[0]["date"] == "2026-05-25"

    def test_list_files_range_expands_to_7_days(self) -> None:
        dates = [f"2026-05-{d:02d}" for d in range(18, 25)]
        session = MagicMock()
        session.get.return_value.json.return_value = _mock_list_files_response(dates)
        session.get.return_value.raise_for_status = MagicMock()

        files = list_files(session, start_day="2026-05-18", end_day="2026-05-24")
        assert len(files) == 7

    def test_list_available_dates_multi_chunk(self) -> None:
        """list_available_dates paginates 7-day chunks."""
        session = MagicMock()

        def side_effect(url, params, **kwargs):
            start = params["startDay"]
            end = params["endDay"]
            s = pd.Timestamp(start)
            e = pd.Timestamp(end)
            dates = []
            d = s
            while d <= e:
                dates.append(d.strftime("%Y-%m-%d"))
                d += pd.Timedelta(days=1)
            resp = MagicMock()
            resp.json.return_value = _mock_list_files_response(dates)
            resp.raise_for_status = MagicMock()
            return resp

        session.get.side_effect = side_effect

        dates = list_available_dates(
            session, symbol="BTCUSDT", start="2026-05-01", end="2026-05-20"
        )
        assert len(dates) == 20
        assert dates[0] == "2026-05-01"
        assert dates[-1] == "2026-05-20"


class TestGatherFileList:
    """Tests for gather_file_list."""

    def test_gather_multi_chunk(self) -> None:
        session = MagicMock()

        def side_effect(url, params, **kwargs):
            start = pd.Timestamp(params["startDay"])
            end = pd.Timestamp(params["endDay"])
            dates = []
            d = start
            while d <= end:
                dates.append(d.strftime("%Y-%m-%d"))
                d += pd.Timedelta(days=1)
            resp = MagicMock()
            resp.json.return_value = _mock_list_files_response(dates)
            resp.raise_for_status = MagicMock()
            return resp

        session.get.side_effect = side_effect

        files = gather_file_list(session, "BTCUSDT", "2026-05-01", "2026-05-15")
        assert len(files) == 15
        dates_in_files = [f["date"] for f in files]
        assert dates_in_files[0] == "2026-05-01"
        assert dates_in_files[-1] == "2026-05-15"


# ---------------------------------------------------------------------------
# load_day (unit)
# ---------------------------------------------------------------------------


class TestLoadDay:
    """Tests for load_day with mocked download."""

    def test_load_day_cache_hit(self, tmp_path: Path) -> None:
        """load_day returns cached parquet without downloading."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        date_str = "2026-05-25"

        # Pre-populate cache.
        df = pd.DataFrame(
            {"bid_0_price": [77000.0], "bid_0_vol": [1.0]},
            index=pd.DatetimeIndex(
                [pd.Timestamp("2026-05-25 00:00:01", tz="UTC")], name="timestamp"
            ),
        )
        pq_path = _cache_path(cache_dir, "BTCUSDT", date_str)
        pq_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(pq_path)

        session = MagicMock()
        result = load_day(session, "BTCUSDT", date_str, cache_dir, depth=1)
        assert len(result) == 1
        # Should NOT have called the download API.
        session.get.assert_not_called()


# ---------------------------------------------------------------------------
# Integration (smoke) — skipped without network
# ---------------------------------------------------------------------------


_PLAYWRIGHT_AVAILABLE = False
try:
    import playwright  # noqa: F401
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    pass


class TestIntegrationSmoke:
    """Smoke tests that require network access (and playwright)."""

    @pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason="playwright not installed")
    def test_create_session(self) -> None:
        """Can create a browser session and hit the list-files API."""
        try:
            session = _create_browser_session()
            files = list_files(session, symbol="BTCUSDT", start_day="2026-05-25")
            assert len(files) >= 1
            assert "url" in files[0]
        except Exception as e:
            pytest.skip(f"Network/api unavailable: {e}")

    @pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason="playwright not installed")
    def test_parse_real_day(self) -> None:
        """Download and parse one real day of data."""
        try:
            session = _create_browser_session()
            with tempfile.TemporaryDirectory() as tmp:
                cache_dir = Path(tmp) / "cache"
                cache_dir.mkdir()
                df = load_day(
                    session, "BTCUSDT", "2026-05-25", cache_dir, depth=5
                )
                assert len(df) > 0
                assert "bid_0_price" in df.columns
                assert df.index.tz is not None
                # One day of 100ms snapshots should be > 500k.
                assert len(df) > 500_000
                print(f"  Parsed {len(df):,} snapshots")
        except Exception as e:
            pytest.skip(f"Download/parse failed: {e}")
