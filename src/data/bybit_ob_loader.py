"""Bybit historical spot orderbook loader.

Downloads daily L2 orderbook snapshots from Bybit's public history data
API, parses the JSONLines format, and caches them as parquet files
compatible with the existing OrderBookFetcher cache format.

API is public — no API key required, but a browser session (cookies
from visiting the history-data page) is needed.
"""

from __future__ import annotations

import json
import logging
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "https://www.bybit.com/x-api/quote/public/support/download"
_CDN_BASE = "https://quote-saver.bycsi.com"
_HISTORY_PAGE = "https://www.bybit.com/derivatives/en/history-data"
_CHUNK_DAYS = 7  # API returns max 7 days per request
_DEFAULT_DEPTH = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_ts(ts_ms: int) -> pd.Timestamp:
    """Convert epoch milliseconds to pandas Timestamp (UTC)."""
    return pd.Timestamp(ts_ms, unit="ms", tz="UTC")


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


def _create_browser_session() -> requests.Session:
    """Create a ``requests.Session`` with cookies from a Playwright browser.

    Launches Firefox in headless mode, visits the Bybit history-data page
    to obtain a valid session, then extracts cookies and User-Agent into a
    requests session that can call the download APIs directly.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise RuntimeError(
            "playwright is required for Bybit OB download. "
            "Install with: pip install playwright && python -m playwright install firefox"
        )

    session = requests.Session()

    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto(_HISTORY_PAGE, timeout=30000, wait_until="networkidle")
            page.wait_for_timeout(3000)

            # Use the actual browser User-Agent (critical — Bybit checks this).
            ua = page.evaluate("() => navigator.userAgent")

            cookies = page.context.cookies()
            for c in cookies:
                session.cookies.set(
                    c["name"], c["value"],
                    domain=c.get("domain", ".bybit.com"),
                    path=c.get("path", "/"),
                )

            session.headers.update({
                "User-Agent": ua,
                "Accept": "application/json",
                "Referer": _HISTORY_PAGE,
            })
        finally:
            browser.close()

    return session


# ---------------------------------------------------------------------------
# API wrappers
# ---------------------------------------------------------------------------


def list_files(
    session: requests.Session,
    symbol: str = "BTCUSDT",
    start_day: str = "2025-04-01",
    end_day: Optional[str] = None,
) -> list[dict]:
    """Call the Bybit list-files API for a 7-day (or less) window.

    Args:
        session: Authenticated ``requests.Session``.
        symbol: Trading pair, e.g. ``"BTCUSDT"``.
        start_day: Start date ``YYYY-MM-DD``.
        end_day: End date ``YYYY-MM-DD`` (default: start_day + 6 days).

    Returns:
        List of file-info dicts with keys ``date``, ``filename``, ``size``, ``url``.
    """
    if end_day is None:
        end_day = str(pd.Timestamp(start_day) + pd.Timedelta(days=6))[:10]

    params = {
        "bizType": "spot",
        "productId": "orderbook",
        "symbols": symbol,
        "interval": "daily",
        "startDay": start_day,
        "endDay": end_day,
    }
    resp = session.get(f"{_BASE_URL}/list-files", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if data.get("ret_code") != 0:
        raise RuntimeError(f"list-files API error: {data.get('ret_msg', 'unknown')}")
    return data["result"]["list"]


def list_available_dates(
    session: requests.Session,
    symbol: str = "BTCUSDT",
    start: str = "2025-04-01",
    end: Optional[str] = None,
) -> list[str]:
    """Return all available dates for *symbol* in [*start*, *end*].

    Queries the API in 7-day chunks.  If *end* is ``None``, uses today's
    date in UTC.
    """
    if end is None:
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    dates: list[str] = []
    chunk_start = pd.Timestamp(start)
    chunk_stop = pd.Timestamp(end)

    while chunk_start <= chunk_stop:
        chunk_end = min(chunk_start + pd.Timedelta(days=6), chunk_stop)
        files = list_files(
            session,
            symbol=symbol,
            start_day=chunk_start.strftime("%Y-%m-%d"),
            end_day=chunk_end.strftime("%Y-%m-%d"),
        )
        for f in files:
            dates.append(f["date"])
        chunk_start = chunk_end + pd.Timedelta(days=1)

    return sorted(set(dates))


def available_date_range(
    session: requests.Session,
    symbol: str = "BTCUSDT",
) -> tuple[str, str]:
    """Return ``(earliest, latest)`` available dates for *symbol*.

    Does a coarse binary search for the start date, then scans forward
    to find the exact earliest day.
    """
    # First find latest (today).
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    latest_files = list_files(session, symbol=symbol, start_day=today)
    if not latest_files:
        # Try yesterday
        yesterday = (pd.Timestamp(today) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        latest_files = list_files(session, symbol=symbol, start_day=yesterday)
    latest = latest_files[-1]["date"] if latest_files else today

    # Binary search for earliest.
    lo = pd.Timestamp("2020-01-01")
    hi = pd.Timestamp(today)
    earliest: Optional[str] = None

    while lo <= hi:
        mid = lo + (hi - lo) // 2
        mid_str = mid.strftime("%Y-%m-%d")
        try:
            files = list_files(session, symbol=symbol, start_day=mid_str)
            if files:
                earliest = files[0]["date"]
                hi = mid - pd.Timedelta(days=1)
            else:
                lo = mid + pd.Timedelta(days=7)
        except Exception:
            lo = mid + pd.Timedelta(days=7)

    if earliest is None:
        raise RuntimeError(f"No available data found for {symbol}")

    return earliest, latest


# ---------------------------------------------------------------------------
# Download & parse
# ---------------------------------------------------------------------------


def _cache_path(cache_dir: Path, symbol: str, date_str: str) -> Path:
    """Parquet cache path for a given symbol + date."""
    safe = symbol.replace("/", "_")
    return cache_dir / f"{safe}_{date_str}.parquet"


def _zip_path(cache_dir: Path, symbol: str, date_str: str) -> Path:
    """Download cache path for a raw zip."""
    safe = symbol.replace("/", "_")
    downloads = cache_dir / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    return downloads / f"{safe}_{date_str}_ob200.data.zip"


def download_day(
    session: requests.Session,
    symbol: str,
    date_str: str,
    cache_dir: Path,
) -> Path:
    """Download the OB zip for one day, cache it, and return the extracted .data path.

    Args:
        session: Authenticated requests session.
        symbol: Trading pair.
        date_str: Date ``YYYY-MM-DD``.
        cache_dir: Directory for cached files.

    Returns:
        Path to the extracted ``.data`` file.
    """
    # First, get the download URL from the API.
    files = list_files(session, symbol=symbol, start_day=date_str, end_day=date_str)
    if not files:
        raise FileNotFoundError(f"No file available for {symbol} on {date_str}")
    file_info = files[0]
    url = file_info["url"]

    # Download zip if not cached.
    zip_path = _zip_path(cache_dir, symbol, date_str)
    if not zip_path.exists():
        logger.info("Downloading %s (%s MB) ...", date_str, int(file_info["size"]) // 1_000_000)
        resp = session.get(url, timeout=600, stream=True)
        resp.raise_for_status()
        tmp = zip_path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
        tmp.rename(zip_path)

    # Extract.
    data_path = cache_dir / "downloads" / f"{symbol.replace('/', '_')}_{date_str}_ob200.data"
    if not data_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if name.endswith(".data"):
                    zf.extract(name, cache_dir / "downloads")
                    # Rename to our canonical name.
                    extracted = cache_dir / "downloads" / name
                    if extracted != data_path:
                        extracted.rename(data_path)
                    break

    return data_path


def parse_snapshots(
    data_path: Path,
    depth: int = _DEFAULT_DEPTH,
) -> pd.DataFrame:
    """Parse a Bybit JSONLines `.data` file into a flat snapshots DataFrame.

    Args:
        data_path: Path to the extracted ``.data`` file.
        depth: Number of price levels to keep per side.

    Returns:
        DataFrame with columns ``timestamp``, ``bid_0_price``,
        ``bid_0_vol``, ..., ``ask_N_price``, ``ask_N_vol``.
        Index is a UTC DatetimeIndex from the snapshot timestamps.
    """
    rows: list[dict] = []
    timestamps: list[pd.Timestamp] = []

    with open(data_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                snap = json.loads(line)
            except json.JSONDecodeError:
                continue

            ts_ms = snap.get("ts", 0)
            data = snap.get("data", {})
            bids = data.get("b", [])
            asks = data.get("a", [])

            row = {"timestamp": _parse_ts(ts_ms)}
            for i in range(min(len(bids), depth)):
                price_str, vol_str = bids[i][0], bids[i][1]
                row[f"bid_{i}_price"] = float(price_str)
                row[f"bid_{i}_vol"] = float(vol_str)
            for i in range(min(len(asks), depth)):
                price_str, vol_str = asks[i][0], asks[i][1]
                row[f"ask_{i}_price"] = float(price_str)
                row[f"ask_{i}_vol"] = float(vol_str)

            rows.append(row)
            timestamps.append(_parse_ts(ts_ms))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, index=pd.DatetimeIndex(timestamps, name="timestamp"))
    df.sort_index(inplace=True)
    return df


def load_day(
    session: requests.Session,
    symbol: str,
    date_str: str,
    cache_dir: Path,
    depth: int = _DEFAULT_DEPTH,
) -> pd.DataFrame:
    """Load one day of OB data, using parquet cache if available.

    Returns a DataFrame in the format described by :func:`parse_snapshots`.
    """
    pq_path = _cache_path(cache_dir, symbol, date_str)
    if pq_path.exists():
        logger.debug("Cache hit: %s", pq_path)
        return pd.read_parquet(pq_path)

    data_path = download_day(session, symbol, date_str, cache_dir)
    df = parse_snapshots(data_path, depth=depth)

    pq_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(pq_path, index=True)
    logger.info("Cached %s → %s (%d snapshots)", date_str, pq_path, len(df))
    return df


def load_range(
    session: requests.Session,
    symbol: str,
    start_date: str,
    end_date: str,
    cache_dir: Path,
    depth: int = _DEFAULT_DEPTH,
    max_workers: int = 3,
) -> pd.DataFrame:
    """Load multiple days of OB data, downloading in parallel.

    Args:
        session: Authenticated requests session.
        symbol: Trading pair.
        start_date: First date ``YYYY-MM-DD``.
        end_date: Last date ``YYYY-MM-DD``.
        cache_dir: Directory for parquet cache.
        depth: Number of price levels to keep.
        max_workers: Parallel download threads.

    Returns:
        Concatenated DataFrame sorted by timestamp.
    """
    dates = pd.date_range(start_date, end_date, freq="D")
    date_strs = [d.strftime("%Y-%m-%d") for d in dates]

    frames: list[pd.DataFrame] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(load_day, session, symbol, d, cache_dir, depth): d
            for d in date_strs
        }
        for future in as_completed(futures):
            d = futures[future]
            try:
                df = future.result()
                if not df.empty:
                    frames.append(df)
                    logger.info("Loaded %s: %d snapshots", d, len(df))
            except Exception as exc:
                logger.warning("Failed to load %s: %s", d, exc)

    if not frames:
        raise RuntimeError(f"No data loaded for {symbol} in [{start_date}, {end_date}]")

    result = pd.concat(frames).sort_index()
    return result


# ---------------------------------------------------------------------------
# Dry-run / info helpers
# ---------------------------------------------------------------------------


def estimate_size(files: list[dict]) -> int:
    """Sum of file sizes in bytes from a list-files response."""
    return sum(int(f.get("size", 0)) for f in files)


def gather_file_list(
    session: requests.Session,
    symbol: str,
    start: str,
    end: str,
) -> list[dict]:
    """Return the full list of downloadable files for a date range."""
    all_files: list[dict] = []
    chunk_start = pd.Timestamp(start)
    chunk_stop = pd.Timestamp(end)
    while chunk_start <= chunk_stop:
        chunk_end = min(chunk_start + pd.Timedelta(days=6), chunk_stop)
        files = list_files(
            session,
            symbol=symbol,
            start_day=chunk_start.strftime("%Y-%m-%d"),
            end_day=chunk_end.strftime("%Y-%m-%d"),
        )
        all_files.extend(files)
        chunk_start = chunk_end + pd.Timedelta(days=1)
    return all_files


# ---------------------------------------------------------------------------
# Pipeline integration helpers
# ---------------------------------------------------------------------------


def sync_missing_dates(
    session: requests.Session,
    symbol: str = "BTCUSDT",
    cache_dir: Path | None = None,
    depth: int = _DEFAULT_DEPTH,
    max_workers: int = 3,
) -> int:
    """Download any days available on Bybit but missing from local cache.

    Args:
        session: Authenticated requests session.
        symbol: Trading pair.
        cache_dir: Parquet cache directory (default: ``config.bybit_ob_cache_dir``).
        depth: Number of price levels to keep.
        max_workers: Parallel download threads.

    Returns:
        Number of new days downloaded.
    """
    if cache_dir is None:
        from src.config import config as _cfg
        cache_dir = Path(_cfg.bybit_ob_cache_dir)

    available = list_available_dates(session, symbol=symbol)
    if not available:
        logger.warning("No dates available for %s via Bybit API.", symbol)
        return 0

    missing = []
    for d in available:
        pq_path = _cache_path(cache_dir, symbol, d)
        if not pq_path.exists():
            missing.append(d)

    if not missing:
        logger.info("All %d dates already cached locally.", len(available))
        return 0

    logger.info(
        "Syncing %d missing day(s) out of %d available for %s.",
        len(missing), len(available), symbol,
    )

    # Download missing in batches via load_range.
    missing_start = missing[0]
    missing_end = missing[-1]
    load_range(
        session=session,
        symbol=symbol,
        start_date=missing_start,
        end_date=missing_end,
        cache_dir=cache_dir,
        depth=depth,
        max_workers=max_workers,
    )
    return len(missing)


def load_bybit_ob_for_range(
    symbol: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    cache_dir: Path | None = None,
    depth: int = _DEFAULT_DEPTH,
    auto_sync: bool = True,
) -> pd.DataFrame | None:
    """Load Bybit OB snapshots for a time range, with optional auto-sync.

    Args:
        symbol: Trading pair (e.g. ``"BTCUSDT"``).
        start_dt: Start of time range (inclusive, tz-aware).
        end_dt: End of time range (inclusive, tz-aware).
        cache_dir: Parquet cache directory.
        depth: Number of price levels to keep.
        auto_sync: If ``True``, download any missing days first.

    Returns:
        DataFrame with snapshots, or ``None`` if no Bybit data is available.
    """
    if cache_dir is None:
        from src.config import config as _cfg
        cache_dir = Path(_cfg.bybit_ob_cache_dir)

    if auto_sync:
        # Only launch browser if there's no cache for ANY date of this symbol.
        # A quick glob check prevents browser launch when cache is totally empty.
        symbol_safe = symbol.replace("/", "_")
        existing = list(cache_dir.glob(f"{symbol_safe}_*.parquet"))
        if not existing:
            logger.debug("No Bybit OB cache at all — skipping auto-sync.")
        else:
            have_all = True
            for d in pd.date_range(start_dt.normalize(), end_dt.normalize(), freq="D", tz=start_dt.tz):
                if not _cache_path(cache_dir, symbol, d.strftime("%Y-%m-%d")).exists():
                    have_all = False
                    break
            if not have_all:
                try:
                    session = _create_browser_session()
                    sync_missing_dates(
                        session=session,
                        symbol=symbol,
                        cache_dir=cache_dir,
                        depth=depth,
                        max_workers=2,
                    )
                except Exception as exc:
                    logger.warning("Auto-sync failed: %s — using cached data only.", exc)

    # Find cached parquet files within the date range.
    date_range = pd.date_range(
        start_dt.normalize(), end_dt.normalize(), freq="D", tz=start_dt.tz
    )
    frames: list[pd.DataFrame] = []
    for d in date_range:
        date_str = d.strftime("%Y-%m-%d")
        pq_path = _cache_path(cache_dir, symbol, date_str)
        if pq_path.exists():
            try:
                df_day = pd.read_parquet(pq_path)
                if not df_day.empty:
                    frames.append(df_day)
            except Exception as exc:
                logger.warning("Failed to read %s: %s", pq_path, exc)

    if not frames:
        return None

    result = pd.concat(frames).sort_index()
    # Filter to the exact time range.
    mask = (result.index >= start_dt) & (result.index <= end_dt)
    result = result.loc[mask]
    return result if not result.empty else None
