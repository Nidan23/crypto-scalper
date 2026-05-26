"""argparse CLI for the crypto-scalping ML system."""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.config import config
from src.data.fetcher import fetch_multiple, fetch_ohlcv
from src.data.features import build_features, create_sequences
from src.data.pipeline import run_pipeline
from src.features.regime_features import build_regime_features
from src.model.predict import (
    load_trained_model,
    predict_single as _predict_single,
)
from src.model.regime_detector import _classify_no_trade_reason, train_regime_detector
from src.model.train import compute_metrics as compute_model_metrics
from src.strategy.backtest import compute_metrics, plot_equity_curve, run_backtest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def handle_fetch(args: argparse.Namespace) -> None:
    """Fetch and cache OHLCV data, then print a summary."""
    symbols: list = [args.symbol] if args.symbol else config.symbols
    timeframe: str = args.timeframe or config.timeframe

    print(f"Fetching {timeframe} OHLCV for {symbols} ...")
    result = fetch_multiple(symbols, timeframe=timeframe)

    for sym, df in result.items():
        print(f"\n{sym}:")
        print(f"  Rows:      {len(df)}")
        print(f"  Period:    {df.index[0]}  to  {df.index[-1]}")
        print(
            f"  Close:     {df['close'].min():.2f}  -  {df['close'].max():.2f}"
        )


def handle_train(args: argparse.Namespace) -> None:
    """Run the pipeline and train the model."""
    symbols: list = args.symbols or config.symbols
    print(f"Running pipeline for {symbols} ...")

    data = run_pipeline(symbols)

    print(f"  Train:  {data['X_train'].shape}")
    print(f"  Val:    {data['X_val'].shape}")
    print(f"  Test:   {data['X_test'].shape}")
    print(f"\nTraining model ...")

    from src.model.train import train_model

    model, history = train_model(data)

    # Evaluate on test set.
    model.eval()
    X_test_t = torch.FloatTensor(data["X_test"])
    with torch.no_grad():
        logits = model(X_test_t)
    probs = logits.numpy().flatten()
    preds = (probs >= 0.5).astype(int)
    accuracy = float((preds == data["y_test"]).mean())

    print(f"\nTest accuracy:               {accuracy:.4f}")
    print(f"Test samples:                {len(data['y_test'])}")
    print(f"Best validation loss:        {min(history['val_loss']):.6f}")
    print(f"Epochs trained:              {len(history['train_loss'])}")
    print(f"Model saved to               {config.model_dir}/")


def handle_predict(args: argparse.Namespace) -> None:
    """Load a model, fetch latest data, and print the prediction."""
    symbol: str = args.symbol or config.symbols[0]
    model_path: str = args.model_path or os.path.join(
        config.model_dir, "best.pt"
    )

    if not os.path.exists(model_path):
        print(
            f"Error: model not found at '{model_path}'. "
            f"Train a model first with: python -m src.cli train",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading model from {model_path} ...")
    model, scaler, feature_names = load_trained_model(model_path)

    if scaler is None:
        print(
            "Error: checkpoint has no scaler.  Cannot normalise features.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Fetch the latest data.
    print(f"Fetching {symbol} ...")
    df = fetch_ohlcv(symbol)
    if df.empty:
        print("Error: no data returned from exchange.", file=sys.stderr)
        sys.exit(1)

    # Build features and normalise.
    features = build_features(df)
    feature_cols = feature_names or [c for c in features.columns]

    # Ensure all required feature columns are present.
    missing = set(feature_cols) - set(features.columns)
    if missing:
        print(
            f"Error: feature columns missing from data: {missing}",
            file=sys.stderr,
        )
        sys.exit(1)

    X_raw = features[feature_cols].values.astype(np.float64)
    X_scaled: np.ndarray = scaler.transform(X_raw)  # type: ignore[union-attr]

    # Create the most recent sequence.
    if X_scaled.shape[0] < config.seq_len:
        n_pad = config.seq_len - X_scaled.shape[0]
        pad = np.repeat(X_scaled[:1], n_pad, axis=0)
        X_scaled = np.concatenate([pad, X_scaled], axis=0)

    # We need a dummy target for create_sequences — ignore the y output.
    dummy_target = np.zeros(X_scaled.shape[0], dtype=np.float32)
    X_seq, _ = create_sequences(
        X_scaled, dummy_target, config.seq_len
    )

    if X_seq.shape[0] == 0:
        print(
            "Error: cannot create a sequence from the available data.",
            file=sys.stderr,
        )
        sys.exit(1)

    last_seq = X_seq[-1]
    model.eval()
    with torch.no_grad():
        tensor = torch.FloatTensor(last_seq).unsqueeze(0)
        logit = model(tensor)
        prob_up = float(torch.sigmoid(logit).item())
    prob_down = 1.0 - prob_up
    confidence = abs(prob_up - 0.5) * 2.0

    print(f"\n{symbol} Prediction")
    print(f"  Probability UP:     {prob_up:.4f}")
    print(f"  Probability DOWN:   {prob_down:.4f}")
    print(f"  Confidence:         {confidence:.4f}")
    if prob_up >= config.confidence_threshold_long:
        print(f"  Signal:             LONG")
    elif prob_up <= config.confidence_threshold_short:
        print(f"  Signal:             SHORT")
    else:
        print(f"  Signal:             NEUTRAL")


def handle_backtest(args: argparse.Namespace) -> None:
    """Run pipeline, train model, backtest, and print a report."""
    symbols: list = args.symbols or config.symbols
    capital: float = args.capital
    regime_mode: str = getattr(config, "regime_mode", "off")

    print(f"Running pipeline for {symbols} ...")
    data = run_pipeline(symbols)

    print(f"Training model ...")
    from src.model.train import train_model

    model, _ = train_model(data)

    # Get model predictions on the test set.
    model.eval()
    X_test_t = torch.FloatTensor(data["X_test"])
    with torch.no_grad():
        test_logits = model(X_test_t)
    test_probs = test_logits.numpy().flatten()

    # --- Model accuracy metrics on test set ---
    model_metrics = compute_model_metrics(data["y_test"], test_probs)
    n_features = len(data["feature_names"])
    ob_enabled = getattr(config, "orderbook_enabled", False)
    print(f"\n{'=' * 52}")
    print(f"  MODEL ACCURACY (TEST SET)")
    print(f"{'=' * 52}")
    print(f"  Features:         {n_features:>8d}  (OB: {'on' if ob_enabled else 'off'})")
    print(f"  Accuracy:         {model_metrics['accuracy']:>8.4f}")
    print(f"  Precision:        {model_metrics['precision']:>8.4f}")
    print(f"  Recall:           {model_metrics['recall']:>8.4f}")
    print(f"  F1 Score:         {model_metrics['f1']:>8.4f}")
    print(f"  AUC-ROC:          {model_metrics['auc']:>8.4f}")
    accuracy = model_metrics['accuracy']
    if accuracy < 0.55:
        print(f"\n  Accuracy is below 55%. Suggestions to improve:")
        print(f"    - Increase lookback_candles (currently {config.lookback_candles})")
        print(f"    - Try different seq_len values (currently {config.seq_len})")
        print(f"    - Add more informative features or external data")
        print(f"    - Tune hidden_dim, num_layers, dropout, learning_rate")
        print(f"    - Reduce noise in target by using larger timeframe candles")
    n_test_seqs = len(test_probs)

    if n_test_seqs == 0:
        print(
            "Error: no test sequences available for backtesting.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Test sequences: {n_test_seqs}")

    # Collect per-symbol metadata for train/test split alignment and
    # OHLCV test data construction.
    sym_meta: Dict[str, dict] = {}
    for sym in symbols:
        df = fetch_ohlcv(sym)
        features = build_features(df)
        n_total = len(features)
        train_end = int(n_total * config.train_split)
        val_end = train_end + int(n_total * config.val_split)

        sym_meta[sym] = {
            "df": df,
            "features": features,
            "n_total": n_total,
            "train_end": train_end,
            "val_end": val_end,
        }

    # Build per-symbol OHLCV data for the test period.
    ohlcv_test: dict = {}
    for sym in symbols:
        meta = sym_meta[sym]
        df = meta["df"]
        features = meta["features"]
        val_end = meta["val_end"]
        test_start_idx = val_end + config.seq_len

        if test_start_idx + n_test_seqs > len(features):
            n_available = len(features) - test_start_idx
            if n_available < 1:
                ohlcv_slice = df.iloc[-n_test_seqs:]
            else:
                ts = features.iloc[
                    test_start_idx : test_start_idx + n_available
                ].index
                ohlcv_slice = df.loc[ts, ["open", "high", "low", "close", "volume"]]
        else:
            ts = features.iloc[
                test_start_idx : test_start_idx + n_test_seqs
            ].index
            ohlcv_slice = df.loc[ts, ["open", "high", "low", "close", "volume"]]

        ohlcv_test[sym] = ohlcv_slice

    min_rows = min(len(v) for v in ohlcv_test.values())
    if min_rows < 2:
        print(
            "Error: insufficient test OHLCV data for backtesting.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Regime Detection Gate ---
    regime_mask: Optional[np.ndarray] = None
    regime_stats: Optional[dict] = None

    if regime_mode != "off":
        # Build aligned regime features for all symbols.
        all_regime_frames: List[pd.DataFrame] = []
        train_end_total = 0
        val_end_total = 0

        for sym in symbols:
            meta = sym_meta[sym]
            regime_feat = build_regime_features(meta["df"])
            # Align to TA feature index by reindexing.
            aligned = regime_feat.reindex(meta["features"].index)
            train_end_total += meta["train_end"]
            val_end_total += meta["val_end"]
            all_regime_frames.append(aligned)

        combined_regime = pd.concat(all_regime_frames, axis=0).reset_index(drop=True)
        # Forward-fill any NaNs introduced by reindex alignment.
        combined_regime = combined_regime.ffill().bfill()

        train_regime = combined_regime.iloc[:train_end_total]
        test_regime = combined_regime.iloc[val_end_total:]

        try:
            confidence = (
                config.regime_strict_confidence if regime_mode == "strict"
                else config.regime_loose_confidence
            )

            detector = train_regime_detector(train_regime)
            regime_path = os.path.join(config.model_dir, "regime_detector.pkl")
            os.makedirs(config.model_dir, exist_ok=True)
            detector.save(regime_path)

            preds, probs = detector.predict(test_regime)
            n_regime = len(preds)

            # Build sequence-level mask: sequence i uses regime at row i+seq_len-1.
            n_mask = min(min_rows, n_regime - config.seq_len + 1)
            regime_mask = np.ones(n_mask, dtype=bool)
            reasons: Dict[str, int] = {}

            for i in range(n_mask):
                regime_idx = i + config.seq_len - 1
                if regime_idx >= n_regime:
                    break
                is_trade = bool(preds[regime_idx])
                trade_prob = float(probs[regime_idx])

                if not is_trade or trade_prob < confidence:
                    regime_mask[i] = False
                    reason = _classify_no_trade_reason(test_regime.iloc[regime_idx])
                    reasons[reason] = reasons.get(reason, 0) + 1

            total_filtered = int((~regime_mask).sum())
            regime_stats = {
                "mode": regime_mode,
                "confidence_threshold": confidence,
                "total_checked": n_mask,
                "total_filtered": total_filtered,
                "filter_rate": total_filtered / max(n_mask, 1),
                "reasons": reasons,
            }
        except Exception as exc:
            logger.warning("Regime detector failed: %s — trading everything.", exc)
            regime_mask = None
            regime_stats = {"mode": regime_mode, "error": str(exc)}

    # --- Apply regime mask ---
    if regime_mask is not None:
        # Build filtered per-symbol predictions.
        predictions: dict = {}
        for sym in symbols:
            predictions[sym] = test_probs[:len(regime_mask)][regime_mask]

        # Filter y_test.
        y_test = data["y_test"][:len(regime_mask)][regime_mask]

        # Filter OHLCV test data to matching rows.
        ohlcv_test_filtered: dict = {}
        for sym in symbols:
            df = ohlcv_test[sym]
            keep_idx = np.where(regime_mask[:len(df)])[0]
            ohlcv_test_filtered[sym] = df.iloc[keep_idx]
        ohlcv_test = ohlcv_test_filtered

        n_filtered = len(y_test)
        print(f"Test sequences after regime filter: {n_filtered} "
              f"({regime_stats['total_filtered']} filtered, "
              f"{regime_stats['filter_rate']:.1%} rate)")
    else:
        predictions = {}
        for sym in symbols:
            predictions[sym] = test_probs[:min_rows]
        y_test = data["y_test"][:min_rows]
        n_filtered = min_rows

    if n_filtered < 2:
        print(
            "Error: insufficient sequences after regime filtering.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Backtesting with ${capital:,.0f} capital "
          f"({n_filtered} steps) ...")
    result = run_backtest(predictions, y_test, ohlcv_test, capital, regime=regime_mode)

    # --- Print regime stats ---
    if regime_stats and regime_stats.get("mode") != "off":
        print(f"\n{'=' * 52}")
        print(f"  REGIME DETECTION GATE")
        print(f"{'=' * 52}")
        print(f"  Mode:              {regime_stats['mode']:>12}")
        print(f"  Confidence:        {regime_stats.get('confidence_threshold', 0):>12.2f}")
        print(f"  Checked:           {regime_stats.get('total_checked', 0):>12d}")
        print(f"  Filtered:          {regime_stats.get('total_filtered', 0):>12d}")
        print(f"  Filter Rate:       {regime_stats.get('filter_rate', 0):>11.1%}")
        if regime_stats.get("reasons"):
            reasons_str = " ".join(
                f"{k}({v})" for k, v in sorted(
                    regime_stats["reasons"].items(), key=lambda x: -x[1]
                )
            )
            print(f"  Reasons:           {reasons_str}")

    # --- Backtest results ---
    metrics = result["metrics"]
    print(f"\n{'=' * 52}")
    print(f"  BACKTEST RESULTS")
    print(f"{'=' * 52}")
    print(f"  Total Return:     {metrics['total_return']:>8.2f}%")
    print(f"  Sharpe Ratio:     {metrics['sharpe_ratio']:>8.2f}")
    print(f"  Max Drawdown:     {metrics['max_drawdown']:>8.2f}%")
    print(f"  Win Rate:         {metrics['win_rate']:>7.1f}%")
    print(f"  Profit Factor:    {metrics['profit_factor']:>8.2f}")
    print(f"  Total Trades:     {metrics['total_trades']:>8d}")

    # Plot equity curve.
    plot_path = os.path.join(
        config.plot_dir, "backtest_equity_curve.png"
    )
    os.makedirs(config.plot_dir, exist_ok=True)
    plot_equity_curve(
        result["equity_curve"], metrics, save_path=plot_path
    )
    print(f"\nEquity curve saved to {plot_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate handler."""
    parser = argparse.ArgumentParser(
        description="Crypto Scalping ML System",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = False

    # --- fetch ----------------------------------------------------------
    fetch_p = subparsers.add_parser(
        "fetch", help="Fetch and cache OHLCV data"
    )
    fetch_p.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Trading pair (default: first from config)",
    )
    fetch_p.add_argument(
        "--timeframe",
        type=str,
        default=None,
        help="Candle timeframe (default: config value)",
    )

    # --- train ----------------------------------------------------------
    train_p = subparsers.add_parser(
        "train", help="Train the LSTM model"
    )
    train_p.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=None,
        help="Trading pairs (default: config.symbols)",
    )

    # --- predict --------------------------------------------------------
    predict_p = subparsers.add_parser(
        "predict", help="Run model prediction"
    )
    predict_p.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Trading pair (default: first from config)",
    )
    predict_p.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model checkpoint (default: models/best.pt)",
    )

    # --- backtest -------------------------------------------------------
    backtest_p = subparsers.add_parser(
        "backtest", help="Run walk-forward backtest"
    )
    backtest_p.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=None,
        help="Trading pairs (default: config.symbols)",
    )
    backtest_p.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Initial capital (default: 10000)",
    )
    backtest_p.add_argument(
        "--regime",
        type=str,
        default=None,
        choices=["strict", "loose", "off"],
        help="Regime detection gate: strict, loose, off (default)",
    )

    args = parser.parse_args()

    handlers = {
        "fetch": handle_fetch,
        "train": handle_train,
        "predict": handle_predict,
        "backtest": handle_backtest,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


# ---------------------------------------------------------------------------
# Bybit OB fetch handler
# ---------------------------------------------------------------------------


def handle_bybit_fetch(args: argparse.Namespace) -> None:
    """Download Bybit historical spot orderbook data and cache as parquet."""
    from src.data.bybit_ob_loader import (
        _create_browser_session,
        available_date_range,
        gather_file_list,
        load_range,
    )
    from src.config import config as cfg

    symbol: str = args.symbol
    start: str = args.start
    end: str = args.end if args.end else pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    depth: int = args.depth
    workers: int = args.workers
    dry_run: bool = args.dry_run
    cache_dir = Path(cfg.bybit_ob_cache_dir)

    print(f"Bybit OB Download")
    print(f"  Symbol:  {symbol}")
    print(f"  Depth:   {depth}")
    print()

    print("Getting browser session ...")
    session = _create_browser_session()

    print("Checking available date range ...")
    earliest, latest = available_date_range(session, symbol)
    print(f"  Available: {earliest} → {latest}")
    print(f"  Requested: {start} → {end}")

    # Clamp to available range.
    start_dt = max(pd.Timestamp(start), pd.Timestamp(earliest))
    end_dt = min(pd.Timestamp(end), pd.Timestamp(latest))
    if start_dt > end_dt:
        print(f"Error: no data in requested range (available: {earliest} → {latest})")
        return

    start = start_dt.strftime("%Y-%m-%d")
    end = end_dt.strftime("%Y-%m-%d")
    n_days = (end_dt - start_dt).days + 1

    print(f"  Effective: {start} → {end} ({n_days} days)")

    files = gather_file_list(session, symbol, start, end)
    total_size = sum(int(f.get("size", 0)) for f in files)
    print(f"  Files:    {len(files)}")
    print(f"  Size:     {total_size / 1e9:.1f} GB")

    if dry_run:
        print("\n  [DRY RUN] No files downloaded.")
        print(f"  Cache dir: {cache_dir.resolve()}")
        return

    print(f"\nDownloading {n_days} days ({workers} workers) ...")
    print(f"  Cache dir: {cache_dir.resolve()}")
    print()

    df = load_range(
        session=session,
        symbol=symbol,
        start_date=start,
        end_date=end,
        cache_dir=cache_dir,
        depth=depth,
        max_workers=workers,
    )

    gb = df.memory_usage(deep=True).sum() / 1e9
    print(f"\nDone. Loaded {len(df):,} snapshots ({gb:.1f} GB RAM)")
    print(f"  Date range: {df.index.min()} → {df.index.max()}")
    print(f"  Cached to:  {cache_dir.resolve()}")


if __name__ == "__main__":
    main()
