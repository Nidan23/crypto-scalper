#!/usr/bin/env python3
"""Crypto Scalper — one-command backtesting CLI.

Usage:
    python scalper.py backtest --pair BTC/USDT --capital 100 --leverage 10
    python scalper.py backtest --pair ETH/USDT --timeframe 5m --leverage 5
    python scalper.py train --pair BTC/USDT
    python scalper.py fetch --pair SOL/USDT
    python scalper.py predict --pair BTC/USDT
"""

import argparse
import sys

from src.cli import handle_backtest, handle_fetch, handle_predict, handle_train
from src.cli import handle_bybit_fetch  # type: ignore[attr-defined]
from src.config import config


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="scalper",
        description="Crypto Scalping ML — backtest, train, predict, fetch",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── backtest ──────────────────────────────────────────────────────
    bt = sub.add_parser("backtest", help="Run walk-forward backtest")
    bt.add_argument("--pair", type=str, default=None,
                    help="Trading pair (e.g. BTC/USDT)")
    bt.add_argument("--timeframe", type=str, default=None,
                    help="Candle interval: 1m, 5m, 15m, 1h (default: 1m)")
    bt.add_argument("--capital", type=float, default=10000.0,
                    help="Initial capital in USD (default: 10000)")
    bt.add_argument("--leverage", type=float, default=None,
                    help="Leverage multiplier. Position size = 2%% base × leverage. "
                         "(default: use config value)")
    bt.add_argument("--no-orderbook", action="store_true",
                    help="Disable order book features (TA-only, backward compat)")
    bt.add_argument("--orderbook-depth", type=int, default=None,
                    help="Order book depth levels (default: 20)")
    bt.add_argument("--regime", type=str, default=None,
                    choices=["strict", "loose", "off"],
                    help="Regime detection gate: strict (conf>0.7), loose (conf>0.5), off (skip)")
    bt.add_argument("--no-bybit-ob", action="store_true",
                    help="Disable Bybit OB data (use CCXT or TA-only fallback)")

    # ── train ─────────────────────────────────────────────────────────
    tr = sub.add_parser("train", help="Fetch data + train the LSTM model")
    tr.add_argument("--pair", type=str, default=None,
                    help="Trading pair (e.g. BTC/USDT)")
    tr.add_argument("--timeframe", type=str, default=None,
                    help="Candle interval (default: 1m)")
    tr.add_argument("--no-bybit-ob", action="store_true",
                    help="Disable Bybit OB data (use CCXT or TA-only fallback)")

    # ── predict ───────────────────────────────────────────────────────
    pr = sub.add_parser("predict", help="Predict direction for a pair")
    pr.add_argument("--pair", type=str, default=None,
                    help="Trading pair (e.g. BTC/USDT)")
    pr.add_argument("--timeframe", type=str, default=None,
                    help="Candle interval (default: 1m)")
    pr.add_argument("--model-path", type=str, default=None,
                    help="Path to model checkpoint (default: models/best.pt)")

    # ── fetch ─────────────────────────────────────────────────────────
    ft = sub.add_parser("fetch", help="Fetch and cache OHLCV data")
    ft.add_argument("--pair", type=str, default=None,
                    help="Trading pair (e.g. BTC/USDT)")
    ft.add_argument("--timeframe", type=str, default=None,
                    help="Candle interval (default: 1m)")

    # ── bybit-fetch ───────────────────────────────────────────────────
    bf = sub.add_parser("bybit-fetch", help="Download Bybit spot OB historical data")
    bf.add_argument("--symbol", type=str, default="BTCUSDT",
                    help="Bybit symbol (default: BTCUSDT)")
    bf.add_argument("--start", type=str, default="2025-04-29",
                    help="Start date YYYY-MM-DD (default: 2025-04-29)")
    bf.add_argument("--end", type=str, default=None,
                    help="End date YYYY-MM-DD (default: today)")
    bf.add_argument("--depth", type=int, default=20,
                    help="OB depth levels (default: 20)")
    bf.add_argument("--workers", type=int, default=3,
                    help="Parallel downloads (default: 3)")
    bf.add_argument("--dry-run", action="store_true",
                    help="List files and sizes without downloading")

    args = parser.parse_args()

    # ── apply overrides to config before handlers run ─────────────────
    if getattr(args, "pair", None):
        config.symbols = [args.pair]
    if getattr(args, "timeframe", None):
        config.timeframe = args.timeframe
    if getattr(args, "leverage", None) is not None:
        config.position_size_pct = 0.02 * args.leverage
    if getattr(args, "no_orderbook", False):
        config.orderbook_enabled = False
    if getattr(args, "orderbook_depth", None) is not None:
        config.orderbook_depth = args.orderbook_depth
    if getattr(args, "regime", None) is not None:
        config.regime_mode = args.regime
    if getattr(args, "no_bybit_ob", False):
        config.bybit_ob_enabled = False

    # ── build a namespace matching what the old handlers expect ───────
    ns = argparse.Namespace(
        symbol=getattr(args, "pair", None) or getattr(args, "symbol", None),
        symbols=[args.pair] if getattr(args, "pair", None) else config.symbols,
        timeframe=getattr(args, "timeframe", None),
        capital=getattr(args, "capital", 10000.0),
        model_path=getattr(args, "model_path", None),
        # bybit-fetch args
        start=getattr(args, "start", "2025-04-29"),
        end=getattr(args, "end", None),
        depth=getattr(args, "depth", 20),
        workers=getattr(args, "workers", 3),
        dry_run=getattr(args, "dry_run", False),
    )

    handlers = {
        "backtest": handle_backtest,
        "train": handle_train,
        "predict": handle_predict,
        "fetch": handle_fetch,
        "bybit-fetch": handle_bybit_fetch,
    }
    handlers[args.command](ns)


if __name__ == "__main__":
    main()
