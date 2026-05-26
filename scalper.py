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

    # ── train ─────────────────────────────────────────────────────────
    tr = sub.add_parser("train", help="Fetch data + train the LSTM model")
    tr.add_argument("--pair", type=str, default=None,
                    help="Trading pair (e.g. BTC/USDT)")
    tr.add_argument("--timeframe", type=str, default=None,
                    help="Candle interval (default: 1m)")

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

    args = parser.parse_args()

    # ── apply overrides to config before handlers run ─────────────────
    if args.pair:
        config.symbols = [args.pair]
    if args.timeframe:
        config.timeframe = args.timeframe
    if getattr(args, "leverage", None) is not None:
        config.position_size_pct = 0.02 * args.leverage

    # ── build a namespace matching what the old handlers expect ───────
    ns = argparse.Namespace(
        symbol=args.pair,
        symbols=[args.pair] if args.pair else config.symbols,
        timeframe=args.timeframe,
        capital=getattr(args, "capital", 10000.0),
        model_path=getattr(args, "model_path", None),
    )

    handlers = {
        "backtest": handle_backtest,
        "train": handle_train,
        "predict": handle_predict,
        "fetch": handle_fetch,
    }
    handlers[args.command](ns)


if __name__ == "__main__":
    main()
