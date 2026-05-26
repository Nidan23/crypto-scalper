# Crypto Scalper — Project Handover

> Generated 2026-05-26. Knowledge graph: 873 nodes, 1546 edges, 66 communities.

## Project Overview

Crypto scalping ML system — LSTM-based 1-minute BTC/USDT direction prediction. Goal: build a CLI that makes profitable backtests using TA + order book microstructure features.

**GitHub**: `github.com/Nidan23/crypto-scalper`, branch `main` (uncommitted changes as of this writing).

**Status**: OB features hurt accuracy (48.85% vs 55.68% TA-only). Model has no predictive power (AUC 0.495). Needs deeper architecture or better feature engineering.

---

## Architecture

```
scalper.py                          # CLI entry point
├── src/config.py                   # Config dataclass (singleton)
├── src/cli.py                      # CLI handlers: backtest, train, fetch, predict, bybit-fetch
├── src/data/
│   ├── fetcher.py                  # CCXT OHLCV fetcher with pagination + parquet cache
│   ├── features.py                 # TA feature engineering (27 features) + normalization + sequences
│   ├── pipeline.py                 # Orchestrator: fetch → features → OB merge → NaN fill → split → scale → seqs
│   ├── bybit_ob_loader.py          # Bybit L2 OB downloader (Playwright → API → JSONLines → parquet)
│   └── orderbook.py                # Live CCXT OrderBookFetcher (fallback)
├── src/features/
│   └── orderbook_features.py       # 12 OB features → 24 candle columns (mean+std)
├── src/model/
│   ├── architecture.py             # CryptoLSTM: 2-layer stacked LSTM + residual + BatchNorm + Sigmoid
│   ├── train.py                    # Training loop: pos_weight, early stopping, auto-val-split
│   └── predict.py                  # Inference helpers
├── src/strategy/
│   └── backtest.py                 # Walk-forward backtest, metrics, equity curve
└── tests/                          # 214 passed, 0 failed
```

**Data flow**: `fetch_ohlcv(paginated CCXT) → build_features(27 TA) → _merge_ob_features(Bybit cache) → ffill NaNs → 50/50 split → StandardScaler → create_sequences → train LSTM → backtest`

---

## Key Bugs Fixed

### Timezone mismatch
Bybit parquet timestamps are **naive**; OHLCV from CCXT is **UTC-aware**. Comparing them crashes pandas.
- `bybit_ob_loader.py:584`: `result.index.tz_localize("UTC")` after loading parquet
- `bybit_ob_loader.py:563-568`: Use naive dates for `pd.date_range` (filename matching)
- `orderbook_features.py:221-227`: Timezone normalization in `aggregate_to_candles()`
- `pipeline.py:85-87`: Localize parsed filename dates before comparing with tz-aware bounds

### CCXT pagination
Binance caps at **1000 candles per call**. `fetch_ohlcv(limit=90000)` silently returned 1000.
- `fetcher.py:110-160`: Pagination loop — start from `now - limit*60s`, fetch 1000-candle chunks, advance `since_ms` forward

### NaN from OB feature gaps
OB snapshots don't cover every candle → NaN columns after left-join.
- `pipeline.py:269-275`: `combined_features.ffill().fillna(0.0)` after concatenation

### Empty validation set
`val_split=0.0` crashed `StandardScaler.transform()` and `train_model`.
- `features.py:401-407`: Check `not df.empty` before transform
- `train.py:127-139`: Auto-carve 10% from training when `n_val == 0`

### `pd.to_datetime` drops UTC
- `fetcher.py:175`: Added `utc=True` parameter

---

## Current Config (`src/config.py`)

```python
lookback_candles = 21000         # ~14 days at 1m
train_split = 0.5                # 50% train
val_split = 0.0                  # 0% val (trainer carves 10%)
seq_len = 30                     # LSTM sequence length
hidden_dim = 128
num_layers = 2
dropout = 0.3
batch_size = 32
learning_rate = 0.001
num_epochs = 50
early_stopping_patience = 15
orderbook_enabled = True
orderbook_depth = 10             # keep ≤10 for 16 GB RAM
bybit_ob_enabled = True
bybit_ob_auto_sync = False       # manual sync to avoid Playwright hang
augmentation_enabled = True
augmentation_factor = 2
```

---

## Latest Results

| Metric | TA-Only (baseline) | TA + OB (current) |
|---|---|---|
| Accuracy | 55.68% | **48.85%** ↓ |
| AUC-ROC | 0.541 | 0.495 |
| Precision | — | 47.88% |
| Recall | — | 63.11% |
| F1 | — | 54.45% |
| Total Return | — | +1.14% |
| Sharpe | — | 0.23 |
| Win Rate | — | 40.7% |
| Trades | — | 4,519 |
| Features | 27 | 51 (27 TA + 24 OB) |

OB features made performance **worse**. Raw 100ms L2 microstructure is too noisy for a simple LSTM.

---

## Data

### Bybit OB Cache
- Path: `ob_cache/bybit/BTCUSDT_*.parquet`
- 30 days retained (June 2026), ~10 GB
- Download: `python scalper.py bybit-fetch --start YYYY-MM-DD --end YYYY-MM-DD --workers 4`
- Source: Bybit public history API (reverse-engineered, Playwright browser session required)
- Format: 200-level L2 snapshots at 100ms intervals

### OHLCV Cache
- Path: `data_cache/*.parquet`
- Key: `MD5(exchange:symbol:timeframe:limit)`
- Source: CCXT → Binance public API

### Tests
```bash
source .venv/bin/activate
python -m pytest tests/ -x -q
# 214 passed, 0 failed, 2 warnings
```

---

## ⚠️ Memory Warning

Loading 60 days of 20-level Bybit OB data uses **43-92 GB RAM** (swapping on 16 GB machines). Keep:
- `orderbook_depth ≤ 10`
- `lookback_candles ≤ 21000`

---

## CLI Commands

```bash
# Backtest
python scalper.py backtest --pair BTC/USDT --capital 100 --leverage 10

# TA-only (skip OB)
python scalper.py backtest --pair BTC/USDT --no-bybit-ob

# Download OB data
python scalper.py bybit-fetch --start 2026-05-01 --end 2026-05-25 --workers 4

# Dry run
python scalper.py bybit-fetch --dry-run
```

---

## Potential Next Steps

1. **Better OB features**: OFI (order flow imbalance), trade-sign features, volume-weighted micro-price
2. **Deeper model**: Transformer or TCN instead of LSTM
3. **Ensemble**: Gradient boosting on aggregated features
4. **Hyperparameter tuning**: Grid search seq_len, hidden_dim, learning_rate on TA-only first
5. **MPS GPU support**: Add `mps` device for Apple Silicon (LSTM on MPS has been flaky historically)
6. **Regime detection**: Already implemented but disabled (`regime_mode=off`) — made accuracy worse in earlier tests
