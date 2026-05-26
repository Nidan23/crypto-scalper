# Crypto Scalper v0.3 — Improvement Plan

## Order Book Microstructure + Regime Detection Gate

---

## Architecture

```
scalper.py CLI
  │
  ▼
PIPELINE ORCHESTRATOR (pipeline.py)
  │
  ├── OHLCV Fetch (existing)      ──► TA Features (27)
  ├── Order Book Fetch (NEW)      ──► OB Features (12)    ──► MERGE (39)
  │                                                              │
  └── Regime Detector (NEW) ◄── Regime Features (10, OHLCV)      │
           │                                                      │
           ├── NO_TRADE → skip, log reason                       │
           └── TRADE ────────────────────────────────────────────►
                    LSTM Direction Model → UP/DOWN → Strategy
```

**Key flow:** Raw data → features → regime gate filters → LSTM predicts → execute

---

## Phase 1: Order Book Microstructure (Core Signal)

### New files
| File | Purpose |
|---|---|
| `src/data/orderbook.py` | CCXT order book fetch + parquet caching |
| `src/features/orderbook_features.py` | 12 microstructure features |
| `tests/test_orderbook.py` | Fetch + cache tests |
| `tests/test_orderbook_features.py` | Feature computation tests |

### 12 Order Book Features
**Level 1 — Immediate liquidity (4)**
- `bid_ask_spread_pct` — (ask[0] - bid[0]) / mid × 100
- `depth_imbalance` — (bid_vol - ask_vol) / (bid_vol + ask_vol) at touch
- `bid_vol_0`, `ask_vol_0` — normalized by rolling median

**Level 2 — Multi-level depth (4)**
- `depth_imbalance_top5`
- `weighted_bid_price`, `weighted_ask_price` — deviation from mid
- `depth_slope_ratio` — volume decay rate bid vs ask

**Level 3 — Temporal (4)**
- `order_flow_delta` — net change in resting liquidity
- `quote_imbalance_roc` — rate of change over 3 snapshots
- `wall_density_bid`, `wall_density_ask`

**Snapshot interval:** 10 seconds → 6 snapshots per 1m candle → aggregate as mean + std

---

## Phase 2: Regime Detection Gate (Trade Filtering)

### New files
| File | Purpose |
|---|---|
| `src/features/regime_features.py` | 10 regime detection features |
| `src/model/regime_detector.py` | RF classifier: TRADE / NO_TRADE |
| `tests/test_regime_detector.py` | Training + inference tests |

### 10 Regime Features (OHLCV-only, no OB dependency)
**Volatility cluster (3):** `volatility_percentile`, `range_pct`, `volatility_ratio`
**Trend/Chop cluster (4):** `adx`, `chop_index`, `trend_strength`, `efficiency_ratio`
**Volume cluster (3):** `volume_percentile`, `volume_trend`, `spread_width_pct`

### Training strategy
- **Bootstrap labels:** Rule-based — NO_TRADE if extreme vol, choppy, low volume, or wide spread
- **Model:** Random Forest (100 estimators, max_depth=8) — fast, calibrated probs, interpretable
- **Iterative:** Backtest → review false positives/negatives → refine labels → retrain

---

## CLI Changes

```
python scalper.py backtest --pair BTC/USDT --regime strict --orderbook-depth 20

Regime: strict | Filtered: 312/1247 (25.0%) candles skipped
Regime reasons: extreme_vol(142) chop(98) low_volume(52) wide_spread(20)
TA features: 27 | OB features: 12 | Total: 39
Accuracy: 61.34% | Sharpe: 0.91 | Trades: 935
```

| Flag | Values | Default |
|---|---|---|
| `--regime` | strict, loose, off | off (backward compat) |
| `--orderbook-depth` | 1-50 | 20 |
| `--no-orderbook` | flag | false |

---

## Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| OB snapshot interval | 10s | 6 snapshots/candle, enough for mean+std, won't hit rate limits |
| OB aggregation | mean + std per candle | Captures average state AND intra-candle liquidity volatility |
| Regime model | Random Forest | Calibrated probs, interpretable importance, no GPU |
| Regime features | OHLCV-only | Works even without OB data; `spread_width_pct` as proxy |
| Regime labels | Rules → ML | Domain knowledge bootstrap, then let model learn nonlinear boundaries |
| Backward compat | `--regime off` | Existing behavior preserved |

---

## Implementation Order

```
Phase 1                          Phase 2
─────────────────                ─────────────────
1.1 Config + Fetch      ──►     2.1 Rule-based labels
1.2 OB Features         ──►     2.2 RF Classifier
1.3 Pipeline Merge      ──►     2.3 Pipeline Gate integration
1.4 Retrain + Eval      ──►     2.4 CLI + output
1.5 Tests                       2.5 Label refinement
                                2.6 Tests
```

**Target accuracy:** 55.68% → 61%+ (OB features add ~3%, regime filtering adds ~3%)
**Estimated effort:** 2-3 coding sessions (~8 new files, ~5 modified)
