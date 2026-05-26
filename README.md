# Crypto Scalper

Deep-learning-driven crypto scalping system using LSTM-based price prediction with multi-source signal combination.

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Fetch market data
python -m src.cli fetch

# Train the model
python -m src.cli train

# Backtest strategy
python -m src.cli backtest

# Generate live predictions
python -m src.cli predict
```

## Architecture

```
┌──────────────┐    ┌──────────────────┐    ┌────────────────┐
│   Exchange   │    │   Data Pipeline   │    │  Feature Eng   │
│  (ccxt)      │───>│  fetch / cache    │───>│ RSI, MACD, BB, │
│              │    │  normalize / split│    │  ATR, seq      │
└──────────────┘    └──────────────────┘    └────────────────┘
                                                    │
                                                    ▼
┌──────────────┐    ┌──────────────────┐    ┌────────────────┐
│  Backtest /  │<───│  Signal Combiner  │<───│   LSTM Model   │
│  Live Trade  │    │  (weighted blend) │    │  (PyTorch)     │
└──────────────┘    └──────────────────┘    └────────────────┘
                           │
                           ▼
                    ┌──────────────────┐
                    │   News NLP       │  (future)
                    │   Sentiment      │
                    └──────────────────┘
```

## Project Structure

```
src/
├── config.py        # Global configuration (pair-agnostic)
├── data/            # Data fetching, caching, preprocessing
├── features/        # Technical indicator computation
├── model/           # LSTM architecture, training loop
├── signals/         # SignalSource ABC + SignalCombiner
├── strategy/        # Entry/exit logic, risk management
└── cli.py           # Command-line interface

tests/
├── test_config.py
├── test_signals.py
└── ...
```

## Pair Agnostic Design

All configuration is symbol-independent. The system works with any CCXT-supported trading pair — BTC/USDT, ETH/GBP, SOL/EUR, etc. — by configuring the `symbols` list in `src/config.py`.

## Roadmap

- [x] Foundation: data pipeline, LSTM model, config, signal architecture
- [ ] News sentiment signal source (NLP-based)
- [ ] Real-time paper trading via WebSocket
- [ ] Reinforcement learning agent for entry/exit optimization
- [ ] Portfolio-level position sizing across multiple symbols
