from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Exchange
    exchange_id: str = "binance"

    # Trading pairs — single-pair focus for accuracy. Add more pairs later.
    symbols: List[str] = field(default_factory=lambda: ["BTC/USDT"])

    # Timeframe for OHLCV
    timeframe: str = "1m"

    # Data
    lookback_candles: int = 50000  # candles to fetch (~35 days of 1m data)
    target_forward_periods: int = 1  # predict next candle
    augmentation_enabled: bool = True
    augmentation_factor: int = 2  # copies per original sequence
    augmentation_noise_std: float = 0.02  # std dev of Gaussian jitter
    train_split: float = 0.7
    val_split: float = 0.15  # test gets remainder

    # Features
    seq_len: int = 60  # sequence length for LSTM input
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    obv_period: int = 20
    ema_fast: int = 9
    ema_slow: int = 21
    roc_periods: List[int] = field(default_factory=lambda: [5, 10, 20])

    # Model
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    early_stopping_patience: int = 15

    # Trading
    confidence_threshold_long: float = 0.55
    confidence_threshold_short: float = 0.45
    position_size_pct: float = 0.20  # 20% — simulating 10x leverage on 2% base
    stop_loss_atr_mult: float = 2.0
    take_profit_atr_mult: float = 3.0
    max_hold_candles: int = 15  # 15 min on 1m candles

    # Paths
    model_dir: str = "models"
    data_dir: str = "data_cache"
    plot_dir: str = "plots"


config = Config()
