"""Feature engineering package.

Re-exports TA feature builders from ``src.data.features`` and OB features
from ``src.features.orderbook_features``.
"""

from src.data.features import build_features, create_sequences
from src.features.orderbook_features import aggregate_to_candles, compute_ob_features

__all__ = [
    "build_features",
    "create_sequences",
    "compute_ob_features",
    "aggregate_to_candles",
]
