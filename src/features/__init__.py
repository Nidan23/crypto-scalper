"""Feature engineering package.

Re-exports TA feature builders from ``src.data.features``, OB features
from ``src.features.orderbook_features``, and regime features from
``src.features.regime_features``.
"""

from src.data.features import build_features, create_sequences
from src.features.orderbook_features import aggregate_to_candles, compute_ob_features
from src.features.regime_features import build_regime_features, create_bootstrap_labels

__all__ = [
    "build_features",
    "create_sequences",
    "compute_ob_features",
    "aggregate_to_candles",
    "build_regime_features",
    "create_bootstrap_labels",
]
