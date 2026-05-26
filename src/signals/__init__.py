"""
Signal sources for trade generation.

This module provides a pluggable architecture for combining multiple
signal sources (technical ML predictions, news sentiment, etc.)
into unified trade signals.

To add a new signal source:
1. Subclass SignalSource and implement generate()
2. Register it with SignalCombiner
3. Signals are automatically weighted and combined
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import pandas as pd


class SignalSource(ABC):
    """Abstract base for any signal generator (ML model, news NLP, etc.)."""

    def __init__(self, weight: float = 1.0, name: Optional[str] = None):
        self.weight = weight
        self.name = name or self.__class__.__name__

    @abstractmethod
    def generate(self, ohlcv: pd.DataFrame) -> pd.Series:
        """
        Generate signals from OHLCV data.

        Args:
            ohlcv: DataFrame with columns ['open','high','low','close','volume']

        Returns:
            Series with same index, values in [-1, 1] where
            +1 = strong long, -1 = strong short, 0 = neutral
        """
        ...


class SignalCombiner:
    """
    Combines multiple signal sources with weights.

    Usage:
        combiner = SignalCombiner()
        combiner.add_source(ml_model_source, weight=0.7)
        combiner.add_source(news_sentiment_source, weight=0.3)  # future
        combined = combiner.combine(ohlcv)
    """

    def __init__(self):
        self._sources: Dict[str, SignalSource] = {}

    def add_source(
        self, source: SignalSource, weight: Optional[float] = None
    ) -> None:
        """Register a signal source with an optional override weight."""
        if weight is not None:
            source.weight = weight
        self._sources[source.name] = source

    def remove_source(self, name: str) -> None:
        """Unregister a signal source by name."""
        self._sources.pop(name, None)

    def combine(self, ohlcv: pd.DataFrame) -> pd.Series:
        """
        Produce a weighted ensemble signal from all registered sources.

        Args:
            ohlcv: DataFrame with columns ['open','high','low','close','volume']

        Returns:
            Series with values clipped to [-1, 1].

        Raises:
            ValueError: If no signal sources are registered.
        """
        if not self._sources:
            raise ValueError("No signal sources registered")

        signals = []
        weights = []
        for src in self._sources.values():
            sig = src.generate(ohlcv)
            signals.append(sig)
            weights.append(src.weight)

        total_weight = sum(weights)
        if total_weight == 0.0:
            return pd.Series(0.0, index=ohlcv.index)

        combined = sum(s * w / total_weight for s, w in zip(signals, weights))
        return combined.clip(-1, 1)
