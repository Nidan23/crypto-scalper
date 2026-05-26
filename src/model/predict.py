"""
Inference helpers for the crypto scalping LSTM model.

Provides single-symbol and batch prediction interfaces that accept raw
OHLCV DataFrames, run the full feature-engineering → normalisation →
sequence-creation pipeline, and return directional probabilities with
confidence scores.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.config import config
from src.model.architecture import CryptoLSTM, load_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading with auto-detection of input_dim
# ---------------------------------------------------------------------------


def load_trained_model(
    model_path: str,
    input_dim: Optional[int] = None,
    device: str = "cpu",
) -> Tuple[CryptoLSTM, object, list]:
    """Load a trained model, auto-detecting ``input_dim`` if not provided.

    This is a convenience wrapper around :func:`load_model` that peeks at
    the saved state dict to determine ``input_dim`` when the caller does
    not know it ahead of time.

    Args:
        model_path: Path to a ``.pt`` checkpoint created by
            :func:`~src.model.architecture.save_model`.
        input_dim: Number of input features.  If ``None``, inferred from
            the first LSTM layer's learned weight shape.
        device: Target device (``"cpu"`` or ``"cuda"``).

    Returns:
        Tuple ``(model, scaler, feature_names)``.
    """
    if input_dim is None:
        # Lightweight peek: load only the state dict keys we need.
        checkpoint = torch.load(
            model_path, map_location="cpu", weights_only=False
        )
        state = checkpoint["model_state_dict"]
        # lstm1.weight_ih_l0 shape: (4 * hidden_dim, input_dim)
        input_dim = state["lstm1.weight_ih_l0"].shape[1]
        # Avoid holding the full checkpoint in memory during the second load.
        del checkpoint

    return load_model(model_path, input_dim, device=device)


# ---------------------------------------------------------------------------
# Single-symbol prediction
# ---------------------------------------------------------------------------


def predict_single(
    model: CryptoLSTM,
    scaler: object,
    ohlcv_df: pd.DataFrame,
    device: str = "cpu",
) -> Tuple[float, float]:
    """Predict direction for a single symbol from raw OHLCV data.

    The function runs the full inference pipeline:
        ``build_features`` → ``normalise (via scaler)`` → ``create_sequences``
    → ``forward pass``

    Args:
        model: Trained ``CryptoLSTM`` model instance.
        scaler: Fitted sklearn-like scaler (must have a ``transform`` method).
        ohlcv_df: Raw OHLCV DataFrame with at least ``open``, ``high``,
            ``low``, ``close``, ``volume`` columns.
        device: Torch device string.

    Returns:
        Tuple ``(direction_probability, confidence)`` where:

        * ``direction_probability`` is the raw sigmoid output in ``[0, 1]``.
           Values > 0.5 indicate a long (up) bias; < 0.5 indicates short.
        * ``confidence`` is the scaled distance from 0.5 in ``[0, 1]``
           (e.g. probability 0.8 → confidence 0.6 toward long).
    """
    # Lazy imports so these module-level dependencies are not required at
    # import time (the features module may not exist yet).
    from src.features import build_features, create_sequences

    # --- 1. Feature engineering -------------------------------------------
    features_df = build_features(ohlcv_df)
    if features_df.empty:
        logger.warning("build_features returned empty DataFrame — returning "
                       "neutral prediction.")
        return 0.5, 0.0

    # --- 2. Normalisation -------------------------------------------------
    # Drop any target column if it leaked through.
    feature_cols = [
        c for c in features_df.columns if c.lower() != "target"
    ]
    if not feature_cols:
        logger.warning("No feature columns found after build_features — "
                       "returning neutral prediction.")
        return 0.5, 0.0

    X_raw = features_df[feature_cols].values.astype(np.float64)
    X_scaled: np.ndarray = scaler.transform(X_raw)  # type: ignore[union-attr]

    # --- 3. Ensure minimum length for sequence creation -------------------
    if X_scaled.shape[0] < config.seq_len:
        # Pad at the start by repeating the first observation (edge padding).
        n_pad = config.seq_len - X_scaled.shape[0]
        pad = np.repeat(X_scaled[:1], n_pad, axis=0)
        X_scaled = np.concatenate([pad, X_scaled], axis=0)
        logger.debug("Padded feature array by %d rows (edge replication) "
                     "to meet seq_len=%d", n_pad, config.seq_len)

    # --- 4. Create sequences ----------------------------------------------
    # create_sequences is expected to return (X_seq, y_seq) — we ignore y.
    X_seq, _ = create_sequences(
        X_scaled,
        np.zeros(X_scaled.shape[0], dtype=np.float32),
        config.seq_len,
    )

    if X_seq.shape[0] == 0:
        logger.warning("create_sequences produced no windows — returning "
                       "neutral prediction.")
        return 0.5, 0.0

    # --- 5. Forward pass --------------------------------------------------
    # Use the most recent (last) window.
    X_input = X_seq[-1:]  # (1, seq_len, input_dim)

    model.eval()
    with torch.no_grad():
        tensor = torch.FloatTensor(X_input).to(device)
        prob = model(tensor).item()

    # --- 6. Confidence ----------------------------------------------------
    confidence: float = abs(prob - 0.5) * 2.0  # map [0, 0.5] → [0, 1]

    return prob, confidence


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------


def predict_batch(
    model: CryptoLSTM,
    scaler: object,
    ohlcv_dict: Dict[str, pd.DataFrame],
    device: str = "cpu",
) -> Dict[str, Tuple[float, float]]:
    """Run prediction for multiple symbols.

    Args:
        model: Trained ``CryptoLSTM`` model instance.
        scaler: Fitted sklearn-like scaler.
        ohlcv_dict: Mapping of symbol name (e.g. ``"BTC/USDT"``) to its
            raw OHLCV DataFrame.
        device: Torch device string.

    Returns:
        Dictionary mapping each symbol to
        ``(direction_probability, confidence)``.
    """
    results: Dict[str, Tuple[float, float]] = {}
    for symbol, df in ohlcv_dict.items():
        try:
            prob, conf = predict_single(model, scaler, df, device=device)
            results[symbol] = (prob, conf)
        except Exception:
            logger.exception(
                "Prediction failed for symbol %s — returning neutral", symbol
            )
            results[symbol] = (0.5, 0.0)

    return results
