"""End-to-end data pipeline for crypto scalping ML.

Orchestrates fetching, feature engineering, normalisation, and sequence
creation across one or more trading pairs.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import config
from src.data.augmentation import augment_sequences
from src.data.fetcher import fetch_multiple
from src.data.features import build_features, build_target, create_sequences, normalize_data


def run_pipeline(
    symbols: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Execute the full data pipeline.

    Steps
    -----
    1. Fetch OHLCV data for each symbol via :func:`~src.data.fetcher.fetch_multiple`.
    2. Build feature columns and binary targets per symbol.
    3. Concatenate all symbols' feature/target data (pair-agnostic).
    4. Time-series split into train / validation / test (no shuffle).
    5. Normalise using a StandardScaler fit on the training split only.
    6. Create sliding-window LSTM sequences.

    Args:
        symbols: List of trading pairs.  Defaults to ``config.symbols``.

    Returns:
        Dictionary with keys:

        - ``X_train``: ``np.ndarray``, shape ``(n, seq_len, n_features)``
        - ``y_train``: ``np.ndarray``, shape ``(n,)``
        - ``X_val``:   ``np.ndarray``, shape ``(n, seq_len, n_features)``
        - ``y_val``:   ``np.ndarray``, shape ``(n,)``
        - ``X_test``:  ``np.ndarray``, shape ``(n, seq_len, n_features)``
        - ``y_test``:  ``np.ndarray``, shape ``(n,)``
        - ``scaler``:  :class:`sklearn.preprocessing.StandardScaler` (fitted)
        - ``feature_names``: ``List[str]`` of feature column names

    Raises:
        RuntimeError: If any symbol produces insufficient data for sequence
            creation, or if the pipeline fails at any step.
    """
    if symbols is None:
        symbols = config.symbols

    # ------------------------------------------------------------------
    # 1. Fetch raw data
    # ------------------------------------------------------------------
    try:
        raw_data: Dict[str, pd.DataFrame] = fetch_multiple(symbols)
    except Exception as e:
        raise RuntimeError(f"Pipeline failed during data fetch: {e}") from e

    if not raw_data:
        raise RuntimeError("No data returned from fetch_multiple.")

    # ------------------------------------------------------------------
    # 2. Build features and targets per symbol
    # ------------------------------------------------------------------
    all_feature_frames: List[pd.DataFrame] = []
    all_target_series: List[pd.Series] = []

    for symbol in symbols:
        df = raw_data.get(symbol)
        if df is None or df.empty:
            raise RuntimeError(f"No OHLCV data available for symbol '{symbol}'.")

        features = build_features(df)
        targets = build_target(df, forward_periods=config.target_forward_periods)

        # Align features and targets: targets has NaN in the last N rows
        # (no future candles to compare).  Features have NaN rows dropped from
        # the front.  The intersection gives fully valid pairs.
        valid_idx = features.index.intersection(targets.dropna().index)
        if len(valid_idx) == 0:
            raise RuntimeError(
                f"Symbol '{symbol}': no aligned feature/target rows remain "
                "after NaN removal.  Increase lookback_candles or reduce "
                "indicator periods."
            )

        all_feature_frames.append(features.loc[valid_idx])
        all_target_series.append(targets.loc[valid_idx])

    # ------------------------------------------------------------------
    # 3. Concatenate all symbols (pair-agnostic)
    # ------------------------------------------------------------------
    combined_features: pd.DataFrame = pd.concat(all_feature_frames, axis=0)
    combined_targets: pd.Series = pd.concat(all_target_series, axis=0)

    # Reset index to get a clean positional split (index values from
    # different symbols may overlap).
    combined_features = combined_features.reset_index(drop=True)
    combined_targets = combined_targets.reset_index(drop=True)

    n_total = len(combined_features)
    if n_total == 0:
        raise RuntimeError("Combined feature set is empty after alignment.")

    # ------------------------------------------------------------------
    # 4. Time-series split (no shuffle)
    # ------------------------------------------------------------------
    train_end = int(n_total * config.train_split)
    val_end = train_end + int(n_total * config.val_split)

    train_df = combined_features.iloc[:train_end]
    val_df = combined_features.iloc[train_end:val_end]
    test_df = combined_features.iloc[val_end:]

    train_targets = combined_targets.iloc[:train_end]
    val_targets = combined_targets.iloc[train_end:val_end]
    test_targets = combined_targets.iloc[val_end:]

    # ------------------------------------------------------------------
    # 5. Normalize
    # ------------------------------------------------------------------
    scaled_train, scaled_val, scaled_test, scaler_tuple = normalize_data(
        train_df, val_df, test_df,
    )
    scaler, feature_names = scaler_tuple

    # ------------------------------------------------------------------
    # 6. Create sequences
    # ------------------------------------------------------------------
    seq_len = config.seq_len

    def _check_seq_viable(n_samples: int, label: str) -> None:
        if n_samples < seq_len + 1:
            raise RuntimeError(
                f"{label} split has only {n_samples} samples, but need at least "
                f"{seq_len + 1} to create one sequence (seq_len={seq_len}). "
                f"Consider increasing lookback_candles or reducing seq_len/train_split."
            )

    _check_seq_viable(len(scaled_train), "Train")
    X_train, y_train = create_sequences(scaled_train, train_targets.values, seq_len)

    # ------------------------------------------------------------------
    # 6a. Augment training data (optional, train split only — no leakage)
    # ------------------------------------------------------------------
    if getattr(config, "augmentation_enabled", False):
        factor = getattr(config, "augmentation_factor", 2)
        noise_std = getattr(config, "augmentation_noise_std", 0.02)
        X_train, y_train = augment_sequences(
            X_train, y_train,
            factor=factor,
            noise_std=noise_std,
        )

    if scaled_val is not None and len(scaled_val) > 0:
        _check_seq_viable(len(scaled_val), "Validation")
        X_val, y_val = create_sequences(scaled_val, val_targets.values, seq_len)
    else:
        X_val, y_val = np.array([]), np.array([])

    if scaled_test is not None and len(scaled_test) > 0:
        _check_seq_viable(len(scaled_test), "Test")
        X_test, y_test = create_sequences(scaled_test, test_targets.values, seq_len)
    else:
        X_test, y_test = np.array([]), np.array([])

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": feature_names,
    }
