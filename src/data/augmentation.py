"""Time-series data augmentation for LSTM training sequences.

Augmentation is applied only to training data after sequence creation,
avoiding any leakage into validation/test splits.  Techniques:

* **Jittering**: Add small Gaussian noise to the normalised feature values.
  This forces the model to learn robust representations that are not
  dependent on exact feature values.

* **Magnitude scaling**: Multiply each sequence by a random factor close
  to 1.0, simulating slightly different volatility regimes.

Both techniques preserve the binary target (a small perturbation should
not change the next-candle direction).
"""

from __future__ import annotations

import numpy as np


def augment_sequences(
    X: np.ndarray,
    y: np.ndarray,
    factor: int = 2,
    noise_std: float = 0.02,
    scale_range: tuple[float, float] = (0.97, 1.03),
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create augmented copies of training sequences.

    Args:
        X: Array of shape ``(n_seq, seq_len, n_features)``.
        y: Array of shape ``(n_seq,)`` — binary targets.
        factor: Number of augmented copies per original sequence.
            The output will have ``n_seq * (factor + 1)`` entries
            (originals + *factor* augmented copies).
        noise_std: Standard deviation of Gaussian jitter added to each
            feature.  Features are assumed to be Z-scored (mean 0, std 1),
            so ``0.02`` means ~2 % of a standard deviation.
        scale_range: ``(low, high)`` for uniform magnitude scaling.
        seed: Optional RNG seed for reproducibility.

    Returns:
        ``(X_aug, y_aug)`` — concatenation of originals with augmented
        copies.  Targets are simply tiled; they are not modified.
    """
    if factor < 0:
        raise ValueError(f"factor must be >= 0, got {factor}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X ({X.shape[0]}) and y ({y.shape[0]}) must have the same "
            "first dimension"
        )

    rng = np.random.default_rng(seed)
    n_seq, seq_len, n_features = X.shape

    augmented_X: list[np.ndarray] = [X]
    augmented_y: list[np.ndarray] = [y]

    for _ in range(factor):
        # Independent noise per copy
        X_copy = X.copy().astype(np.float64)

        # Magnitude scaling: scale the whole per-sample sequence
        scales = rng.uniform(*scale_range, size=(n_seq, 1, 1))
        X_copy *= scales

        # Gaussian jitter
        jitter = rng.normal(0, noise_std, size=X_copy.shape)
        X_copy += jitter

        augmented_X.append(X_copy.astype(X.dtype))
        augmented_y.append(y.copy())

    return (
        np.concatenate(augmented_X, axis=0),
        np.concatenate(augmented_y, axis=0),
    )
