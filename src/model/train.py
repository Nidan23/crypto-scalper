"""
Training loop, metric computation, and history visualisation for the
crypto scalping LSTM model.

Handles class-imbalanced binary classification via ``pos_weight`` in the
BCE loss, early stopping with configurable patience, and automatic
checkpointing of the best model state to disk.
"""

from __future__ import annotations

import logging
import os
import warnings
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from src.config import config
from src.model.architecture import CryptoLSTM, save_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute classification metrics for binary predictions.

    Args:
        y_true: Ground-truth binary labels (0/1).
        y_pred: Predicted probabilities in ``[0, 1]`` (sigmoid outputs).

    Returns:
        Dictionary with keys ``accuracy``, ``precision``, ``recall``, ``f1``,
        and ``auc``.  AUC defaults to 0.0 when only one class is present in
        the ground truth.
    """
    y_pred_binary = (y_pred >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred_binary)
    prec = precision_score(y_true, y_pred_binary, zero_division=0.0)
    rec = recall_score(y_true, y_pred_binary, zero_division=0.0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0.0)

    try:
        auc = roc_auc_score(y_true, y_pred)
        if np.isnan(auc):
            auc = 0.0  # Only one class present in y_true
    except (ValueError, NotImplementedError):
        auc = 0.0  # Only one class present in y_true

    return {
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
        "auc": round(auc, 6),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(
    pipeline_data: dict,
    input_dim: Optional[int] = None,
) -> tuple[CryptoLSTM, dict]:
    """Train the ``CryptoLSTM`` model using a pre-built data pipeline.

    The pipeline dictionary **must** contain the following NumPy arrays:

    * ``X_train`` -- shape ``(n_train, seq_len, n_features)``
    * ``y_train`` -- shape ``(n_train,)``, binary labels (0/1)
    * ``X_val``   -- shape ``(n_val,   seq_len, n_features)``
    * ``y_val``   -- shape ``(n_val,)``, binary labels (0/1)

    Early stopping saves the best model (lowest validation loss) to
    ``<config.model_dir>/<timestamp>_model.pt`` and also returns it.

    Args:
        pipeline_data: Dictionary from the data pipeline.
        input_dim: Number of features.  If ``None``, inferred from
            ``X_train.shape[2]``.

    Returns:
        Tuple of ``(trained_model, history_dict)``.  The history dict
        contains one list per epoch for: ``train_loss``, ``val_loss``,
        ``val_accuracy``, ``val_precision``, ``val_recall``, ``val_f1``,
        and ``val_auc``.
    """
    # ------------------------------------------------------------------
    # Extract & validate
    # ------------------------------------------------------------------
    X_train: np.ndarray = pipeline_data["X_train"]
    y_train: np.ndarray = pipeline_data["y_train"]
    X_val: np.ndarray = pipeline_data["X_val"]
    y_val: np.ndarray = pipeline_data["y_val"]

    if input_dim is None:
        input_dim = X_train.shape[2]

    n_train = len(X_train)
    n_val = len(X_val)

    if n_train == 0:
        raise ValueError("Training data is empty: n_train=0")

    # If no explicit validation set, carve 10% from training data.
    if n_val == 0:
        split_idx = int(n_train * 0.9)
        X_val = X_train[split_idx:]
        y_val = y_train[split_idx:]
        X_train = X_train[:split_idx]
        y_train = y_train[:split_idx]
        n_train = len(X_train)
        n_val = len(X_val)
        logger.info(
            "No validation set provided — using last %d training samples "
            "(%.0f%%) for validation.",
            n_val, 100 * n_val / (n_train + n_val),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "Training on %s — %d train / %d val samples, input_dim=%d, "
        "seq_len=%d",
        device,
        n_train,
        n_val,
        input_dim,
        config.seq_len,
    )

    # ------------------------------------------------------------------
    # Class-balanced pos_weight
    # ------------------------------------------------------------------
    n_pos = int(y_train.sum())
    n_neg = n_train - n_pos

    if n_pos == 0:
        logger.warning("No positive samples in training set — using "
                       "pos_weight=1.0")
        pos_weight_value = 1.0
    elif n_neg == 0:
        logger.warning("No negative samples in training set — using "
                       "pos_weight=1.0")
        pos_weight_value = 1.0
    else:
        pos_weight_value = n_neg / n_pos

    logger.info(
        "Class balance — positives=%d, negatives=%d, pos_weight=%.4f",
        n_pos, n_neg, pos_weight_value,
    )

    pos_weight = torch.FloatTensor([pos_weight_value]).to(device)

    # ------------------------------------------------------------------
    # Model, loss, optimizer
    # ------------------------------------------------------------------
    model = CryptoLSTM(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)

    criterion = nn.BCELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # ------------------------------------------------------------------
    # DataLoaders (no shuffle — time series)
    # ------------------------------------------------------------------
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train).unsqueeze(1),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(config.batch_size, n_train),
        shuffle=False,
    )

    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val).unsqueeze(1),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(config.batch_size, n_val),
        shuffle=False,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    history: dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_auc": [],
    }

    best_val_loss = float("inf")
    best_model_state: Optional[dict[str, torch.Tensor]] = None
    patience_counter = 0

    for epoch in range(1, config.num_epochs + 1):
        # -- Train ----------------------------------------------------------
        model.train()
        train_loss_sum = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            # Apply class weighting manually (pos_weight * loss on positives)
            weighted_loss = _apply_pos_weight(loss, outputs, y_batch, pos_weight)
            weighted_loss.backward()
            optimizer.step()
            train_loss_sum += weighted_loss.item() * X_batch.size(0)

        avg_train_loss = train_loss_sum / n_train

        # -- Validate -------------------------------------------------------
        model.eval()
        val_loss_sum = 0.0
        all_val_preds: list[np.ndarray] = []
        all_val_true: list[np.ndarray] = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                weighted_loss_val = _apply_pos_weight(
                    loss, outputs, y_batch, pos_weight
                )
                val_loss_sum += weighted_loss_val.item() * X_batch.size(0)
                all_val_preds.append(outputs.cpu().numpy())
                all_val_true.append(y_batch.cpu().numpy())

        avg_val_loss = val_loss_sum / n_val

        y_pred_concat = np.concatenate(all_val_preds).flatten()
        y_true_concat = np.concatenate(all_val_true).flatten()
        metrics = compute_metrics(y_true_concat, y_pred_concat)

        # -- Record history -------------------------------------------------
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_accuracy"].append(metrics["accuracy"])
        history["val_precision"].append(metrics["precision"])
        history["val_recall"].append(metrics["recall"])
        history["val_f1"].append(metrics["f1"])
        history["val_auc"].append(metrics["auc"])

        logger.info(
            "Epoch %3d/%d — train_loss=%.6f  val_loss=%.6f  "
            "acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f  auc=%.4f",
            epoch,
            config.num_epochs,
            avg_train_loss,
            avg_val_loss,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["auc"],
        )

        # -- Early stopping / checkpoint ------------------------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {
                k: v.clone().cpu() for k, v in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            logger.debug(
                "Validation loss did not improve for %d epoch(s)",
                patience_counter,
            )
            if patience_counter >= config.early_stopping_patience:
                logger.info(
                    "Early stopping triggered after %d epochs "
                    "(patience=%d)",
                    epoch,
                    config.early_stopping_patience,
                )
                break

    # ------------------------------------------------------------------
    # Restore best model and save
    # ------------------------------------------------------------------
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(config.model_dir, f"{timestamp}_model.pt")
    save_model(
        model,
        model_path,
        scaler=pipeline_data.get("scaler"),
        feature_names=pipeline_data.get("feature_names"),
    )
    logger.info("Best model saved to %s", model_path)

    return model, history


def _apply_pos_weight(
    loss: torch.Tensor,
    outputs: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    """Weight the per-sample BCE loss by ``pos_weight`` for positive targets.

    ``nn.BCELoss`` does not natively support a ``pos_weight`` argument
    (unlike ``BCEWithLogitsLoss``), so we apply the weighting manually:
    scale the loss of positive samples by *pos_weight*.

    Args:
        loss: Per-sample loss tensor (same shape as ``outputs``), produced
            by ``BCELoss(reduction='none')``.
        outputs: Model predictions (sigmoid probabilities).
        targets: Ground-truth labels (0/1).

    Returns:
        Weighted scalar loss (0-d tensor, mean over the batch).
    """
    if pos_weight.item() == 1.0:
        return loss.mean()

    # Expand pos_weight to match the loss shape
    weight = torch.where(
        targets > 0.5,
        pos_weight,
        torch.ones_like(pos_weight),
    )
    weighted = loss * weight
    return weighted.mean()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_training_history(
    history: dict,
    save_path: Optional[str] = None,
) -> None:
    """Plot training/validation loss and validation metrics over epochs.

    Produces a two-panel figure:
    * Left: train and validation loss.
    * Right: validation accuracy, precision, recall, F1, and AUC.

    Args:
        history: Dictionary returned by :func:`train_model`.  Must contain
            at least ``train_loss`` and ``val_loss`` keys.
        save_path: If provided, the figure is saved to this path.  The
            parent directory is created if it does not exist.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        logger.warning("matplotlib is not available — skipping plot: %s", exc)
        return

    epochs = range(1, len(history.get("train_loss", [])) + 1)
    if not epochs:
        logger.warning("History is empty — nothing to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # -- Loss ---------------------------------------------------------------
    if "train_loss" in history and history["train_loss"]:
        axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    if "val_loss" in history and history["val_loss"]:
        axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss over Epochs")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # -- Metrics ------------------------------------------------------------
    metric_keys = [
        ("val_accuracy", "Accuracy"),
        ("val_precision", "Precision"),
        ("val_recall", "Recall"),
        ("val_f1", "F1"),
        ("val_auc", "AUC"),
    ]
    for key, label in metric_keys:
        values = history.get(key, [])
        if values:
            axes[1].plot(epochs[: len(values)], values, label=label)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation Metrics over Epochs")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.05, 1.05)

    plt.tight_layout()

    if save_path:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Training history plot saved to %s", save_path)

    plt.close(fig)
