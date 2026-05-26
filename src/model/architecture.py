"""
Neural network architecture for cryptocurrency directional prediction.

Provides a stacked LSTM model with residual connections, batch normalisation,
and serialisation helpers that preserve scaler and feature metadata alongside
model weights in a single .pt file.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn


class CryptoLSTM(nn.Module):
    """Bidirectional-price-prediction LSTM with residual skip connection.

    Architecture (exactly two LSTM blocks):
        LSTM1 -> BatchNorm1d -> Dropout ->
        LSTM2 (+) -> BatchNorm1d -> Dropout ->
        Linear -> Sigmoid

    The residual connection adds the *full sequence output* of LSTM1 to the
    full sequence output of LSTM2 *before* the second batch-normalisation
    layer.  Both LSTMs output the same shape ``(seq_len, batch, hidden_dim)``
    so element-wise addition is well-defined.

    Parameters
    ----------
    input_dim : int
        Number of features per timestep (the last dimension of the input).
    hidden_dim : int, default 128
        Number of units in each LSTM layer.
    num_layers : int, default 2
        Reserved for API consistency.  The architecture always uses exactly
        two LSTM blocks with a residual connection between them.
    dropout : float, default 0.3
        Dropout probability applied after each batch-normalisation layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout

        # -- LSTM block 1 ---------------------------------------------------
        self.lstm1 = nn.LSTM(
            input_dim, hidden_dim, batch_first=False, dropout=0.0
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)

        # -- LSTM block 2 (residual from block 1) ---------------------------
        self.lstm2 = nn.LSTM(
            hidden_dim, hidden_dim, batch_first=False, dropout=0.0
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

        # -- Output head ----------------------------------------------------
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, seq_len, input_dim)``.

        Returns:
            Tensor of shape ``(batch, 1)`` with sigmoid-activated values
            in the range ``[0, 1]``.
        """
        # (batch, seq_len, input_dim) -> (seq_len, batch, input_dim)
        x = x.permute(1, 0, 2)

        # --- LSTM1 -> BN -> Dropout ----------------------------------------
        out1, _ = self.lstm1(x)                     # (S, B, H)
        out1 = out1.permute(1, 2, 0)                # (B, H, S)  for BN1d
        out1 = self.bn1(out1)
        out1 = out1.permute(2, 0, 1)                # (S, B, H)  back
        out1 = self.dropout1(out1)

        # --- LSTM2 -> residual + BN -> Dropout -----------------------------
        out2, _ = self.lstm2(out1)                  # (S, B, H)
        out2 = out2 + out1                          # residual connection
        out2 = out2.permute(1, 2, 0)                # (B, H, S)  for BN1d
        out2 = self.bn2(out2)
        out2 = out2.permute(2, 0, 1)                # (S, B, H)  back
        out2 = self.dropout2(out2)

        # --- Take last timestep -> Linear -> Sigmoid -----------------------
        out = out2[-1, :, :]                        # (B, H)
        out = self.fc(out)                          # (B, 1)
        out = self.sigmoid(out)
        return out


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

_CHECKPOINT_KEYS = frozenset(
    {"model_state_dict", "scaler", "feature_names"}
)


def save_model(
    model: nn.Module,
    path: str,
    scaler: object = None,
    feature_names: Optional[list[str]] = None,
) -> None:
    """Save model weights together with optional scaler and feature names.

    The saved file is a plain dictionary loaded by ``torch.load``:

    .. code-block:: python

        {
            "model_state_dict": ...,
            "scaler": ...,           # may be None
            "feature_names": ...,    # may be None
        }

    Args:
        model: The PyTorch model to persist (``model.state_dict()`` is called
            internally).
        path: Destination file path (``.pt`` extension recommended).
        scaler: Optional fitted scaler (e.g. ``sklearn.preprocessing.
            StandardScaler``).  Saved via pickle inside the checkpoint.
        feature_names: Optional list of feature column names corresponding
            to the scaler.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "scaler": scaler,
        "feature_names": feature_names,
    }
    torch.save(checkpoint, path)


def load_model(
    path: str,
    input_dim: int,
    device: str = "cpu",
) -> tuple:
    """Load model, scaler, and feature names from a checkpoint file.

    Args:
        path: Path to the ``.pt`` checkpoint created by :func:`save_model`.
        input_dim: Number of input features (the last dimension of the
            input tensor).  **Must be known**; use
            :func:`~src.model.predict.load_trained_model` for auto-detection.
        device: Device string (``"cpu"`` or ``"cuda"``).

    Returns:
        A tuple ``(model, scaler, feature_names)`` where *scaler* and
        *feature_names* are ``None`` if they were not saved.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"]

    # Infer hidden_dim from the stored weight shape.
    # weight_ih_l0 shape: (4 * hidden_dim, input_dim)
    hidden_dim = state["lstm1.weight_ih_l0"].shape[0] // 4

    model = CryptoLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=checkpoint.get("dropout", 0.3),
    )
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    scaler = checkpoint.get("scaler")
    feature_names = checkpoint.get("feature_names")

    return model, scaler, feature_names
