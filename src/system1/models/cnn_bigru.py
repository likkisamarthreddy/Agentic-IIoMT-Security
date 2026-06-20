# -*- coding: utf-8 -*-
"""
CNN-BiGRU Hybrid Classifier
============================

A lightweight hybrid deep-learning model combining 1-D convolutional
feature extraction with bidirectional GRU temporal modelling for
real-time IIoMT traffic classification.

Architecture (default hyper-parameters from ``config/settings.yaml``)::

    Input  (batch, seq_len, num_features)
      → permute → Conv1d(num_features→64, k=3, pad=1) → BN → ReLU → Dropout
      → Conv1d(64→128, k=3, pad=1) → BN → ReLU → Dropout
      → 2-layer BiGRU (hidden=64, dropout=0.3)
      → Attention pooling over the sequence
      → Linear(128→64) → ReLU → Dropout(0.3)
      → Linear(64→num_classes)

Typical edge-device footprint after INT8 quantisation: < 15 MB.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "settings.yaml"


def _load_config(config_path: Path = _DEFAULT_CONFIG) -> Dict[str, Any]:
    """Load and return the ``system1`` section of *settings.yaml*.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary with the ``system1`` configuration block.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        KeyError: If the ``system1`` key is missing.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


# ---------------------------------------------------------------------------
# CNN-BiGRU Model
# ---------------------------------------------------------------------------


class CNNBiGRU(nn.Module):
    """Hybrid CNN–BiGRU network for multi-class traffic classification.

    The model first applies two 1-D convolutional blocks to extract
    local spatial features, then feeds the resulting sequence into a
    bidirectional GRU to capture temporal dependencies.  A two-layer
    fully-connected classifier head produces the final logits.

    Args:
        num_features: Number of input features per sample.
        num_classes: Number of output classes.
        config_path: Path to ``settings.yaml``.  When *None*, the
            default project-root config is used.

    Example::

        model = CNNBiGRU(num_features=46, num_classes=6)
        x = torch.randn(32, 1, 46)
        logits = model(x)          # (32, 6)
        scores = model.get_anomaly_score(x)  # (32,)
    """

    def __init__(
        self,
        num_features: int = 46,
        num_classes: int = 6,
        config_path: Optional[Path] = None,
    ) -> None:
        super().__init__()

        cfg = _load_config(config_path or _DEFAULT_CONFIG)
        s1 = cfg["system1"]["model"]

        self.num_features = num_features
        self.num_classes = num_classes

        # Quantization Stubs
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
        # Stubs to isolate the GRU in FP32 during static quantization
        self.dequant_gru = torch.quantization.DeQuantStub()
        self.quant_gru = torch.quantization.QuantStub()

        # ---- Hyper-parameters from config --------------------------------
        conv_filters: List[int] = s1.get("conv_filters", [64, 128])
        kernel_size: int = s1.get("conv_kernel_size", 3)
        gru_hidden: int = s1.get("gru_hidden_size", 64)
        gru_layers: int = s1.get("gru_num_layers", 2)
        gru_dropout: float = s1.get("gru_dropout", 0.3)
        cnn_dropout: float = s1.get("cnn_dropout", 0.25)
        fc_hidden: int = s1.get("fc_hidden", 64)
        fc_dropout: float = s1.get("fc_dropout", 0.3)

        # ---- CNN Feature Extractor ---------------------------------------
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(
                in_channels=num_features,
                out_channels=conv_filters[0],
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(conv_filters[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(cnn_dropout),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(
                in_channels=conv_filters[0],
                out_channels=conv_filters[1],
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(conv_filters[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(cnn_dropout),
        )

        # ---- BiGRU Temporal Encoder --------------------------------------
        self.gru = nn.GRU(
            input_size=conv_filters[1],
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
        )

        # BiGRU output dimension = 2 * gru_hidden (fwd + bwd)
        gru_output_size = gru_hidden * 2
        
        # ---- Attention Layer ---------------------------------------------
        self.attention = nn.Linear(gru_output_size, 1)

        # ---- Classifier Head ---------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(gru_output_size, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_hidden, num_classes),
        )

        logger.info(
            "CNNBiGRU initialised — features=%d, classes=%d, "
            "conv_filters=%s, gru_hidden=%d, params=%s",
            num_features,
            num_classes,
            conv_filters,
            gru_hidden,
            f"{self.count_parameters():,}",
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the network.

        Args:
            x: Input tensor of shape ``(batch, 1, num_features)``.

        Returns:
            Logits tensor of shape ``(batch, num_classes)``.
        """
        x = self.quant(x)

        # Input: (B, Seq, Features)
        # Transpose for CNN: (B, Features, Seq)
        out = x.permute(0, 2, 1)
        out = self.conv_block1(out)
        out = self.conv_block2(out)

        # Transpose for GRU: (B, Channels, Seq) → (B, Seq, Channels)
        out = out.permute(0, 2, 1)

        # Dequantise back to FP32 because the GRU will not be statically quantised
        out = self.dequant_gru(out)

        # BiGRU: (B, Seq, Channels) → (B, Seq, 2*H)
        gru_out, _ = self.gru(out)
        
        # Re-quantise back to INT8 for the attention and classifier heads
        gru_out = self.quant_gru(gru_out)

        # Attention scoring
        attn_weights = torch.softmax(self.attention(gru_out), dim=1)
        context = (attn_weights * gru_out).sum(dim=1)  # (B, 2*H)

        # Classifier
        logits = self.classifier(context)  # (B, num_classes)
        
        logits = self.dequant(logits)
        return logits

    # ------------------------------------------------------------------
    # Anomaly scoring
    # ------------------------------------------------------------------

    def get_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute an anomaly score as ``1 − max(softmax(logits))``.

        Higher values indicate more anomalous inputs (lower classifier
        confidence).

        Args:
            x: Input tensor of shape ``(batch, 1, num_features)``.

        Returns:
            Tensor of shape ``(batch,)`` with anomaly scores in [0, 1].
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            max_probs, _ = torch.max(probabilities, dim=1)
            anomaly_scores = 1.0 - max_probs
        return anomaly_scores

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_summary(self) -> Dict[str, Any]:
        """Return a summary dict with parameter count and estimated size.

        The estimated size is computed as
        ``total_params × 4 bytes`` (FP32) converted to megabytes.

        Returns:
            Dictionary with keys ``total_params``, ``trainable_params``,
            ``non_trainable_params``, ``estimated_size_mb``,
            and ``layer_details``.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable = total - trainable

        # Estimated FP32 model size in MB
        estimated_mb = (total * 4) / (1024 ** 2)

        layer_details: List[Dict[str, Any]] = []
        for name, param in self.named_parameters():
            layer_details.append(
                {
                    "name": name,
                    "shape": list(param.shape),
                    "params": param.numel(),
                    "requires_grad": param.requires_grad,
                }
            )

        summary = {
            "total_params": total,
            "trainable_params": trainable,
            "non_trainable_params": non_trainable,
            "estimated_size_mb": round(estimated_mb, 4),
            "layer_details": layer_details,
        }

        logger.info(
            "Model summary — total=%s, trainable=%s, size≈%.4f MB",
            f"{total:,}",
            f"{trainable:,}",
            estimated_mb,
        )
        return summary

    def get_estimated_size_mb(self) -> float:
        """Return the estimated FP32 model size in megabytes.

        Returns:
            Model size in MB (FP32, 4 bytes per parameter).
        """
        total_params = sum(p.numel() for p in self.parameters())
        size_mb = (total_params * 4) / (1024 ** 2)
        return round(size_mb, 4)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    model = CNNBiGRU(num_features=46, num_classes=6)
    summary = model.model_summary()
    print(f"\n{'='*60}")
    print(f"Total parameters : {summary['total_params']:,}")
    print(f"Trainable params : {summary['trainable_params']:,}")
    print(f"Estimated size   : {summary['estimated_size_mb']:.4f} MB")
    print(f"{'='*60}")

    # Dummy forward pass
    dummy = torch.randn(4, 1, 46)
    logits = model(dummy)
    print(f"Logits shape     : {logits.shape}")

    scores = model.get_anomaly_score(dummy)
    print(f"Anomaly scores   : {scores}")
