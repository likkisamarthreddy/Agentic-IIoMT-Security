# -*- coding: utf-8 -*-
"""
Channel Pruning
================

L1-structured channel pruning for Conv1d layers in the CNN-BiGRU
classifier.  Pruning reduces computational cost and model size
before (or after) INT8 quantization.

Default sparsity and norm settings are loaded from
``system1.pruning`` in ``config/settings.yaml``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "settings.yaml"


def _load_config(config_path: Path = _DEFAULT_CONFIG) -> Dict[str, Any]:
    """Load the full project configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Channel Pruner
# ---------------------------------------------------------------------------


class ChannelPruner:
    """L1-structured channel pruner for ``nn.Conv1d`` layers.

    Applies PyTorch's ``ln_structured`` pruning (L1 norm by default)
    on the output-channel dimension of every Conv1d layer in a model.

    Args:
        amount: Fraction of channels to prune (0–1).  If *None*, read
            from ``system1.pruning.sparsity`` in config.
        pruning_norm: Lp norm used to rank channels.  Default ``1``
            (L1 norm).
        config_path: Path to ``settings.yaml``.
    """

    def __init__(
        self,
        amount: Optional[float] = None,
        pruning_norm: Optional[int] = None,
        config_path: Optional[Path] = None,
    ) -> None:
        cfg = _load_config(config_path or _DEFAULT_CONFIG)
        s1p = cfg.get("system1", {}).get("pruning", {})

        self.amount: float = amount if amount is not None else s1p.get("sparsity", 0.3)
        self.pruning_norm: int = pruning_norm if pruning_norm is not None else s1p.get("pruning_norm", 1)

        logger.info(
            "ChannelPruner initialised — amount=%.2f, L%d norm",
            self.amount,
            self.pruning_norm,
        )

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def prune_conv_layers(self, model: nn.Module, amount: Optional[float] = None) -> nn.Module:
        """Apply L1-structured pruning to all ``Conv1d`` layers.

        Args:
            model: The PyTorch model to prune (modified **in-place**).
            amount: Override for the pruning fraction.  Falls back to
                ``self.amount``.

        Returns:
            The pruned model (same object).
        """
        prune_amount = amount if amount is not None else self.amount
        pruned_count = 0

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv1d):
                prune.ln_structured(
                    module,
                    name="weight",
                    amount=prune_amount,
                    n=self.pruning_norm,
                    dim=0,  # output channels
                )
                pruned_count += 1
                logger.debug(
                    "Pruned Conv1d layer '%s' — amount=%.2f",
                    name,
                    prune_amount,
                )

        logger.info(
            "Structured pruning applied to %d Conv1d layers (amount=%.2f)",
            pruned_count,
            prune_amount,
        )
        return model

    # ------------------------------------------------------------------
    # Make permanent
    # ------------------------------------------------------------------

    def make_permanent(self, model: nn.Module) -> nn.Module:
        """Remove pruning re-parametrisations and apply masks permanently.

        After calling this method the model's ``Conv1d`` weight tensors
        are plain parameters again (no ``weight_orig`` / ``weight_mask``).

        Args:
            model: Previously pruned model (modified **in-place**).

        Returns:
            The model with permanent pruning.
        """
        made_permanent = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv1d):
                try:
                    prune.remove(module, "weight")
                    made_permanent += 1
                    logger.debug("Made pruning permanent for '%s'", name)
                except ValueError:
                    # Layer was not pruned — skip silently
                    pass

        logger.info("Pruning made permanent for %d Conv1d layers", made_permanent)
        return model

    # ------------------------------------------------------------------
    # Sparsity report
    # ------------------------------------------------------------------

    def get_sparsity_report(self, model: nn.Module) -> Dict[str, Any]:
        """Generate a per-layer sparsity report.

        For each ``Conv1d`` layer the report includes the total number
        of weight elements, the number of zeros, and the resulting
        sparsity percentage.

        Args:
            model: The (optionally pruned) model to inspect.

        Returns:
            Dictionary with ``layers`` list and ``overall_sparsity``.
        """
        layers: List[Dict[str, Any]] = []
        total_elements = 0
        total_zeros = 0

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv1d):
                weight = module.weight
                n_elements = weight.numel()
                n_zeros = int((weight == 0).sum().item())
                sparsity = n_zeros / n_elements if n_elements > 0 else 0.0

                layers.append(
                    {
                        "layer": name,
                        "shape": list(weight.shape),
                        "total_elements": n_elements,
                        "zero_elements": n_zeros,
                        "sparsity_pct": round(sparsity * 100, 2),
                    }
                )
                total_elements += n_elements
                total_zeros += n_zeros

        overall = total_zeros / total_elements if total_elements > 0 else 0.0

        report = {
            "layers": layers,
            "total_elements": total_elements,
            "total_zeros": total_zeros,
            "overall_sparsity_pct": round(overall * 100, 2),
        }

        logger.info(
            "Sparsity report — %d Conv1d layers, overall sparsity %.2f%%",
            len(layers),
            report["overall_sparsity_pct"],
        )
        return report


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    from system1.models.cnn_bigru import CNNBiGRU

    model = CNNBiGRU(num_features=46, num_classes=6)
    pruner = ChannelPruner()

    # Before pruning
    report_before = pruner.get_sparsity_report(model)
    print(f"Before pruning — overall sparsity: {report_before['overall_sparsity_pct']}%")

    # Apply pruning
    pruner.prune_conv_layers(model)
    report_after = pruner.get_sparsity_report(model)
    print(f"After pruning  — overall sparsity: {report_after['overall_sparsity_pct']}%")

    for layer in report_after["layers"]:
        print(f"  {layer['layer']}: {layer['sparsity_pct']}% sparse")

    # Make permanent
    pruner.make_permanent(model)
    print("Pruning made permanent.")
