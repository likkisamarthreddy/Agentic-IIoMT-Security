# -*- coding: utf-8 -*-
"""
System 1 Quantization & Pruning
================================

Post-training optimisation utilities that shrink models to fit
within the 15 MB / 45 MB RAM edge budget.

Classes:
    ModelQuantizer – INT8 dynamic quantization and ONNX export
    ChannelPruner  – L1-structured channel pruning for Conv1d layers
"""

from system1.quantization.quantizer import ModelQuantizer
from system1.quantization.pruner import ChannelPruner

__all__ = [
    "ModelQuantizer",
    "ChannelPruner",
]
