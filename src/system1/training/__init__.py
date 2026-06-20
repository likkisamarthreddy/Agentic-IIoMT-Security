# -*- coding: utf-8 -*-
"""
System 1 Training
==================

End-to-end training, evaluation, and benchmarking pipeline
for the CNN-BiGRU edge classifier.

Classes:
    ModelTrainer – Full training pipeline with early stopping, scheduling,
                   per-attack evaluation, and quantization comparison.
"""

from system1.training.trainer import ModelTrainer

__all__ = [
    "ModelTrainer",
]
