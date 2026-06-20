# -*- coding: utf-8 -*-
"""
Data Pipeline Package for Cross-Domain Agentic Security (IIoMT).

This package provides:
    - **SyntheticIIoMTGenerator**: Generates synthetic network traffic matching
      CICIoMT2024 and Edge-IIoTset schemas with realistic attack signatures.
    - **DataPreprocessor**: Feature engineering pipeline including cleaning,
      encoding, scaling, sliding-window creation, and PyTorch DataLoader
      construction for the CNN-BiGRU model.
    - **TrafficReplayEngine**: Streams feature vectors over MQTT at a
      configurable rate, simulating ``tcpreplay`` for live inference.
"""

from data.synthetic_generator import SyntheticIIoMTGenerator
from data.preprocessor import DataPreprocessor
from data.traffic_replay import TrafficReplayEngine

__all__ = [
    "SyntheticIIoMTGenerator",
    "DataPreprocessor",
    "TrafficReplayEngine",
]
