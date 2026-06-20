# -*- coding: utf-8 -*-
"""
System 1 — Edge Reflex Layer
=============================

Sub-3 ms anomaly detection pipeline for IIoMT edge devices.
Provides CNN-BiGRU classification, adaptive KDE thresholding,
INT8 quantization, and emergency SDN micro-mitigation.

Modules:
    models       – CNN-BiGRU classifier and Isolation Forest lite
    quantization – INT8 quantization and channel pruning
    detection    – Adaptive KDE threshold and emergency brake
    training     – Full training and evaluation pipeline
    edge_agent   – MQTT-driven edge orchestrator
"""

from system1.edge_agent import EdgeAgent

__all__ = [
    "EdgeAgent",
]
