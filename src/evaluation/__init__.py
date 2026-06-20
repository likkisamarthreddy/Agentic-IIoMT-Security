# ============================================================
# Cross-Domain Agentic Security for Industrial Medical IoT
# Evaluation Package
# ============================================================
"""
Evaluation toolkit for the IIoMT agentic security framework.

This package provides:
    - MetricsCollector: Comprehensive latency, resource, and detection
      metrics collection aligned with paper benchmarks.
    - AttackInjector: Phase 3 attack-scenario injection engine for
      DDoS, Spoofing, and MITM traffic generation.
    - BenchmarkReport: Publication-quality table/figure generation
      reproducing results from the research paper.
"""

from evaluation.metrics_collector import MetricsCollector
from evaluation.attack_injector import AttackInjector
from evaluation.benchmark_report import BenchmarkReport

__all__ = [
    "MetricsCollector",
    "AttackInjector",
    "BenchmarkReport",
]
