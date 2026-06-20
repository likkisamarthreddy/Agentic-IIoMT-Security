# ============================================================
# Cross-Domain Agentic Security for Industrial Medical IoT
# Infrastructure Package
# ============================================================
"""
Infrastructure utilities for the IIoMT agentic security framework.

This package provides:
    - NetworkEmulator: Windows-compatible multi-node topology emulator
      using Python threading and psutil (Mininet alternative).
    - Docker Compose / Dockerfile definitions (YAML, not importable).
"""

from infrastructure.network_emulator import NetworkEmulator, VirtualNode

__all__ = [
    "NetworkEmulator",
    "VirtualNode",
]
