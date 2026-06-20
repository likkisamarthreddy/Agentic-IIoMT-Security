# -*- coding: utf-8 -*-
"""
INT8 Post-Training Quantization
================================

Provides dynamic INT8 quantization, ONNX export, model size
measurement, latency benchmarking, and FP32 vs INT8 comparison
utilities for the CNN-BiGRU edge classifier.

The target model size after quantization is < 15 MB
(``system1.quantization.target_model_size_mb`` in config).
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.quantization
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "settings.yaml"


def _load_config(config_path: Path = _DEFAULT_CONFIG) -> Dict[str, Any]:
    """Load the full configuration from *settings.yaml*."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Model Quantizer
# ---------------------------------------------------------------------------


class ModelQuantizer:
    """INT8 dynamic post-training quantization and benchmarking toolkit.

    This class wraps PyTorch's ``quantize_dynamic`` to convert GRU and
    Linear layers to INT8 representations, exports models to ONNX, and
    provides helpers to verify that the resulting artifact meets the
    edge size budget (< 15 MB by default).

    Args:
        config_path: Path to ``settings.yaml``.
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        cfg = _load_config(config_path or _DEFAULT_CONFIG)
        s1q = cfg.get("system1", {}).get("quantization", {})

        self.backend: str = s1q.get("backend", "fbgemm")
        self.calibration_samples: int = s1q.get("calibration_samples", 500)
        self.target_size_mb: float = s1q.get("target_model_size_mb", 15.0)

        logger.info(
            "ModelQuantizer initialised — backend=%s, target_size=%.1f MB",
            self.backend,
            self.target_size_mb,
        )

    # ------------------------------------------------------------------
    # Dynamic INT8 quantization
    # ------------------------------------------------------------------

    def quantize_static(self, model: nn.Module, dataloader: torch.utils.data.DataLoader) -> nn.Module:
        """Apply static INT8 quantization to the network.

        Args:
            model: A PyTorch ``nn.Module`` (typically a ``CNNBiGRU``).
            dataloader: DataLoader providing calibration data.

        Returns:
            A new quantised model.
        """
        logger.info("Applying static INT8 quantization (backend=%s)", self.backend)

        try:
            torch.backends.quantized.engine = self.backend
        except RuntimeError as e:
            logger.warning(f"Could not set quantization backend to {self.backend}: {e}. Using default.")

        # Create a copy so we don't mutate the original FP32
        import copy
        quantized_model = copy.deepcopy(model)
        quantized_model.eval()

        qconfig = torch.quantization.get_default_qconfig(self.backend)
        quantized_model.qconfig = qconfig
        
        # Fix for "AttributeError: 'tuple' object has no attribute 'numel'"
        # nn.GRU returns (output, h_n), which crashes the observer.
        # We explicitly keep the GRU in FP32 (which is better for accuracy anyway).
        if hasattr(quantized_model, "gru"):
            quantized_model.gru.qconfig = None
        
        # Fuse layers if applicable, though not strictly required
        
        torch.quantization.prepare(quantized_model, inplace=True)

        logger.info("Calibrating on %d batches...", len(dataloader))
        with torch.no_grad():
            for i, (inputs, _) in enumerate(dataloader):
                if i >= self.calibration_samples:
                    break
                quantized_model(inputs)

        torch.quantization.convert(quantized_model, inplace=True)

        original_mb = self.measure_model_size(model)
        quantized_mb = self.measure_model_size(quantized_model)
        compression = (1 - quantized_mb / original_mb) * 100 if original_mb > 0 else 0.0

        logger.info(
            "Quantization complete - FP32=%.4f MB -> INT8=%.4f MB (%.1f%% reduction)",
            original_mb,
            quantized_mb,
            compression,
        )

        if quantized_mb > self.target_size_mb:
            logger.warning(
                "Quantized model (%.4f MB) exceeds target (%.1f MB)",
                quantized_mb,
                self.target_size_mb,
            )
        return quantized_model

    def quantize_dynamic(self, model: nn.Module) -> nn.Module:
        """Apply dynamic INT8 quantization to the network."""
        logger.info("Applying dynamic INT8 quantization")
        import torch.quantization
        # Create a copy so we don't mutate the original FP32
        import copy
        quantized_model = copy.deepcopy(model)
        quantized_model.eval()
        
        # Dynamically quantize Linear and GRU layers
        quantized_model = torch.quantization.quantize_dynamic(
            quantized_model,
            {torch.nn.Linear, torch.nn.GRU},
            dtype=torch.qint8
        )
        
        original_mb = self.measure_model_size(model)
        quantized_mb = self.measure_model_size(quantized_model)
        compression = (1 - quantized_mb / original_mb) * 100 if original_mb > 0 else 0.0

        logger.info(
            "Dynamic Quantization complete - FP32=%.4f MB -> INT8=%.4f MB (%.1f%% reduction)",
            original_mb,
            quantized_mb,
            compression,
        )
        return quantized_model

    # ------------------------------------------------------------------
    # ONNX export
    # ------------------------------------------------------------------

    def export_onnx(
        self,
        model: nn.Module,
        filepath: str | Path,
        input_shape: Tuple[int, ...] = (1, 1, 46),
    ) -> Path:
        """Export a PyTorch model to ONNX format.

        Args:
            model: The model to export.
            filepath: Destination ``.onnx`` file path.
            input_shape: Shape of a single input sample
                (default ``(1, 1, 46)``).

        Returns:
            Resolved path to the exported ONNX file.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model.eval()
        dummy_input = torch.randn(*input_shape)

        try:
            torch.onnx.export(
                model,
                dummy_input,
                str(filepath),
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"},
                },
            )
            logger.info("ONNX model exported to %s", filepath)
        except Exception as exc:
            logger.error("ONNX export failed: %s", exc)
            raise

        return filepath.resolve()

    # ------------------------------------------------------------------
    # Model size measurement
    # ------------------------------------------------------------------

    def measure_model_size(self, model: nn.Module) -> float:
        """Return the serialised model size in megabytes.

        The model is saved to a temporary file via ``torch.save``
        and the file size is measured.

        Args:
            model: The model to measure.

        Returns:
            Size in MB.
        """
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            torch.save(model.state_dict(), tmp_path)
            size_bytes = os.path.getsize(tmp_path)
        finally:
            os.remove(tmp_path)

        size_mb = size_bytes / (1024 ** 2)
        return round(size_mb, 4)

    # ------------------------------------------------------------------
    # Latency benchmarking
    # ------------------------------------------------------------------

    def benchmark_latency(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        iterations: int = 1000,
        warmup: int = 100,
    ) -> Dict[str, float]:
        """Benchmark mean inference latency in milliseconds.

        Args:
            model: The model to benchmark.
            input_tensor: A representative input tensor.
            iterations: Number of timed forward passes.
            warmup: Number of untimed warm-up iterations.

        Returns:
            Dictionary with ``mean_latency_ms``, ``std_latency_ms``,
            ``min_latency_ms``, and ``max_latency_ms``.
        """
        model.eval()

        # Warm-up (not measured)
        with torch.no_grad():
            for _ in range(warmup):
                model(input_tensor)

        # Timed iterations
        latencies: List[float] = []
        with torch.no_grad():
            for _ in range(iterations):
                t0 = time.perf_counter_ns()
                model(input_tensor)
                t1 = time.perf_counter_ns()
                latencies.append((t1 - t0) / 1e6)  # ns → ms

        import numpy as np

        arr = np.array(latencies)
        result = {
            "mean_latency_ms": round(float(arr.mean()), 4),
            "std_latency_ms": round(float(arr.std()), 4),
            "min_latency_ms": round(float(arr.min()), 4),
            "max_latency_ms": round(float(arr.max()), 4),
            "iterations": iterations,
        }

        logger.info(
            "Latency benchmark — mean=%.4f ms, std=%.4f ms (%d iters)",
            result["mean_latency_ms"],
            result["std_latency_ms"],
            iterations,
        )
        return result

    # ------------------------------------------------------------------
    # FP32 vs INT8 comparison
    # ------------------------------------------------------------------

    def compare_models(
        self,
        fp32_model: nn.Module,
        int8_model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
    ) -> Dict[str, Any]:
        """Compare FP32 and INT8 models on accuracy, size, and latency.

        Args:
            fp32_model: Original FP32 model.
            int8_model: Quantised INT8 model.
            test_loader: ``DataLoader`` yielding ``(inputs, labels)`` tuples.

        Returns:
            Dictionary with comparative metrics for both models.
        """
        logger.info("Starting FP32 vs INT8 model comparison")

        def _evaluate(model: nn.Module) -> float:
            """Return top-1 accuracy on *test_loader*."""
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, dim=1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            return correct / total if total > 0 else 0.0

        # Accuracy
        fp32_acc = _evaluate(fp32_model)
        int8_acc = _evaluate(int8_model)

        # Size
        fp32_mb = self.measure_model_size(fp32_model)
        int8_mb = self.measure_model_size(int8_model)

        # Latency (use a representative batch from the loader)
        sample_input, _ = next(iter(test_loader))
        fp32_lat = self.benchmark_latency(fp32_model, sample_input, iterations=200, warmup=50)
        int8_lat = self.benchmark_latency(int8_model, sample_input, iterations=200, warmup=50)

        comparison = {
            "fp32": {
                "accuracy": round(fp32_acc, 4),
                "size_mb": fp32_mb,
                "latency": fp32_lat,
            },
            "int8": {
                "accuracy": round(int8_acc, 4),
                "size_mb": int8_mb,
                "latency": int8_lat,
            },
            "accuracy_drop": round(fp32_acc - int8_acc, 4),
            "size_reduction_pct": round(
                (1 - int8_mb / fp32_mb) * 100 if fp32_mb > 0 else 0.0, 2
            ),
            "speedup": round(
                fp32_lat["mean_latency_ms"] / int8_lat["mean_latency_ms"]
                if int8_lat["mean_latency_ms"] > 0
                else 0.0,
                2,
            ),
            "meets_target": int8_mb <= self.target_size_mb,
        }

        logger.info(
            "Comparison — FP32 acc=%.4f (%.4f MB) | INT8 acc=%.4f (%.4f MB) | "
            "drop=%.4f | reduction=%.1f%% | speedup=%.2fx",
            fp32_acc,
            fp32_mb,
            int8_acc,
            int8_mb,
            comparison["accuracy_drop"],
            comparison["size_reduction_pct"],
            comparison["speedup"],
        )
        return comparison


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    # Import here to avoid circular at module level during standalone test
    from system1.models.cnn_bigru import CNNBiGRU

    model = CNNBiGRU(num_features=46, num_classes=6)
    quantizer = ModelQuantizer()

    print(f"FP32 size: {quantizer.measure_model_size(model):.4f} MB")

    int8_model = quantizer.quantize_dynamic(model)
    print(f"INT8 size: {quantizer.measure_model_size(int8_model):.4f} MB")

    dummy = torch.randn(1, 1, 46)
    lat = quantizer.benchmark_latency(int8_model, dummy, iterations=500)
    print(f"Latency  : {lat['mean_latency_ms']:.4f} ms")
