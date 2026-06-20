# -*- coding: utf-8 -*-
"""
Lightweight Isolation Forest
=============================

A memory-efficient Isolation Forest wrapper for unsupervised anomaly
detection on IIoMT edge devices.  The model uses a reduced number of
estimators and samples to minimise RAM and inference latency while
retaining useful anomaly-scoring capability.

Typical usage::

    ifl = IsolationForestLite()
    ifl.fit(X_train)
    scores = ifl.anomaly_score(X_test)   # higher → more anomalous
    preds  = ifl.predict(X_test)          # -1 = anomaly, 1 = normal
"""

from __future__ import annotations

import logging
import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import yaml
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "settings.yaml"


def _load_system1_config(config_path: Path = _DEFAULT_CONFIG) -> Dict[str, Any]:
    """Load the ``system1`` configuration block.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Full parsed YAML dict.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Isolation Forest Lite
# ---------------------------------------------------------------------------


class IsolationForestLite:
    """Memory-efficient Isolation Forest for edge anomaly detection.

    Uses ``sklearn.ensemble.IsolationForest`` under the hood with
    conservative hyper-parameters to cap memory usage.

    Args:
        n_estimators: Number of trees.  Default ``50``.
        max_samples: Maximum samples drawn per tree.  Default ``256``.
        contamination: Expected proportion of anomalies.  Default ``"auto"``.
        random_state: Seed for reproducibility.  Default ``42``.
        config_path: Optional path to ``settings.yaml`` for seed override.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        max_samples: int = 256,
        contamination: Union[str, float] = "auto",
        random_state: Optional[int] = None,
        config_path: Optional[Path] = None,
    ) -> None:
        cfg = _load_system1_config(config_path or _DEFAULT_CONFIG)
        seed = random_state or cfg.get("data", {}).get("random_seed", 42)

        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination

        self._model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=seed,
            n_jobs=-1,
        )
        self._is_fitted: bool = False

        logger.info(
            "IsolationForestLite created — n_estimators=%d, "
            "max_samples=%d, contamination=%s",
            n_estimators,
            max_samples,
            contamination,
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray) -> "IsolationForestLite":
        """Fit the isolation forest on normal-traffic features.

        Args:
            X_train: Training data of shape ``(n_samples, n_features)``.

        Returns:
            ``self`` for method chaining.

        Raises:
            ValueError: If *X_train* is empty or has wrong dimensionality.
        """
        if X_train.ndim != 2:
            raise ValueError(
                f"Expected 2-D array, got shape {X_train.shape}"
            )
        if X_train.shape[0] == 0:
            raise ValueError("Training array must not be empty")

        logger.info(
            "Fitting IsolationForestLite on %d samples, %d features",
            *X_train.shape,
        )
        self._model.fit(X_train)
        self._is_fitted = True
        logger.info("Fit complete — memory footprint %.4f MB", self.get_memory_footprint())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels.

        Args:
            X: Samples of shape ``(n_samples, n_features)``.

        Returns:
            Array of ``+1`` (normal) or ``-1`` (anomaly) per sample.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._check_fitted()
        return self._model.predict(X)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores (negated sklearn decision function).

        Higher values → more anomalous.

        Args:
            X: Samples of shape ``(n_samples, n_features)``.

        Returns:
            1-D array of anomaly scores.
        """
        self._check_fitted()
        # sklearn's decision_function: lower → more anomalous
        # We negate so higher → more anomalous (convention match with CNN-BiGRU)
        return -self._model.decision_function(X)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_onnx(self, filepath: Union[str, Path]) -> Path:
        """Export the fitted model to ONNX format via *skl2onnx*.

        Falls back to a pickle export if *skl2onnx* is unavailable.

        Args:
            filepath: Destination path (should end in ``.onnx``).

        Returns:
            Resolved ``Path`` of the exported file.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._check_fitted()
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType

            n_features = self._model.n_features_in_
            initial_type = [
                ("X", FloatTensorType([None, n_features]))
            ]
            onnx_model = convert_sklearn(
                self._model,
                initial_types=initial_type,
                target_opset=13,
            )
            with open(filepath, "wb") as fh:
                fh.write(onnx_model.SerializeToString())
            logger.info("Exported ONNX model to %s", filepath)

        except ImportError:
            logger.warning(
                "skl2onnx not installed — falling back to pickle export"
            )
            pkl_path = filepath.with_suffix(".pkl")
            with open(pkl_path, "wb") as fh:
                pickle.dump(self._model, fh)
            filepath = pkl_path
            logger.info("Exported pickle model to %s", filepath)

        return filepath.resolve()

    # ------------------------------------------------------------------
    # Memory footprint
    # ------------------------------------------------------------------

    def get_memory_footprint(self) -> float:
        """Estimate the in-memory size of the model in megabytes.

        Serialises the model with ``pickle`` into a temporary buffer to
        get an accurate byte count.

        Returns:
            Model size in MB.
        """
        with tempfile.SpooledTemporaryFile(max_size=50 * 1024 * 1024) as tmp:
            pickle.dump(self._model, tmp)
            size_bytes = tmp.tell()
        size_mb = size_bytes / (1024 ** 2)
        return round(size_mb, 4)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise ``RuntimeError`` if the model has not been fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                "IsolationForestLite has not been fitted yet. Call fit() first."
            )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"IsolationForestLite(n_estimators={self.n_estimators}, "
            f"max_samples={self.max_samples}, "
            f"contamination={self.contamination}, "
            f"fitted={self._is_fitted})"
        )


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    rng = np.random.default_rng(42)
    X_normal = rng.standard_normal((1000, 46))
    X_test = rng.standard_normal((50, 46))
    X_test[0, :] = 10.0  # inject obvious anomaly

    ifl = IsolationForestLite()
    ifl.fit(X_normal)

    preds = ifl.predict(X_test)
    scores = ifl.anomaly_score(X_test)

    print(f"Predictions (first 10): {preds[:10]}")
    print(f"Scores      (first 10): {scores[:10].round(4)}")
    print(f"Memory footprint      : {ifl.get_memory_footprint():.4f} MB")
