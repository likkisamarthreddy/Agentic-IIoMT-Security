# -*- coding: utf-8 -*-
"""
Adaptive KDE Anomaly Threshold
================================

Maintains a dynamic anomaly threshold computed via Kernel Density
Estimation (KDE) over a sliding window of *normal-traffic* anomaly
scores.  The threshold is the score at a configurable percentile
(default 99th) of the fitted density — scores above this value
are classified as anomalous.

The window is updated on-line and the KDE is periodically refitted
(every ``refit_interval`` normal samples).
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional

import numpy as np
from scipy.stats import gaussian_kde
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
# Adaptive KDE Threshold
# ---------------------------------------------------------------------------


class AdaptiveKDEThreshold:
    """Dynamic anomaly threshold via Gaussian KDE on a sliding window.

    The threshold is defined as the score at the *p*-th percentile of
    the fitted density.  Higher percentiles yield more conservative
    (fewer false-positive) thresholds.

    Args:
        window_size: Maximum number of recent normal scores to retain.
        percentile: Percentile (0–100) used to compute the threshold.
        bandwidth_method: Bandwidth selection method passed to
            ``scipy.stats.gaussian_kde``.
        refit_interval: Number of new normal samples between automatic
            KDE refits.
        config_path: Optional path to ``settings.yaml`` for defaults.

    Example::

        kde = AdaptiveKDEThreshold()
        kde.initialize(normal_scores_array)
        if kde.is_anomalous(new_score):
            trigger_alert()
    """

    def __init__(
        self,
        window_size: Optional[int] = None,
        percentile: Optional[float] = None,
        bandwidth_method: Optional[str] = None,
        refit_interval: Optional[int] = None,
        config_path: Optional[Path] = None,
    ) -> None:
        cfg = _load_config(config_path or _DEFAULT_CONFIG)
        kde_cfg = cfg.get("system1", {}).get("kde", {})

        self.window_size: int = int(window_size if window_size is not None else kde_cfg.get("window_size", 1000))
        self.percentile: float = float(percentile if percentile is not None else kde_cfg.get("percentile", 99.0))
        self.bandwidth_method: str = str(bandwidth_method if bandwidth_method is not None else kde_cfg.get("bandwidth_method", "scott"))
        self.refit_interval: int = int(refit_interval if refit_interval is not None else kde_cfg.get("refit_interval", 100))

        self._window: Deque[float] = deque(maxlen=self.window_size)
        self._kde: Optional[gaussian_kde] = None
        self._threshold: float = float("inf")
        self._samples_since_refit: int = 0
        self._is_initialized: bool = False

        logger.info(
            "AdaptiveKDEThreshold created — window=%d, percentile=%.1f, "
            "bandwidth=%s, refit_interval=%d",
            self.window_size,
            self.percentile,
            self.bandwidth_method,
            self.refit_interval,
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, normal_scores: np.ndarray) -> None:
        """Fit the initial KDE on a batch of normal-traffic scores.

        Args:
            normal_scores: 1-D array of anomaly scores from known
                normal traffic.

        Raises:
            ValueError: If *normal_scores* is empty or constant.
        """
        scores = np.asarray(normal_scores, dtype=np.float64).ravel()
        if scores.size < 2:
            raise ValueError("Need at least 2 scores to fit KDE")

        # Populate the sliding window
        self._window.clear()
        self._window.extend(scores[-self.window_size :].tolist())

        self._fit_kde()
        self._is_initialized = True

        logger.info(
            "KDE initialised with %d scores — threshold=%.6f",
            len(self._window),
            self._threshold,
        )

    # ------------------------------------------------------------------
    # Anomaly checking
    # ------------------------------------------------------------------

    def is_anomalous(self, score: float) -> bool:
        """Check whether a score exceeds the current KDE threshold.

        Args:
            score: Anomaly score from the edge classifier.

        Returns:
            ``True`` if the score is at or above the threshold.

        Raises:
            RuntimeError: If the threshold has not been initialised.
        """
        if not self._is_initialized:
            raise RuntimeError(
                "KDE threshold not initialised. Call initialize() first."
            )
        return float(score) >= self._threshold

    # ------------------------------------------------------------------
    # On-line update
    # ------------------------------------------------------------------

    def update(self, score: float, is_normal: bool = True) -> None:
        """Update the sliding window and periodically refit the KDE.

        Only *normal* scores are added to the window; anomalous scores
        are ignored to avoid threshold drift.

        Args:
            score: The latest anomaly score.
            is_normal: Whether the score comes from verified normal
                traffic.
        """
        if not is_normal:
            return

        self._window.append(float(score))
        self._samples_since_refit += 1

        if self._samples_since_refit >= self.refit_interval:
            self._fit_kde()
            self._samples_since_refit = 0
            logger.debug(
                "KDE refitted after %d samples — new threshold=%.6f",
                self.refit_interval,
                self._threshold,
            )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_threshold(self) -> float:
        """Return the current anomaly threshold.

        Returns:
            The score at the configured percentile of the fitted KDE.
        """
        return self._threshold

    def get_density(self, score: float) -> float:
        """Evaluate the KDE density at a given score.

        Args:
            score: The point at which to evaluate the density.

        Returns:
            Probability density value.

        Raises:
            RuntimeError: If the KDE has not been fitted.
        """
        if self._kde is None:
            raise RuntimeError("KDE has not been fitted yet.")
        density = float(self._kde.evaluate(np.array([score]))[0])
        return density

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit_kde(self) -> None:
        """Fit (or refit) the Gaussian KDE on the current window."""
        data = np.array(self._window, dtype=np.float64)

        # Guard against constant data (zero bandwidth)
        if data.std() < 1e-12:
            logger.warning(
                "Score window has near-zero variance (std=%.2e); "
                "threshold set to max(window) + epsilon",
                data.std(),
            )
            self._threshold = float(data.max()) + 1e-6
            return

        try:
            self._kde = gaussian_kde(data, bw_method=self.bandwidth_method)
        except Exception as exc:
            logger.error("KDE fitting failed: %s", exc)
            raise

        # Threshold = percentile from the empirical window
        self._threshold = float(np.percentile(data, self.percentile))

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AdaptiveKDEThreshold(window_size={self.window_size}, "
            f"percentile={self.percentile}, threshold={self._threshold:.6f}, "
            f"initialized={self._is_initialized})"
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

    # Simulate normal-traffic anomaly scores (low values)
    normal_scores = rng.beta(2, 50, size=500)  # mostly near 0

    kde = AdaptiveKDEThreshold()
    kde.initialize(normal_scores)

    print(f"Threshold : {kde.get_threshold():.6f}")
    print(f"Density at 0.1 : {kde.get_density(0.1):.6f}")
    print(f"is_anomalous(0.01): {kde.is_anomalous(0.01)}")
    print(f"is_anomalous(0.5) : {kde.is_anomalous(0.5)}")

    # On-line updates
    for s in rng.beta(2, 50, size=150):
        kde.update(float(s), is_normal=True)
    print(f"Threshold after updates: {kde.get_threshold():.6f}")
