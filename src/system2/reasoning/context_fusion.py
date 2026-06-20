# -*- coding: utf-8 -*-
"""
Context Fusion Engine
=====================

Aggregates edge-agent alerts, device metadata, patient context, and
historical event logs into a unified risk assessment.  Implements
**Equation 1** of the framework:

    RiskMetric = α · Clf_Conf + β · Criticality_Index + γ · Historical_Density

where:
    - α, β, γ are configurable weights loaded from ``settings.yaml``
    - Clf_Conf is the classifier confidence from the edge agent (System 1)
    - Criticality_Index is the device criticality from the device registry
    - Historical_Density is the recent alert frequency for the device

Maintains an in-memory event log per device for temporal density
computation.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Data classes                                                       #
# ------------------------------------------------------------------ #

@dataclass
class RiskAssessment:
    """Unified risk assessment produced by context fusion.

    Attributes:
        risk_score: Composite risk metric in [0, 1].
        classifier_confidence: Edge classifier confidence for the alert.
        criticality_index: Normalised device criticality.
        historical_density: Recent alert frequency (events / second).
        device_id: Identifier of the originating device.
        device_type: Type key (e.g. ``infusion_pump``).
        alert_type: Attack / anomaly category label.
        raw_alert: Original alert payload.
        device_info: Enriched device metadata from the registry.
        patient_context: Optional patient context dict.
        timestamp: Unix epoch when the assessment was created.
    """

    risk_score: float
    classifier_confidence: float
    criticality_index: float
    historical_density: float
    device_id: str
    device_type: str
    alert_type: str
    raw_alert: Dict[str, Any]
    device_info: Dict[str, Any]
    patient_context: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)


# ------------------------------------------------------------------ #
#  Context Fusion Engine                                              #
# ------------------------------------------------------------------ #

class ContextFusionEngine:
    """Fuses heterogeneous context sources into a single risk score.

    Args:
        alpha: Weight for classifier confidence.
        beta: Weight for device criticality index.
        gamma: Weight for historical alert density.
        config_path: Path to ``settings.yaml`` for device registry
            look-ups.  If *None*, device look-ups return a default
            criticality of 0.5.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        config_path: Optional[Path] = None,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Validate weights sum to 1.0 (±tolerance)
        weight_sum = self.alpha + self.beta + self.gamma
        if abs(weight_sum - 1.0) > 1e-6:
            logger.warning(
                "Risk-metric weights do not sum to 1.0 (sum=%.4f). "
                "Normalising automatically.",
                weight_sum,
            )
            self.alpha /= weight_sum
            self.beta /= weight_sum
            self.gamma /= weight_sum

        # Load device registry from config
        self._device_registry: Dict[str, Dict[str, Any]] = {}
        self._criticality_levels: Dict[str, float] = {}
        if config_path is not None:
            self._load_device_registry(config_path)

        # In-memory per-device event log: device_id → list of timestamps
        self._event_log: Dict[str, List[float]] = defaultdict(list)

        logger.info(
            "ContextFusionEngine initialised (alpha=%.2f, beta=%.2f, gamma=%.2f)",
            self.alpha,
            self.beta,
            self.gamma,
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def compute_risk_metric(
        self,
        classifier_confidence: float,
        criticality_index: float,
        historical_density: float,
    ) -> float:
        """Compute composite risk score (Equation 1).

        Args:
            classifier_confidence: Edge classifier confidence ∈ [0, 1].
            criticality_index: Device criticality ∈ [0, 1].
            historical_density: Alert density ∈ [0, 1] (clamped).

        Returns:
            Composite risk score ∈ [0, 1].
        """
        # Clamp inputs to [0, 1]
        clf = max(0.0, min(1.0, classifier_confidence))
        crit = max(0.0, min(1.0, criticality_index))
        hist = max(0.0, min(1.0, historical_density))

        risk = self.alpha * clf + self.beta * crit + self.gamma * hist
        risk = max(0.0, min(1.0, risk))

        logger.debug(
            "RiskMetric = %.2f·%.3f + %.2f·%.3f + %.2f·%.3f = %.4f",
            self.alpha, clf, self.beta, crit, self.gamma, hist, risk,
        )
        return risk

    def aggregate_context(
        self,
        alert_data: Dict[str, Any],
        device_info: Dict[str, Any],
        patient_context: Optional[Dict[str, Any]] = None,
        historical_logs: Optional[List[Dict[str, Any]]] = None,
    ) -> RiskAssessment:
        """Fuse all context sources into a unified risk assessment.

        Args:
            alert_data: Alert payload from the edge agent.  Expected
                keys: ``device_id``, ``confidence``, ``attack_type``,
                ``timestamp``.
            device_info: Device metadata dict.  Expected keys: ``id``,
                ``type``, ``criticality``.
            patient_context: Optional patient-specific context dict
                (e.g. active procedures, prescriptions).
            historical_logs: Optional list of prior alert log dicts.
                If *None*, the internal per-device event log is used.

        Returns:
            A :class:`RiskAssessment` with the computed risk score and
            enriched metadata.
        """
        device_id: str = alert_data.get("device_id", device_info.get("id", "unknown"))
        alert_ts_raw = alert_data.get("timestamp", time.time())
        if isinstance(alert_ts_raw, str):
            from datetime import datetime
            try:
                alert_ts = datetime.fromisoformat(alert_ts_raw).timestamp()
            except ValueError:
                alert_ts = time.time()
        else:
            alert_ts = float(alert_ts_raw)

        # 1. Classifier confidence from edge
        clf_conf: float = float(alert_data.get("confidence", 0.0))

        # 2. Device criticality look-up
        crit_index = self.get_device_criticality(
            device_id, self._device_registry,
        )
        # Allow override from passed-in device_info
        if "criticality" in device_info:
            crit_label = device_info["criticality"]
            crit_index = self._criticality_levels.get(crit_label, crit_index)

        # 3. Historical density
        if historical_logs is not None:
            # Inject external logs as timestamps
            for entry in historical_logs:
                ts = entry.get("timestamp", alert_ts)
                self._event_log[device_id].append(ts)

        # Record current alert
        self._event_log[device_id].append(alert_ts)

        hist_density = self.compute_historical_density(
            device_id,
            self._event_log,
            time_window_sec=300.0,  # 5-minute window
        )

        # 4. Compute composite risk
        risk_score = self.compute_risk_metric(clf_conf, crit_index, hist_density)

        device_type = device_info.get("type", "unknown")
        alert_type = alert_data.get("attack_type", "UNKNOWN")

        assessment = RiskAssessment(
            risk_score=risk_score,
            classifier_confidence=clf_conf,
            criticality_index=crit_index,
            historical_density=hist_density,
            device_id=device_id,
            device_type=device_type,
            alert_type=alert_type,
            raw_alert=alert_data,
            device_info=device_info,
            patient_context=patient_context,
            timestamp=alert_ts,
        )

        logger.info(
            "Context fused for device=%s | risk=%.3f (clf=%.3f, crit=%.3f, hist=%.3f) | type=%s",
            device_id,
            risk_score,
            clf_conf,
            crit_index,
            hist_density,
            alert_type,
        )
        return assessment

    def get_device_criticality(
        self,
        device_id: str,
        device_registry: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> float:
        """Look up device criticality from the device registry.

        Args:
            device_id: Device identifier string.
            device_registry: Optional registry mapping device IDs to
                metadata dicts.  Falls back to the internally loaded
                registry.

        Returns:
            Normalised criticality value ∈ [0, 1].  Defaults to 0.5 if
            the device is not found.
        """
        registry = device_registry or self._device_registry
        if device_id in registry:
            dev = registry[device_id]
            crit_label = dev.get("criticality", "MEDIUM")
            crit_val = self._criticality_levels.get(crit_label, 0.5)
            logger.debug(
                "Device %s criticality: %s → %.2f", device_id, crit_label, crit_val,
            )
            return crit_val

        logger.debug(
            "Device %s not in registry — using default criticality 0.5",
            device_id,
        )
        return 0.5

    def compute_historical_density(
        self,
        device_id: str,
        event_log: Optional[Dict[str, List[float]]] = None,
        time_window_sec: float = 300.0,
    ) -> float:
        """Calculate recent alert frequency for a device.

        The density is the number of events in the time window divided
        by a saturation threshold of 10 events, clamped to [0, 1].

        Args:
            device_id: Device identifier string.
            event_log: Mapping of device IDs to lists of Unix timestamps.
                Falls back to the internal event log.
            time_window_sec: Look-back window in seconds.

        Returns:
            Normalised historical density ∈ [0, 1].
        """
        log = event_log if event_log is not None else self._event_log
        timestamps = log.get(device_id, [])

        if not timestamps:
            return 0.0

        now = time.time()
        cutoff = now - time_window_sec
        recent = [t for t in timestamps if t >= cutoff]

        # Saturation threshold: ≥10 events in the window → density 1.0
        saturation = 10.0
        density = min(len(recent) / saturation, 1.0)

        logger.debug(
            "Historical density for %s: %d events in %.0fs window → %.3f",
            device_id,
            len(recent),
            time_window_sec,
            density,
        )
        return density

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _load_device_registry(self, config_path: Path) -> None:
        """Load device registry and criticality levels from settings.yaml.

        Args:
            config_path: Path to the YAML configuration file.
        """
        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                config = yaml.safe_load(fh) or {}

            # Criticality levels mapping
            devices_cfg = config.get("devices", {})
            self._criticality_levels = devices_cfg.get("criticality_levels", {})

            # Build ID-indexed registry
            for dev in devices_cfg.get("default_devices", []):
                dev_id = dev.get("id", "")
                if dev_id:
                    self._device_registry[dev_id] = dev

            logger.info(
                "Loaded %d devices and %d criticality levels from %s",
                len(self._device_registry),
                len(self._criticality_levels),
                config_path,
            )
        except FileNotFoundError:
            logger.error("Config file not found: %s", config_path)
        except yaml.YAMLError as exc:
            logger.error("Failed to parse config YAML: %s", exc)


# ------------------------------------------------------------------ #
#  Standalone smoke test                                              #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    project_root = Path(__file__).resolve().parents[2]
    cfg_path = project_root / "config" / "settings.yaml"

    engine = ContextFusionEngine(alpha=0.5, beta=0.3, gamma=0.2, config_path=cfg_path)

    # Simulate an alert
    alert = {
        "device_id": "dev-001",
        "confidence": 0.92,
        "attack_type": "DDoS",
        "timestamp": time.time(),
    }
    device = {
        "id": "dev-001",
        "type": "infusion_pump",
        "criticality": "LIFE_CRITICAL",
    }

    result = engine.aggregate_context(alert, device)
    print(f"Risk Assessment: {result}")
