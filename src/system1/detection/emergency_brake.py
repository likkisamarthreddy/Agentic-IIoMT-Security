# -*- coding: utf-8 -*-
"""
Emergency Brake — SDN Micro-Mitigation
========================================

When the sliding-window average of anomaly scores breaches the
adaptive KDE threshold, the emergency brake generates SDN
micro-mitigation rules that preserve life-critical telemetry
streams while throttling or isolating the offending device.

Supported SDN actions:

* **THROTTLE** — rate-limit device traffic (low severity).
* **DROP_UNAUTH** — drop unauthorised flows (medium severity).
* **READ_ONLY_TELEMETRY** — restrict device to read-only telemetry
  and block all command/write packets (critical severity).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np
import yaml

from system1.detection.kde_threshold import AdaptiveKDEThreshold

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
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MitigationAction:
    """Immutable record of an SDN micro-mitigation action.

    Attributes:
        action: SDN action type (``THROTTLE``, ``DROP_UNAUTH``,
            ``READ_ONLY_TELEMETRY``).
        device_id: Target device identifier.
        severity: Severity level (``low``, ``medium``, ``critical``).
        timestamp: UTC timestamp of the action.
        preserve_streams: List of protocol streams that must **not**
            be interrupted (e.g. vital-sign telemetry).
        metadata: Optional key-value bag for extra context.
    """

    action: str
    device_id: str
    severity: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    preserve_streams: List[str] = field(default_factory=lambda: ["vital_signs_telemetry"])
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Severity ↔ action mapping
# ---------------------------------------------------------------------------

_SEVERITY_ACTION_MAP: Dict[str, str] = {
    "low": "THROTTLE",
    "medium": "DROP_UNAUTH",
    "critical": "READ_ONLY_TELEMETRY",
}


# ---------------------------------------------------------------------------
# Emergency Brake
# ---------------------------------------------------------------------------


class EmergencyBrake:
    """Sliding-window anomaly evaluator and SDN rule generator.

    Maintains a window of the most recent per-packet anomaly scores.
    When the window average exceeds the adaptive KDE threshold, the
    brake triggers a ``MitigationAction`` and generates the
    corresponding SDN flow rule.

    Args:
        score_window: Number of recent scores to average.
        kde_threshold: A pre-initialised ``AdaptiveKDEThreshold``.
        min_confidence: Minimum average score to consider triggering.
        config_path: Path to ``settings.yaml`` for defaults.

    Example::

        brake = EmergencyBrake(
            score_window=10,
            kde_threshold=kde,
        )
        result = brake.evaluate([0.9, 0.85, 0.92, ...])
        if result is not None:
            sdn_rule = brake.trigger_mitigation("dev-001", "critical")
    """

    def __init__(
        self,
        score_window: Optional[int] = None,
        kde_threshold: Optional[AdaptiveKDEThreshold] = None,
        min_confidence: Optional[float] = None,
        config_path: Optional[Path] = None,
    ) -> None:
        cfg = _load_config(config_path or _DEFAULT_CONFIG)
        eb_cfg = cfg.get("system1", {}).get("emergency_brake", {})

        self.score_window: int = score_window or eb_cfg.get("score_window", 10)
        self.min_confidence: float = (
            min_confidence if min_confidence is not None else eb_cfg.get("min_confidence", 0.5)
        )
        self.kde_threshold = kde_threshold

        self._recent_scores: Deque[float] = deque(maxlen=self.score_window)
        self._mitigation_log: List[MitigationAction] = []

        logger.info(
            "EmergencyBrake initialised — score_window=%d, min_confidence=%.2f",
            self.score_window,
            self.min_confidence,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, packet_scores: List[float]) -> Optional[Dict[str, Any]]:
        """Evaluate a batch of per-packet anomaly scores.

        Appends each score to the sliding window.  If the window is
        full and the average score exceeds the KDE threshold **and**
        ``min_confidence``, returns an alert dictionary; otherwise
        returns ``None``.

        Args:
            packet_scores: List of anomaly scores (one per packet).

        Returns:
            Alert dict with ``mean_score``, ``threshold``, ``triggered``
            keys, or ``None`` if no alert is warranted.
        """
        for score in packet_scores:
            self._recent_scores.append(float(score))

        if len(self._recent_scores) < self.score_window:
            logger.debug(
                "Insufficient scores (%d/%d) — skipping evaluation",
                len(self._recent_scores),
                self.score_window,
            )
            return None

        mean_score = float(np.mean(list(self._recent_scores)))

        # Check both KDE threshold and minimum confidence
        kde_triggered = (
            self.kde_threshold.is_anomalous(mean_score)
            if self.kde_threshold is not None
            else mean_score >= self.min_confidence
        )

        if kde_triggered and mean_score >= self.min_confidence:
            threshold = (
                self.kde_threshold.get_threshold()
                if self.kde_threshold is not None
                else self.min_confidence
            )
            alert = {
                "triggered": True,
                "mean_score": round(mean_score, 6),
                "threshold": round(threshold, 6),
                "window_size": self.score_window,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            logger.warning(
                "Emergency brake TRIGGERED — mean_score=%.6f > threshold=%.6f",
                mean_score,
                threshold,
            )
            return alert

        return None

    # ------------------------------------------------------------------
    # Mitigation actions
    # ------------------------------------------------------------------

    def trigger_mitigation(
        self,
        device_id: str,
        severity: str = "medium",
    ) -> MitigationAction:
        """Create and log a mitigation action for a device.

        Args:
            device_id: Identifier of the target device.
            severity: One of ``low``, ``medium``, ``critical``.

        Returns:
            A ``MitigationAction`` dataclass instance.

        Raises:
            ValueError: If *severity* is not recognised.
        """
        if severity not in _SEVERITY_ACTION_MAP:
            raise ValueError(
                f"Unknown severity '{severity}'. "
                f"Expected one of {list(_SEVERITY_ACTION_MAP.keys())}"
            )

        action_type = _SEVERITY_ACTION_MAP[severity]
        sdn_rule = self.generate_sdn_rule(action_type, device_id)

        mitigation = MitigationAction(
            action=action_type,
            device_id=device_id,
            severity=severity,
            preserve_streams=["vital_signs_telemetry", "alarm_notifications"],
            metadata={"sdn_rule": sdn_rule},
        )

        self._mitigation_log.append(mitigation)

        logger.info(
            "Mitigation triggered — device=%s, action=%s, severity=%s",
            device_id,
            action_type,
            severity,
        )
        return mitigation

    # ------------------------------------------------------------------
    # SDN rule generation
    # ------------------------------------------------------------------

    def generate_sdn_rule(
        self,
        action_type: str,
        device_id: str,
    ) -> Dict[str, Any]:
        """Generate an SDN micro-mitigation flow rule.

        Supported *action_type* values:

        * ``THROTTLE`` — rate-limit to 10 % of normal bandwidth.
        * ``DROP_UNAUTH`` — drop flows not matching an ACL whitelist.
        * ``READ_ONLY_TELEMETRY`` — allow only read/telemetry flows;
          drop all write/command packets.

        Args:
            action_type: SDN action (see above).
            device_id: Target device identifier.

        Returns:
            Dictionary describing the SDN flow rule.

        Raises:
            ValueError: If *action_type* is unrecognised.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        if action_type == "THROTTLE":
            rule = {
                "rule_type": "RATE_LIMIT",
                "device_id": device_id,
                "parameters": {
                    "max_bandwidth_pct": 10,
                    "burst_size": 64,
                },
                "priority": 100,
                "timeout_sec": 300,
                "preserve": ["vital_signs_telemetry"],
                "timestamp": timestamp,
            }

        elif action_type == "DROP_UNAUTH":
            rule = {
                "rule_type": "ACL_DROP",
                "device_id": device_id,
                "parameters": {
                    "whitelist_protocols": ["MQTT", "CoAP", "HL7"],
                    "drop_unknown_src": True,
                },
                "priority": 200,
                "timeout_sec": 600,
                "preserve": ["vital_signs_telemetry", "alarm_notifications"],
                "timestamp": timestamp,
            }

        elif action_type == "READ_ONLY_TELEMETRY":
            rule = {
                "rule_type": "FLOW_RESTRICT",
                "device_id": device_id,
                "parameters": {
                    "allowed_directions": ["device_to_gateway"],
                    "blocked_operations": ["write", "command", "firmware_update"],
                    "allowed_operations": ["read", "telemetry", "alarm"],
                },
                "priority": 500,
                "timeout_sec": 900,
                "preserve": [
                    "vital_signs_telemetry",
                    "alarm_notifications",
                    "diagnostic_read",
                ],
                "timestamp": timestamp,
            }

        else:
            raise ValueError(
                f"Unknown SDN action '{action_type}'. "
                f"Expected THROTTLE | DROP_UNAUTH | READ_ONLY_TELEMETRY"
            )

        logger.debug("Generated SDN rule: %s", rule)
        return rule

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def mitigation_history(self) -> List[MitigationAction]:
        """Return the list of all past mitigation actions."""
        return list(self._mitigation_log)

    def clear_history(self) -> None:
        """Clear the mitigation log and score window."""
        self._mitigation_log.clear()
        self._recent_scores.clear()
        logger.info("Emergency brake history cleared")


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    # Create a KDE threshold from synthetic data
    rng = np.random.default_rng(42)
    normal_scores = rng.beta(2, 50, size=500)

    kde = AdaptiveKDEThreshold()
    kde.initialize(normal_scores)

    brake = EmergencyBrake(score_window=5, kde_threshold=kde)

    # Normal traffic — should NOT trigger
    result = brake.evaluate([0.01, 0.02, 0.015, 0.018, 0.012])
    print(f"Normal eval: {result}")

    # Anomalous traffic — should trigger
    result = brake.evaluate([0.9, 0.85, 0.92, 0.88, 0.95])
    print(f"Attack eval: {result}")

    if result and result.get("triggered"):
        action = brake.trigger_mitigation("dev-001", "critical")
        print(f"Mitigation : {action}")
        print(f"SDN rule   : {action.metadata['sdn_rule']}")
