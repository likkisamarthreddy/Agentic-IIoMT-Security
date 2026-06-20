# -*- coding: utf-8 -*-
"""
Symbolic Safety Rule Validation Engine
======================================

Evaluates proposed mitigation actions against the deterministic safety
policies defined in ``safety_policies.yaml``.  Every rule is evaluated
in **priority order** (lower number = higher priority) and the first
blocking rule wins.

Rule semantics supported:

* **RULE_001 — Life-Critical Protection**: blocks Level ≥ 4 for
  life-critical devices unless HITL approval is present.
* **RULE_002 — Telemetry Preservation**: ensures that critical data
  streams listed in ``preserve_streams`` are whitelisted during any
  mitigation action.
* **RULE_003 — Anti-Flap Guard**: holds the current state if a device
  has received more than 3 mitigation changes in the last 5 minutes.
* **RULE_004 — Correlated Threat Assessment**: (advisory) flags when
  ≥ 3 concurrent alerts exist on the same subnet.
* **RULE_005 — Operational Change Detection**: rescinds anomalous-
  command alerts when a recent prescription / operational change is
  present in context.
* **RULE_006 — Off-Hours Escalation**: lowers autonomous thresholds
  outside business hours.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Data classes                                                       #
# ------------------------------------------------------------------ #

@dataclass
class ValidationResult:
    """Outcome of validating a proposed action against symbolic rules.

    Attributes:
        is_valid: ``True`` if no blocking rule was triggered.
        blocked_by: Rule ID that blocked the action (or ``None``).
        message: Human-readable explanation.
        suggested_alternative: Alternative action dict if the original
            was blocked.
        rules_evaluated: List of rule IDs that were checked.
        preserve_streams: Streams that must stay active during the
            action.
    """

    is_valid: bool
    blocked_by: Optional[str] = None
    message: str = ""
    suggested_alternative: Optional[Dict[str, Any]] = None
    rules_evaluated: List[str] = field(default_factory=list)
    preserve_streams: List[str] = field(default_factory=list)


# ------------------------------------------------------------------ #
#  Symbolic Rule Engine                                               #
# ------------------------------------------------------------------ #

class SymbolicRuleEngine:
    """Deterministic safety-rule evaluator.

    Args:
        policies_path: Absolute or project-relative path to the
            ``safety_policies.yaml`` file.
    """

    def __init__(self, policies_path: Path) -> None:
        self._policies_path = Path(policies_path)
        self._policies: Dict[str, Any] = {}
        self._rules: List[Dict[str, Any]] = []
        self._device_constraints: Dict[str, Dict[str, Any]] = {}
        self._mitigation_levels: List[Dict[str, Any]] = []

        # Per-device recent mitigation change timestamps (anti-flap)
        self._mitigation_history: Dict[str, List[float]] = defaultdict(list)

        self._load_policies()

        logger.info(
            "SymbolicRuleEngine initialised — %d rules, %d device constraints",
            len(self._rules),
            len(self._device_constraints),
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def validate_action(
        self,
        proposed_action: Dict[str, Any],
        device_info: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """Check a proposed mitigation action against all symbolic rules.

        Rules are evaluated in ascending priority order.  The first
        *blocking* rule short-circuits the evaluation.

        Args:
            proposed_action: Dict with at least ``level`` (int),
                ``action_name`` (str), ``device_id`` (str).
            device_info: Device metadata (``type``, ``criticality``).
            context: Additional context for contextual rules (e.g.
                ``recent_prescription_change``, ``concurrent_alerts``).

        Returns:
            :class:`ValidationResult` describing the validation outcome.
        """
        # Normalise inputs: accept both dicts and dataclass objects
        import dataclasses as _dc
        if _dc.is_dataclass(proposed_action) and not isinstance(proposed_action, type):
            proposed_action = _dc.asdict(proposed_action)
        if _dc.is_dataclass(context) and not isinstance(context, type):
            context = _dc.asdict(context)
        context = context or {}
        if _dc.is_dataclass(device_info) and not isinstance(device_info, type):
            device_info = _dc.asdict(device_info)
        device_type: str = device_info.get("type", "unknown") if isinstance(device_info, dict) else getattr(device_info, "type", "unknown")
        device_id: str = proposed_action.get("device_id", device_info.get("id", "unknown") if isinstance(device_info, dict) else getattr(device_info, "id", "unknown"))
        action_level: int = int(proposed_action.get("level", 0))
        action_name: str = proposed_action.get("action_name", "")

        rules_evaluated: List[str] = []
        preserve_streams = self.get_preserve_streams(device_type)

        # Sort rules by priority (ascending)
        sorted_rules = sorted(self._rules, key=lambda r: r.get("priority", 99))

        for rule in sorted_rules:
            rule_id: str = rule.get("id", "UNKNOWN")
            rules_evaluated.append(rule_id)

            # --- RULE_001: Life-Critical Protection ---
            if rule_id == "RULE_001":
                criticality = device_info.get("criticality", "MEDIUM")
                hitl_approved = context.get("hitl_approved", False)
                if criticality == "LIFE_CRITICAL" and action_level >= 4 and not hitl_approved:
                    max_level = self.get_max_auto_level(device_type)
                    alternative = dict(proposed_action)
                    alternative["level"] = max_level
                    alternative["action_name"] = self._level_name(max_level)
                    logger.warning(
                        "[%s] BLOCKED: Cannot quarantine life-critical device %s without HITL",
                        rule_id,
                        device_id,
                    )
                    return ValidationResult(
                        is_valid=False,
                        blocked_by=rule_id,
                        message=rule.get("message", ""),
                        suggested_alternative=alternative,
                        rules_evaluated=rules_evaluated,
                        preserve_streams=preserve_streams,
                    )

            # --- RULE_002: Telemetry Preservation ---
            elif rule_id == "RULE_002":
                # Advisory: attach preserve_streams to result
                if preserve_streams:
                    logger.info(
                        "[%s] Device %s has %d streams to preserve",
                        rule_id,
                        device_id,
                        len(preserve_streams),
                    )

            # --- RULE_003: Anti-Flap Guard ---
            elif rule_id == "RULE_003":
                recent = self._mitigation_history.get(device_id, [])
                if self.check_anti_flap(device_id, recent):
                    logger.warning(
                        "[%s] BLOCKED: Anti-flap triggered for %s", rule_id, device_id,
                    )
                    return ValidationResult(
                        is_valid=False,
                        blocked_by=rule_id,
                        message=rule.get("message", ""),
                        suggested_alternative=None,
                        rules_evaluated=rules_evaluated,
                        preserve_streams=preserve_streams,
                    )

            # --- RULE_004: Correlated Threat Assessment ---
            elif rule_id == "RULE_004":
                concurrent = int(context.get("concurrent_alerts_same_subnet", 0))
                if concurrent >= 3:
                    logger.info(
                        "[%s] Correlated threat: %d concurrent alerts on subnet",
                        rule_id,
                        concurrent,
                    )
                    # Advisory — does not block, but may upgrade response

            # --- RULE_005: Operational Change Detection ---
            elif rule_id == "RULE_005":
                alert_data = context.get("alert", {})
                if self.check_operational_context(alert_data, context):
                    logger.info(
                        "[%s] Rescinding alert — matches recent operational change",
                        rule_id,
                    )
                    return ValidationResult(
                        is_valid=False,
                        blocked_by=rule_id,
                        message=rule.get("message", ""),
                        suggested_alternative={
                            "level": 0,
                            "action_name": "LOG_ONLY",
                            "device_id": device_id,
                            "reason": "rescind_operational_change",
                        },
                        rules_evaluated=rules_evaluated,
                        preserve_streams=preserve_streams,
                    )

            # --- RULE_006: Off-Hours Escalation ---
            elif rule_id == "RULE_006":
                risk_score = float(context.get("risk_score", 0.0))
                if not self._is_business_hours() and risk_score > 0.5:
                    logger.info(
                        "[%s] Off-hours — lowering auto threshold", rule_id,
                    )
                    # Advisory: caller should reduce max_auto_level by 1

            # --- Check device-specific forbidden actions ---
            forbidden = self.get_forbidden_actions(device_type)
            if action_name in forbidden:
                max_level = self.get_max_auto_level(device_type)
                alternative = dict(proposed_action)
                alternative["level"] = max_level
                alternative["action_name"] = self._level_name(max_level)
                logger.warning(
                    "Forbidden action '%s' for device type '%s' — suggesting level %d",
                    action_name,
                    device_type,
                    max_level,
                )
                return ValidationResult(
                    is_valid=False,
                    blocked_by=f"DEVICE_CONSTRAINT_{device_type}",
                    message=f"Action '{action_name}' is forbidden for {device_type}",
                    suggested_alternative=alternative,
                    rules_evaluated=rules_evaluated,
                    preserve_streams=preserve_streams,
                )

            # --- Check max_auto_mitigation_level ---
            max_auto = self.get_max_auto_level(device_type)
            hitl_approved = context.get("hitl_approved", False)
            if action_level > max_auto and not hitl_approved:
                alternative = dict(proposed_action)
                alternative["level"] = max_auto
                alternative["action_name"] = self._level_name(max_auto)
                logger.warning(
                    "Action level %d exceeds max auto level %d for %s",
                    action_level,
                    max_auto,
                    device_type,
                )
                return ValidationResult(
                    is_valid=False,
                    blocked_by=f"MAX_AUTO_{device_type}",
                    message=(
                        f"Mitigation level {action_level} exceeds maximum "
                        f"autonomous level {max_auto} for {device_type}"
                    ),
                    suggested_alternative=alternative,
                    rules_evaluated=rules_evaluated,
                    preserve_streams=preserve_streams,
                )

        # All rules passed
        # Record mitigation change for anti-flap tracking
        self._mitigation_history[device_id].append(time.time())

        logger.info(
            "Action VALIDATED for device %s: level=%d (%s)",
            device_id,
            action_level,
            action_name,
        )
        return ValidationResult(
            is_valid=True,
            message="All rules passed",
            rules_evaluated=rules_evaluated,
            preserve_streams=preserve_streams,
        )

    def get_max_auto_level(self, device_type: str) -> int:
        """Return the maximum autonomous mitigation level for a device type.

        Args:
            device_type: Device type key (e.g. ``infusion_pump``).

        Returns:
            Max auto mitigation level (int).  Defaults to ``4`` for
            unconstrained device types.
        """
        constraints = self._device_constraints.get(device_type, {})
        return int(constraints.get("max_auto_mitigation_level", 4))

    def get_forbidden_actions(self, device_type: str) -> List[str]:
        """Return actions forbidden for the given device type.

        Args:
            device_type: Device type key.

        Returns:
            List of forbidden action name strings.
        """
        constraints = self._device_constraints.get(device_type, {})
        return list(constraints.get("forbidden_actions", []))

    def get_preserve_streams(self, device_type: str) -> List[str]:
        """Return telemetry streams that must stay active during mitigation.

        Args:
            device_type: Device type key.

        Returns:
            List of stream name strings.
        """
        constraints = self._device_constraints.get(device_type, {})
        return list(constraints.get("preserve_streams", []))

    def check_anti_flap(
        self,
        device_id: str,
        recent_changes: List[float],
    ) -> bool:
        """Check RULE_003 — Anti-Flap Guard.

        Args:
            device_id: Device identifier.
            recent_changes: List of Unix timestamps of recent mitigation
                changes.

        Returns:
            ``True`` if flapping is detected (> 3 changes in 5 min).
        """
        now = time.time()
        cutoff = now - 300.0  # 5 minutes
        recent = [t for t in recent_changes if t >= cutoff]
        is_flapping = len(recent) > 3

        if is_flapping:
            logger.debug(
                "Anti-flap: device %s has %d changes in last 5min",
                device_id,
                len(recent),
            )
        return is_flapping

    def check_operational_context(
        self,
        alert: Dict[str, Any],
        context: Dict[str, Any],
    ) -> bool:
        """Check RULE_005 — Operational Change Detection.

        Returns ``True`` if the alert appears to be a false positive
        caused by a recent operational / prescription change.

        Args:
            alert: Alert data dict (needs ``type`` key).
            context: Context dict (needs
                ``recent_prescription_change`` bool key).

        Returns:
            ``True`` if the alert should be rescinded.
        """
        alert_type = alert.get("type", alert.get("attack_type", ""))
        prescription_changed = context.get("recent_prescription_change", False)

        return alert_type == "ANOMALOUS_COMMAND" and bool(prescription_changed)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _load_policies(self) -> None:
        """Load and parse ``safety_policies.yaml``."""
        try:
            with open(self._policies_path, "r", encoding="utf-8") as fh:
                self._policies = yaml.safe_load(fh) or {}

            self._rules = self._policies.get("rules", [])
            self._device_constraints = self._policies.get("device_constraints", {})
            self._mitigation_levels = self._policies.get("mitigation_levels", [])

            logger.info(
                "Loaded safety policies from %s: %d rules, %d device types",
                self._policies_path,
                len(self._rules),
                len(self._device_constraints),
            )
        except FileNotFoundError:
            logger.error("Safety policies not found: %s", self._policies_path)
        except yaml.YAMLError as exc:
            logger.error("Failed to parse safety policies: %s", exc)

    def _level_name(self, level: int) -> str:
        """Map a numeric mitigation level to its name.

        Args:
            level: Mitigation level integer.

        Returns:
            Name string (e.g. ``THROTTLE``).  Falls back to
            ``LEVEL_<n>`` if not found.
        """
        for entry in self._mitigation_levels:
            if entry.get("level") == level:
                return str(entry.get("name", f"LEVEL_{level}"))
        return f"LEVEL_{level}"

    @staticmethod
    def _is_business_hours() -> bool:
        """Check if current local time is within business hours (08–18).

        Returns:
            ``True`` if local hour is between 08:00 and 18:00.
        """
        current_hour = time.localtime().tm_hour
        return 8 <= current_hour < 18


# ------------------------------------------------------------------ #
#  Standalone smoke test                                              #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    project_root = Path(__file__).resolve().parents[3]
    policies = project_root / "config" / "safety_policies.yaml"

    engine = SymbolicRuleEngine(policies)

    action = {
        "level": 4,
        "action_name": "QUARANTINE",
        "device_id": "dev-001",
    }
    device = {"id": "dev-001", "type": "infusion_pump", "criticality": "LIFE_CRITICAL"}

    result = engine.validate_action(action, device, context={})
    print(f"Validation: valid={result.is_valid}, blocked_by={result.blocked_by}")
    print(f"Message: {result.message}")
    if result.suggested_alternative:
        print(f"Alternative: {result.suggested_alternative}")
