# -*- coding: utf-8 -*-
"""
Graduated Mitigation Action Playbook
=====================================

Maps composite risk scores to graduated mitigation levels defined in
``safety_policies.yaml``.  Each level specifies:

* Whether human-in-the-loop (HITL) approval is required.
* Whether the action is reversible.
* A risk-score range that triggers the level.

Device-specific constraints (max auto level, forbidden actions,
preserve streams) are enforced before returning a mitigation action.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Data classes                                                       #
# ------------------------------------------------------------------ #

@dataclass
class MitigationLevel:
    """Descriptor for a single graduated mitigation level.

    Attributes:
        level: Integer level number (0–4).
        name: Short name (e.g. ``THROTTLE``).
        description: Human-readable description.
        risk_range: ``(low, high)`` risk-score boundaries.
        requires_hitl: Whether clinician approval is needed.
        reversible: Whether the action can be rolled back automatically.
    """

    level: int
    name: str
    description: str
    risk_range: tuple[float, float]
    requires_hitl: bool
    reversible: bool


@dataclass
class MitigationAction:
    """Concrete mitigation action to be executed by the SDN controller.

    Attributes:
        level: Mitigation level (int).
        action_name: Action name string (e.g. ``THROTTLE``).
        device_id: Target device identifier.
        parameters: Action-specific parameters dict.
        timestamp: Unix epoch of creation.
        requires_hitl: Whether the action needs HITL approval.
        preserve_streams: Telemetry streams to whitelist.
    """

    level: int
    action_name: str
    device_id: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    requires_hitl: bool = False
    preserve_streams: List[str] = field(default_factory=list)


# ------------------------------------------------------------------ #
#  Action Playbook                                                    #
# ------------------------------------------------------------------ #

class ActionPlaybook:
    """Graduated mitigation action selector.

    Args:
        policies_path: Path to ``safety_policies.yaml``.
    """

    def __init__(self, policies_path: Path) -> None:
        self._policies_path = Path(policies_path)
        self._levels: List[MitigationLevel] = []
        self._device_constraints: Dict[str, Dict[str, Any]] = {}
        self._load_policies()

        logger.info(
            "ActionPlaybook initialised — %d mitigation levels loaded",
            len(self._levels),
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def select_action(
        self,
        risk_score: float,
        device_info: Dict[str, Any],
    ) -> MitigationAction:
        """Map a risk score to a mitigation action, respecting device constraints.

        Args:
            risk_score: Composite risk score ∈ [0, 1].
            device_info: Device metadata dict with at least ``id``,
                ``type`` keys.

        Returns:
            :class:`MitigationAction` ready for execution.
        """
        device_id: str = device_info.get("id", "unknown")
        device_type: str = device_info.get("type", "unknown")

        # 1. Find matching level by risk score
        selected = self._match_level(risk_score)

        # 2. Enforce device constraints
        constraints = self._device_constraints.get(device_type, {})
        max_auto: int = int(constraints.get("max_auto_mitigation_level", 4))
        forbidden: List[str] = list(constraints.get("forbidden_actions", []))
        preserve: List[str] = list(constraints.get("preserve_streams", []))

        if selected.name in forbidden:
            logger.warning(
                "Level %d (%s) forbidden for %s — capping at max auto %d",
                selected.level,
                selected.name,
                device_type,
                max_auto,
            )
            selected = self._get_level_obj(max_auto) or selected

        if selected.level > max_auto:
            logger.info(
                "Level %d exceeds max_auto %d for %s — capping",
                selected.level,
                max_auto,
                device_type,
            )
            selected = self._get_level_obj(max_auto) or selected

        # 3. Build action parameters
        params = self._build_parameters(selected, device_info)

        action = MitigationAction(
            level=selected.level,
            action_name=selected.name,
            device_id=device_id,
            parameters=params,
            requires_hitl=selected.requires_hitl,
            preserve_streams=preserve,
        )

        logger.info(
            "Selected mitigation: level=%d (%s) for device %s (risk=%.3f)",
            action.level,
            action.action_name,
            device_id,
            risk_score,
        )
        return action

    def get_level(self, level_num: int) -> Optional[MitigationLevel]:
        """Return the :class:`MitigationLevel` for a given level number.

        Args:
            level_num: Integer level (0-based).

        Returns:
            :class:`MitigationLevel` or ``None`` if not found.
        """
        return self._get_level_obj(level_num)

    def get_all_levels(self) -> List[MitigationLevel]:
        """Return all mitigation levels in ascending order.

        Returns:
            Ordered list of :class:`MitigationLevel` instances.
        """
        return sorted(self._levels, key=lambda lv: lv.level)

    def escalate(self, current_level: int) -> Optional[MitigationLevel]:
        """Return the next higher mitigation level.

        Args:
            current_level: Current mitigation level number.

        Returns:
            Next higher :class:`MitigationLevel`, or ``None`` if already
            at maximum.
        """
        candidates = [lv for lv in self._levels if lv.level > current_level]
        if not candidates:
            logger.debug("No escalation possible from level %d", current_level)
            return None
        return min(candidates, key=lambda lv: lv.level)

    def de_escalate(self, current_level: int) -> Optional[MitigationLevel]:
        """Return the next lower mitigation level.

        Args:
            current_level: Current mitigation level number.

        Returns:
            Next lower :class:`MitigationLevel`, or ``None`` if already
            at minimum.
        """
        candidates = [lv for lv in self._levels if lv.level < current_level]
        if not candidates:
            logger.debug("No de-escalation possible from level %d", current_level)
            return None
        return max(candidates, key=lambda lv: lv.level)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _match_level(self, risk_score: float) -> MitigationLevel:
        """Find the mitigation level whose risk range contains the score.

        Args:
            risk_score: Risk score ∈ [0, 1].

        Returns:
            Matching :class:`MitigationLevel`.  Falls back to highest
            level if the score exceeds all ranges.
        """
        clamped = max(0.0, min(1.0, risk_score))
        for lv in sorted(self._levels, key=lambda l: l.level):
            low, high = lv.risk_range
            if low <= clamped < high:
                return lv

        # Edge case: score == 1.0 — match the highest level
        if self._levels:
            return max(self._levels, key=lambda l: l.level)

        # Absolute fallback
        return MitigationLevel(
            level=0, name="LOG_ONLY",
            description="Fallback — no levels configured",
            risk_range=(0.0, 1.0), requires_hitl=False, reversible=True,
        )

    def _get_level_obj(self, level_num: int) -> Optional[MitigationLevel]:
        """Look up a :class:`MitigationLevel` by number."""
        for lv in self._levels:
            if lv.level == level_num:
                return lv
        return None

    @staticmethod
    def _build_parameters(
        level: MitigationLevel,
        device_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build action-specific parameters for a mitigation level.

        Args:
            level: Selected mitigation level.
            device_info: Device metadata.

        Returns:
            Parameters dict to pass to the SDN controller.
        """
        params: Dict[str, Any] = {
            "mitigation_level": level.level,
            "mitigation_name": level.name,
        }

        if level.name == "THROTTLE":
            params["rate_limit_percent"] = 10
        elif level.name == "MICRO_SEGMENT":
            params["vlan_mode"] = "read_only"
            params["preserve_telemetry"] = True
        elif level.name == "RE_AUTHENTICATE":
            params["auth_method"] = "mutual_tls"
            params["force"] = True
        elif level.name == "QUARANTINE":
            params["isolation"] = "full"
            params["allow_management"] = True

        return params

    def _load_policies(self) -> None:
        """Parse mitigation levels and device constraints from YAML."""
        try:
            with open(self._policies_path, "r", encoding="utf-8") as fh:
                policies = yaml.safe_load(fh) or {}
        except FileNotFoundError:
            logger.error("Policies file not found: %s", self._policies_path)
            return
        except yaml.YAMLError as exc:
            logger.error("Failed to parse policies YAML: %s", exc)
            return

        # Parse mitigation levels
        for entry in policies.get("mitigation_levels", []):
            rng = entry.get("risk_score_range", [0.0, 1.0])
            self._levels.append(
                MitigationLevel(
                    level=int(entry.get("level", 0)),
                    name=str(entry.get("name", "UNKNOWN")),
                    description=str(entry.get("description", "")),
                    risk_range=(float(rng[0]), float(rng[1])),
                    requires_hitl=bool(entry.get("requires_hitl", False)),
                    reversible=bool(entry.get("reversible", True)),
                ),
            )

        # Parse device constraints
        self._device_constraints = policies.get("device_constraints", {})

        logger.info(
            "Loaded %d mitigation levels and %d device constraint profiles",
            len(self._levels),
            len(self._device_constraints),
        )


# ------------------------------------------------------------------ #
#  Standalone smoke test                                              #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    project_root = Path(__file__).resolve().parents[3]
    policies = project_root / "config" / "safety_policies.yaml"

    playbook = ActionPlaybook(policies)

    print("All levels:")
    for lv in playbook.get_all_levels():
        print(f"  Level {lv.level}: {lv.name} — {lv.risk_range}")

    device = {"id": "dev-001", "type": "infusion_pump", "criticality": "LIFE_CRITICAL"}
    action = playbook.select_action(risk_score=0.92, device_info=device)
    print(f"\nSelected action for risk=0.92: {action}")

    esc = playbook.escalate(2)
    print(f"Escalate from 2: {esc}")
    de_esc = playbook.de_escalate(2)
    print(f"De-escalate from 2: {de_esc}")
