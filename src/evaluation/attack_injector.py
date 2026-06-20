"""
Phase 3 attack injection engine for IIoMT security evaluation.

Generates synthetic attack traffic patterns (DDoS, Spoofing, MITM)
that are injected into the traffic stream for ground-truth evaluation
of the CNN-BiGRU + ReAct detection pipeline.

Each injector method modifies the feature vector to exhibit signatures
consistent with real CICIoMT2024 attack classes, while logging exact
injection timestamps so that detection metrics can be computed against
known ground-truth labels.

Typical usage::

    injector = AttackInjector(traffic_replay_engine)
    injector.inject_ddos(target_device="dev-001", intensity="high", duration_sec=60)
    # — or —
    injector.run_all_scenarios()
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "settings.yaml"


def _load_config(path: Path | None = None) -> Dict[str, Any]:
    """Load and return the YAML configuration dictionary."""
    config_path = path or _DEFAULT_CONFIG
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Attack label constants (aligned with settings.yaml attack_types)
# ---------------------------------------------------------------------------
class AttackLabel:
    """Integer labels for the 6-class CICIoMT2024 taxonomy."""

    BENIGN = 0
    DDOS = 1
    DOS = 2
    RECON = 3
    SPOOFING = 4
    MITM = 5


# ---------------------------------------------------------------------------
# Injection record
# ---------------------------------------------------------------------------
@dataclass
class InjectionRecord:
    """Ground-truth record for a single injected attack packet."""

    timestamp: float
    attack_type: str
    attack_label: int
    target_device: str
    feature_vector: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# TrafficReplayEngine protocol (structural typing)
# ---------------------------------------------------------------------------
class TrafficReplayProtocol(Protocol):
    """Minimal interface expected from the traffic replay engine."""

    def inject_packet(
        self, feature_vector: np.ndarray, label: int, device_id: str
    ) -> None:
        """Inject a single feature vector into the traffic stream."""
        ...

    @property
    def num_features(self) -> int:
        """Number of features in each traffic vector."""
        ...


# ---------------------------------------------------------------------------
# Feature generation helpers
# ---------------------------------------------------------------------------
class _FeatureGenerator:
    """Generate synthetic feature vectors that mimic CICIoMT2024 attack signatures.

    Feature indices (representative subset of the 46-feature schema):
        0: flow_duration
        1: packet_count
        2: bytes_per_packet
        3: inter_arrival_time (IAT) mean
        4: IAT std
        5: fwd_packet_count
        6: bwd_packet_count
        7: syn_flag_count
        8: ack_flag_count
        9: rst_flag_count
        10: payload_bytes_mean
        11: payload_bytes_std
        12-45: additional flow/header features
    """

    def __init__(self, num_features: int = 46, seed: int = 42) -> None:
        self.num_features = num_features
        self._rng = np.random.default_rng(seed)

    def benign(self) -> np.ndarray:
        """Generate a benign traffic feature vector."""
        vec = self._rng.normal(0.0, 0.1, size=self.num_features)
        vec[1] = abs(self._rng.normal(50, 10))       # moderate packet count
        vec[3] = abs(self._rng.normal(0.05, 0.01))    # normal IAT
        vec[7] = abs(self._rng.normal(1, 0.5))        # low SYN
        return vec.astype(np.float32)

    def ddos(self, intensity: str = "high") -> np.ndarray:
        """Generate a DDoS attack feature vector.

        Characteristics:
            - Very high packet count
            - Very low inter-arrival time
            - High SYN flag count (SYN flood)
            - Small payload (volumetric)
        """
        scale = {"low": 1.0, "medium": 2.0, "high": 5.0}.get(intensity, 3.0)
        vec = self._rng.normal(0.0, 0.2, size=self.num_features)

        vec[0] = abs(self._rng.normal(100, 20))                # long flow
        vec[1] = abs(self._rng.normal(5000 * scale, 500))       # very high pkt count
        vec[2] = abs(self._rng.normal(64, 10))                  # small pkts
        vec[3] = abs(self._rng.normal(0.001 / scale, 0.0005))   # tiny IAT
        vec[4] = abs(self._rng.normal(0.0005, 0.0001))          # low IAT std
        vec[5] = abs(self._rng.normal(4000 * scale, 400))       # mostly fwd
        vec[6] = abs(self._rng.normal(10, 3))                   # few bwd
        vec[7] = abs(self._rng.normal(3000 * scale, 300))       # SYN flood
        vec[8] = abs(self._rng.normal(5, 2))                    # low ACK
        vec[10] = abs(self._rng.normal(0, 5))                   # empty payload

        return vec.astype(np.float32)

    def spoofing(self) -> np.ndarray:
        """Generate a device-spoofing feature vector.

        Characteristics:
            - Inconsistent header fingerprints
            - Normal packet rate but abnormal source patterns
            - High RST flag count (rejected connections)
            - Payload mimics legitimate but with subtle differences
        """
        vec = self._rng.normal(0.0, 0.15, size=self.num_features)

        vec[0] = abs(self._rng.normal(30, 8))           # moderate flow
        vec[1] = abs(self._rng.normal(80, 20))           # normal-ish pkt count
        vec[2] = abs(self._rng.normal(200, 50))          # variable pkt size
        vec[3] = abs(self._rng.normal(0.04, 0.02))       # slightly off IAT
        vec[4] = abs(self._rng.normal(0.03, 0.01))       # high IAT variance
        vec[7] = abs(self._rng.normal(15, 5))            # moderate SYN
        vec[8] = abs(self._rng.normal(10, 4))            # moderate ACK
        vec[9] = abs(self._rng.normal(20, 8))            # high RST (rejected)
        vec[10] = abs(self._rng.normal(150, 40))         # reasonable payload
        vec[11] = abs(self._rng.normal(80, 20))          # high payload std

        # Fingerprint inconsistency (features 30-35 simulate device profile)
        for i in range(30, min(36, self.num_features)):
            vec[i] = self._rng.uniform(-2.0, 2.0)

        return vec.astype(np.float32)

    def mitm(self) -> np.ndarray:
        """Generate a MITM (man-in-the-middle) feature vector.

        Characteristics:
            - Unusual IAT (interception delay)
            - Duplicated flows (forward ≈ backward)
            - Modified payload (slightly altered bytes)
            - Higher-than-normal flow duration
        """
        vec = self._rng.normal(0.0, 0.12, size=self.num_features)

        vec[0] = abs(self._rng.normal(60, 15))           # longer flow
        vec[1] = abs(self._rng.normal(120, 30))          # moderate pkts
        vec[2] = abs(self._rng.normal(300, 80))          # larger pkts
        vec[3] = abs(self._rng.normal(0.08, 0.03))       # higher IAT (delay)
        vec[4] = abs(self._rng.normal(0.05, 0.02))       # high IAT std
        vec[5] = abs(self._rng.normal(60, 15))           # fwd ≈ bwd
        vec[6] = abs(self._rng.normal(55, 15))           # duplicated
        vec[7] = abs(self._rng.normal(5, 2))             # normal SYN
        vec[8] = abs(self._rng.normal(50, 10))           # high ACK (relay)
        vec[10] = abs(self._rng.normal(280, 60))         # modified payload
        vec[11] = abs(self._rng.normal(100, 30))         # high payload std

        return vec.astype(np.float32)


# ---------------------------------------------------------------------------
# AttackInjector
# ---------------------------------------------------------------------------
class AttackInjector:
    """Phase 3 attack-scenario injection engine.

    Injects synthetic attack traffic into the evaluation pipeline,
    recording ground-truth injection timestamps for confusion-matrix
    construction.

    Args:
        traffic_replay_engine: Object implementing ``TrafficReplayProtocol``
            (must support ``inject_packet`` and ``num_features``).
        config: Optional pre-loaded settings dict; defaults to
            ``config/settings.yaml``.
    """

    def __init__(
        self,
        traffic_replay_engine: Any,
        config: Dict[str, Any] | None = None,
    ) -> None:
        self._engine = traffic_replay_engine
        self._config = config or _load_config()
        self._num_features: int = self._config.get("data", {}).get(
            "num_features", 46
        )
        self._replay_rate: int = self._config.get("data", {}).get(
            "replay_rate", 500
        )
        self._gen = _FeatureGenerator(
            num_features=self._num_features,
            seed=self._config.get("data", {}).get("random_seed", 42),
        )
        self._injection_log: List[InjectionRecord] = []

        # Load attack scenarios from config
        eval_cfg = self._config.get("evaluation", {})
        self._scenarios: List[Dict[str, Any]] = eval_cfg.get(
            "attack_scenarios", []
        )

        logger.info(
            "AttackInjector initialised — %d features, %d pkt/s replay rate, "
            "%d configured scenario(s)",
            self._num_features,
            self._replay_rate,
            len(self._scenarios),
        )

    # -- Core injectors -----------------------------------------------------

    def inject_ddos(
        self,
        target_device: str,
        intensity: str = "high",
        duration_sec: float = 60.0,
    ) -> List[InjectionRecord]:
        """Inject a DDoS flood attack pattern.

        Generates high-rate, small-packet SYN flood traffic targeting
        *target_device* for the specified duration.

        Args:
            target_device: Device ID to target (e.g. ``"dev-001"``).
            intensity: One of ``"low"``, ``"medium"``, ``"high"``.
            duration_sec: Duration of the attack in seconds.

        Returns:
            List of ``InjectionRecord`` instances for ground-truth.
        """
        logger.info(
            "Injecting DDoS — target=%s, intensity=%s, duration=%ss",
            target_device, intensity, duration_sec,
        )
        rate_multiplier = {"low": 1, "medium": 3, "high": 10}.get(intensity, 5)
        pps = self._replay_rate * rate_multiplier
        interval = 1.0 / pps if pps > 0 else 0.001

        records: List[InjectionRecord] = []
        t_start = time.monotonic()

        while (time.monotonic() - t_start) < duration_sec:
            vec = self._gen.ddos(intensity)
            record = InjectionRecord(
                timestamp=time.time(),
                attack_type="DDoS",
                attack_label=AttackLabel.DDOS,
                target_device=target_device,
                feature_vector=vec,
                metadata={"intensity": intensity, "pps": pps},
            )
            try:
                self._engine.inject_packet(
                    vec, AttackLabel.DDOS, target_device
                )
            except Exception:
                logger.exception("Failed to inject DDoS packet")
                break

            records.append(record)
            self._injection_log.append(record)

            # Simulate burst timing (don't actually sleep at full rate
            # in evaluation mode — batch inject)
            if interval > 0.001:
                time.sleep(interval)

        logger.info(
            "DDoS injection complete — %d packets over %.1fs",
            len(records), time.monotonic() - t_start,
        )
        return records

    def inject_spoofing(
        self,
        target_device: str,
        duration_sec: float = 45.0,
    ) -> List[InjectionRecord]:
        """Inject a device-spoofing attack pattern.

        Generates traffic with forged MAC/IP headers and inconsistent
        device fingerprints to simulate identity spoofing.

        Args:
            target_device: Device ID to target.
            duration_sec: Duration of the attack in seconds.

        Returns:
            List of ``InjectionRecord`` instances.
        """
        logger.info(
            "Injecting Spoofing — target=%s, duration=%ss",
            target_device, duration_sec,
        )
        # Spoofing is lower-rate than DDoS but persistent
        pps = max(self._replay_rate // 5, 10)
        interval = 1.0 / pps

        records: List[InjectionRecord] = []
        t_start = time.monotonic()

        while (time.monotonic() - t_start) < duration_sec:
            vec = self._gen.spoofing()
            record = InjectionRecord(
                timestamp=time.time(),
                attack_type="Spoofing",
                attack_label=AttackLabel.SPOOFING,
                target_device=target_device,
                metadata={"pps": pps},
            )
            try:
                self._engine.inject_packet(
                    vec, AttackLabel.SPOOFING, target_device
                )
            except Exception:
                logger.exception("Failed to inject Spoofing packet")
                break

            records.append(record)
            self._injection_log.append(record)

            if interval > 0.001:
                time.sleep(interval)

        logger.info(
            "Spoofing injection complete — %d packets over %.1fs",
            len(records), time.monotonic() - t_start,
        )
        return records

    def inject_mitm(
        self,
        target_device: str,
        duration_sec: float = 30.0,
    ) -> List[InjectionRecord]:
        """Inject a man-in-the-middle (MITM) attack pattern.

        Generates traffic exhibiting interception delay, duplicated
        bidirectional flows, and subtly modified payloads.

        Args:
            target_device: Device ID to target.
            duration_sec: Duration of the attack in seconds.

        Returns:
            List of ``InjectionRecord`` instances.
        """
        logger.info(
            "Injecting MITM — target=%s, duration=%ss",
            target_device, duration_sec,
        )
        # MITM is moderate-rate with relay characteristics
        pps = max(self._replay_rate // 3, 20)
        interval = 1.0 / pps

        records: List[InjectionRecord] = []
        t_start = time.monotonic()

        while (time.monotonic() - t_start) < duration_sec:
            vec = self._gen.mitm()
            record = InjectionRecord(
                timestamp=time.time(),
                attack_type="MITM",
                attack_label=AttackLabel.MITM,
                target_device=target_device,
                metadata={"pps": pps},
            )
            try:
                self._engine.inject_packet(
                    vec, AttackLabel.MITM, target_device
                )
            except Exception:
                logger.exception("Failed to inject MITM packet")
                break

            records.append(record)
            self._injection_log.append(record)

            if interval > 0.001:
                time.sleep(interval)

        logger.info(
            "MITM injection complete — %d packets over %.1fs",
            len(records), time.monotonic() - t_start,
        )
        return records

    # -- Scenario runners ---------------------------------------------------

    def run_scenario(
        self, scenario_config: Dict[str, Any]
    ) -> List[InjectionRecord]:
        """Execute a single attack scenario from configuration.

        Args:
            scenario_config: Dict with keys ``type``, ``intensity``,
                ``duration_sec``, and optionally ``target_device``.

        Returns:
            List of injection records produced by the scenario.

        Raises:
            ValueError: If the scenario type is unrecognised.
        """
        attack_type = scenario_config.get("type", "").lower()
        intensity = scenario_config.get("intensity", "medium")
        duration = scenario_config.get("duration_sec", 30)
        target = scenario_config.get("target_device", "dev-001")
        name = scenario_config.get("name", attack_type)

        logger.info("Running scenario: %s", name)

        if attack_type == "ddos":
            return self.inject_ddos(target, intensity, duration)
        elif attack_type == "spoofing":
            return self.inject_spoofing(target, duration)
        elif attack_type == "mitm":
            return self.inject_mitm(target, duration)
        else:
            raise ValueError(f"Unknown attack type: {attack_type!r}")

    def run_all_scenarios(self) -> Dict[str, List[InjectionRecord]]:
        """Run all configured attack scenarios sequentially.

        Reads scenarios from ``evaluation.attack_scenarios`` in
        ``settings.yaml``.

        Returns:
            Dict mapping scenario name → list of injection records.
        """
        logger.info("Running %d attack scenario(s)", len(self._scenarios))
        results: Dict[str, List[InjectionRecord]] = {}

        for scenario in self._scenarios:
            name = scenario.get("name", scenario.get("type", "unknown"))
            try:
                records = self.run_scenario(scenario)
                results[name] = records
                logger.info(
                    "Scenario '%s' complete — %d packets injected",
                    name, len(records),
                )
            except Exception:
                logger.exception("Scenario '%s' failed", name)
                results[name] = []

            # Brief pause between scenarios
            time.sleep(1.0)

        logger.info("All scenarios complete")
        return results

    # -- Accessors ----------------------------------------------------------

    @property
    def injection_log(self) -> List[InjectionRecord]:
        """Return the complete injection log for ground-truth evaluation."""
        return list(self._injection_log)

    @property
    def injection_count(self) -> int:
        """Return the total number of injected packets."""
        return len(self._injection_log)

    def get_injection_summary(self) -> Dict[str, int]:
        """Return per-attack-type injection counts.

        Returns:
            Dict mapping attack type name → injection count.
        """
        summary: Dict[str, int] = {}
        for record in self._injection_log:
            summary[record.attack_type] = (
                summary.get(record.attack_type, 0) + 1
            )
        return summary

    def clear_log(self) -> None:
        """Clear the injection log."""
        self._injection_log.clear()
        logger.info("Injection log cleared")

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"<AttackInjector scenarios={len(self._scenarios)} "
            f"injected={self.injection_count}>"
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    class _MockReplayEngine:
        """Minimal mock for testing."""

        num_features: int = 46
        _packets: List[Any] = []

        def inject_packet(
            self, feature_vector: np.ndarray, label: int, device_id: str
        ) -> None:
            self._packets.append((label, device_id))

    engine = _MockReplayEngine()
    injector = AttackInjector(engine)

    # Run a short DDoS burst (2 seconds)
    records = injector.inject_ddos("dev-001", intensity="low", duration_sec=2.0)
    logger.info("DDoS records: %d", len(records))

    # Run a short spoofing burst
    records = injector.inject_spoofing("dev-002", duration_sec=1.0)
    logger.info("Spoofing records: %d", len(records))

    # Run a short MITM burst
    records = injector.inject_mitm("dev-003", duration_sec=1.0)
    logger.info("MITM records: %d", len(records))

    summary = injector.get_injection_summary()
    logger.info("Injection summary: %s", summary)
    logger.info("Smoke test complete ✓")
