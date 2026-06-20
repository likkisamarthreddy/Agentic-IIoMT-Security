# -*- coding: utf-8 -*-
"""
Traffic Replay Engine for Cross-Domain Agentic Security (IIoMT).

Simulates ``tcpreplay`` by streaming feature-vector rows over MQTT at a
configurable rate.  Each published message is a JSON document containing
``timestamp``, ``device_id``, ``features``, and ``metadata``.

The engine runs in a **background thread** so the caller's event loop is
never blocked.  Attack patterns can be injected on-the-fly via
:meth:`inject_attack`.

Typical usage::

    import pandas as pd
    from data.traffic_replay import TrafficReplayEngine

    df = pd.read_csv("data/synthetic.csv")
    engine = TrafficReplayEngine(data=df)
    engine.start_replay()
    engine.inject_attack("DDoS", intensity=0.8, duration=10.0)
    # … later …
    engine.stop()
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project-root resolution & config loader
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "settings.yaml"


def _load_config(config_path: Path = _DEFAULT_CONFIG) -> Dict[str, Any]:
    """Load YAML configuration.

    Args:
        config_path: Path to ``settings.yaml``.

    Returns:
        Parsed YAML dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Attack injection profiles
# ---------------------------------------------------------------------------
# Maps attack type → feature perturbations applied during injection.
# Each entry is  feature_name → (multiplier_mean, multiplier_std)  which
# scales the original feature value.

_INJECTION_PROFILES: Dict[str, Dict[str, tuple]] = {
    "DDoS": {
        "rate": (100.0, 30.0),
        "pkt_count": (50.0, 15.0),
        "syn_flag_number": (200.0, 60.0),
        "IAT": (0.01, 0.005),
        "flow_duration": (0.1, 0.05),
    },
    "DoS": {
        "rate": (40.0, 12.0),
        "pkt_count": (25.0, 8.0),
        "syn_flag_number": (80.0, 25.0),
        "IAT": (0.05, 0.02),
        "flow_duration": (0.2, 0.1),
    },
    "Reconnaissance": {
        "rate": (10.0, 3.0),
        "pkt_count": (0.3, 0.1),
        "syn_flag_number": (5.0, 1.5),
        "rst_flag_number": (3.0, 1.0),
        "flow_duration": (0.05, 0.02),
    },
    "Spoofing": {
        "header_length": (3.0, 1.0),
        "rate": (1.5, 0.5),
        "pkt_count": (1.5, 0.4),
    },
    "MITM": {
        "IAT": (5.0, 2.0),
        "ack_flag_number": (3.0, 1.0),
        "flow_duration": (2.0, 0.8),
        "rate": (0.8, 0.2),
    },
}


class TrafficReplayEngine:
    """Streams IIoMT feature vectors over MQTT at a configurable rate.

    The engine reads rows from a pandas ``DataFrame`` and publishes each
    row as a JSON message to the configured MQTT topic.

    Args:
        data: DataFrame of feature vectors (with or without a ``label``
            column).
        rate: Packets published per second.  ``None`` → use config value.
        mqtt_broker: MQTT broker hostname.  ``None`` → use config.
        mqtt_port: MQTT broker port.  ``None`` → use config.
        topic: MQTT topic string.  ``None`` → use config.
        config_path: Override path to ``settings.yaml``.

    Example::

        engine = TrafficReplayEngine(data=df, rate=1000)
        engine.start_replay()
    """

    def __init__(
        self,
        data: pd.DataFrame,
        rate: Optional[int] = None,
        mqtt_broker: Optional[str] = None,
        mqtt_port: Optional[int] = None,
        topic: Optional[str] = None,
        config_path: Optional[Path] = None,
    ) -> None:
        cfg = _load_config(config_path or _DEFAULT_CONFIG)
        data_cfg = cfg.get("data", {})
        mqtt_cfg = cfg.get("mqtt", {})
        devices_cfg = cfg.get("devices", {})

        self._data: pd.DataFrame = data.copy()
        self._rate: int = rate or data_cfg.get("replay_rate", 500)
        self._interval: float = 1.0 / self._rate  # seconds between publishes

        self._broker: str = mqtt_broker or mqtt_cfg.get("broker_host", "localhost")
        self._port: int = mqtt_port or mqtt_cfg.get("broker_port", 1883)
        self._topic: str = topic or mqtt_cfg.get("topics", {}).get(
            "traffic_stream", "iimt/traffic/stream"
        )
        self._keepalive: int = mqtt_cfg.get("keepalive", 60)
        self._qos: int = mqtt_cfg.get("qos", 1)

        # Build a list of valid device IDs from config for round-robin
        default_devices: List[Dict[str, Any]] = devices_cfg.get(
            "default_devices", []
        )
        self._device_ids: List[str] = [
            d.get("id", f"dev-{i:03d}") for i, d in enumerate(default_devices)
        ] or ["dev-001"]

        # Separate feature columns from the label column
        self._label_col: Optional[str] = (
            "label" if "label" in self._data.columns else None
        )
        self._feature_cols: List[str] = [
            c for c in self._data.columns if c != "label"
        ]

        # Threading / control
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._mqtt_client: Any = None  # paho.mqtt.client.Client (lazy)
        self._packets_sent: int = 0

        # Scheduled attack injections
        self._injections: List[Dict[str, Any]] = []

        logger.info(
            "TrafficReplayEngine initialised — rate=%d pkt/s, broker=%s:%d, "
            "topic='%s', rows=%d",
            self._rate,
            self._broker,
            self._port,
            self._topic,
            len(self._data),
        )

    # ------------------------------------------------------------------
    # MQTT helpers
    # ------------------------------------------------------------------

    def _connect_mqtt(self) -> None:
        """Create and connect the MQTT client.

        Raises:
            ConnectionError: If the broker is unreachable.
        """
        import paho.mqtt.client as mqtt

        client_id = f"replay-{uuid.uuid4().hex[:8]}"
        self._mqtt_client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
        )

        def _on_connect(
            client: Any, userdata: Any, flags: Any, rc: Any, properties: Any = None
        ) -> None:
            if hasattr(rc, "value"):
                rc_val = rc.value
            else:
                rc_val = rc
            if rc_val == 0:
                logger.info("MQTT connected to %s:%d", self._broker, self._port)
            else:
                logger.error("MQTT connection failed — rc=%s", rc)

        def _on_disconnect(
            client: Any, userdata: Any, flags: Any = None, rc: Any = None,
            properties: Any = None,
        ) -> None:
            logger.warning("MQTT disconnected — rc=%s", rc)

        self._mqtt_client.on_connect = _on_connect
        self._mqtt_client.on_disconnect = _on_disconnect

        try:
            self._mqtt_client.connect(self._broker, self._port, self._keepalive)
            self._mqtt_client.loop_start()
        except Exception as exc:
            raise ConnectionError(
                f"Cannot connect to MQTT broker at "
                f"{self._broker}:{self._port} — {exc}"
            ) from exc

    def _disconnect_mqtt(self) -> None:
        """Gracefully disconnect the MQTT client."""
        if self._mqtt_client is not None:
            try:
                self._mqtt_client.loop_stop()
                self._mqtt_client.disconnect()
            except Exception:
                logger.debug("MQTT disconnect raised (may be expected)")
            self._mqtt_client = None

    # ------------------------------------------------------------------
    # Packet construction
    # ------------------------------------------------------------------

    def _build_packet(self, row: pd.Series, row_idx: int) -> Dict[str, Any]:
        """Build a JSON-serialisable packet from a DataFrame row.

        Args:
            row: Single DataFrame row.
            row_idx: Row index (used for device round-robin).

        Returns:
            Dictionary with keys ``timestamp``, ``device_id``,
            ``features``, ``metadata``.
        """
        device_id = self._device_ids[row_idx % len(self._device_ids)]
        features = {col: float(row[col]) for col in self._feature_cols}

        packet: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device_id": device_id,
            "features": features,
            "metadata": {
                "sequence_num": self._packets_sent,
                "replay_rate": self._rate,
            },
        }
        if self._label_col and self._label_col in row.index:
            packet["metadata"]["ground_truth_label"] = str(row[self._label_col])
        return packet

    # ------------------------------------------------------------------
    # Attack injection helpers
    # ------------------------------------------------------------------

    def _apply_injection(self, features: Dict[str, float]) -> Dict[str, float]:
        """Mutate feature values according to active attack injections.

        Args:
            features: Original feature dict (modified in-place and returned).

        Returns:
            Possibly-mutated feature dict.
        """
        now = time.monotonic()
        with self._lock:
            active = [
                inj for inj in self._injections
                if inj["start"] <= now < inj["end"]
            ]

        if not active:
            return features

        rng = np.random.default_rng()
        for inj in active:
            profile = _INJECTION_PROFILES.get(inj["type"], {})
            intensity: float = inj["intensity"]
            for feat, (mul_mean, mul_std) in profile.items():
                if feat in features:
                    multiplier = rng.normal(mul_mean * intensity, mul_std * intensity)
                    features[feat] = abs(features[feat] * max(multiplier, 0.01))
        return features

    # ------------------------------------------------------------------
    # Replay loop (runs in a background thread)
    # ------------------------------------------------------------------

    def _replay_loop(self) -> None:
        """Stream rows from the DataFrame at the configured rate."""
        logger.info("Replay loop started — streaming %d rows", len(self._data))
        idx = 0
        total = len(self._data)

        while not self._stop_event.is_set():
            if idx >= total:
                logger.info("Reached end of data — looping from beginning")
                idx = 0

            row = self._data.iloc[idx]
            packet = self._build_packet(row, idx)

            # Apply any active attack injections
            packet["features"] = self._apply_injection(packet["features"])

            payload = json.dumps(packet, default=str)
            try:
                self._mqtt_client.publish(
                    self._topic, payload, qos=self._qos
                )
            except Exception as exc:
                logger.error("MQTT publish error: %s", exc)

            self._packets_sent += 1
            idx += 1

            if self._packets_sent % 5000 == 0:
                logger.info("Published %d packets so far", self._packets_sent)

            # Throttle to target rate
            self._stop_event.wait(timeout=self._interval)

        logger.info(
            "Replay loop stopped — total packets sent: %d", self._packets_sent
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_replay(self) -> None:
        """Begin streaming data in a background thread.

        Connects to the MQTT broker and starts a daemon thread that
        publishes one row per ``1 / rate`` seconds.

        Raises:
            RuntimeError: If replay is already running.
            ConnectionError: If the MQTT broker is unreachable.
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Replay is already running")

        self._stop_event.clear()
        self._packets_sent = 0
        self._connect_mqtt()

        self._thread = threading.Thread(
            target=self._replay_loop, name="TrafficReplay", daemon=True
        )
        self._thread.start()
        logger.info("Traffic replay started at %d pkt/s", self._rate)

    def inject_attack(
        self,
        attack_type: str,
        intensity: float = 1.0,
        duration: float = 10.0,
    ) -> None:
        """Schedule an attack injection into the live stream.

        While the injection is active, feature values for published packets
        are perturbed according to the attack profile and *intensity*
        multiplier.

        Args:
            attack_type: One of ``DDoS``, ``DoS``, ``Reconnaissance``,
                ``Spoofing``, ``MITM``.
            intensity: Scaling factor ∈ (0, 1].  ``1.0`` = full strength.
            duration: How long (seconds) the injection lasts.

        Raises:
            ValueError: If *attack_type* is unknown.
        """
        if attack_type not in _INJECTION_PROFILES:
            raise ValueError(
                f"Unknown attack type '{attack_type}'. "
                f"Supported: {list(_INJECTION_PROFILES.keys())}"
            )
        intensity = float(np.clip(intensity, 0.01, 1.0))
        now = time.monotonic()
        injection = {
            "type": attack_type,
            "intensity": intensity,
            "start": now,
            "end": now + duration,
            "id": uuid.uuid4().hex[:8],
        }
        with self._lock:
            self._injections.append(injection)

        logger.warning(
            "Attack injection scheduled — type=%s, intensity=%.2f, "
            "duration=%.1fs, id=%s",
            attack_type,
            intensity,
            duration,
            injection["id"],
        )

    def stop(self) -> None:
        """Stop the replay engine and disconnect MQTT.

        Safe to call even if the engine is not running.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._disconnect_mqtt()
        with self._lock:
            self._injections.clear()
        logger.info("Traffic replay engine stopped")

    @property
    def is_running(self) -> bool:
        """Whether the replay thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def packets_sent(self) -> int:
        """Number of packets published since the last :meth:`start_replay`."""
        return self._packets_sent


# ===================================================================
# Quick smoke-test
# ===================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    from data.synthetic_generator import SyntheticIIoMTGenerator

    gen = SyntheticIIoMTGenerator()
    df = gen.generate_ciciomt_data(100)

    engine = TrafficReplayEngine(data=df, rate=10)
    print(f"Engine ready — rows: {len(df)}, rate: {engine._rate}")
    print("Call engine.start_replay() to begin (requires MQTT broker).")


# ===================================================================
# TCPREPLAY ENGINE (FOR MININET / CONTAINERCERT)
# ===================================================================
import subprocess

class TCPReplayEngine:
    """Streams PCAP files over Mininet virtual interfaces using tcpreplay.
    
    This fulfills the paper's requirement to inject actual network pcaps
    over simulated links, ensuring realistic propagation delays and TCP
    windowing behaviors.
    """
    def __init__(self, pcap_path: str, interface: str, rate_mbps: float = 10.0) -> None:
        self.pcap_path = pcap_path
        self.interface = interface
        self.rate_mbps = rate_mbps
        self._process: Optional[subprocess.Popen] = None
        self._is_running = False

    def start_replay(self) -> None:
        if self._is_running:
            raise RuntimeError("TCPReplay is already running")
        
        cmd = [
            "tcpreplay",
            "-i", self.interface,
            "--mbps", str(self.rate_mbps),
            self.pcap_path
        ]
        
        logger.info(f"Starting tcpreplay: {' '.join(cmd)}")
        self._process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self._is_running = True

    def stop(self) -> None:
        if self._process and self._is_running:
            self._process.terminate()
            self._process.wait(timeout=5.0)
            self._is_running = False
            logger.info("TCPReplay engine stopped")

    @property
    def is_running(self) -> bool:
        return self._is_running
