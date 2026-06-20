# -*- coding: utf-8 -*-
"""
Synthetic IIoMT Traffic Generator.

Generates synthetic network-traffic DataFrames that follow the feature schemas
of the **CICIoMT2024** and **Edge-IIoTset** benchmark datasets.  Six traffic
classes are supported—Benign, DDoS, DoS, Reconnaissance, Spoofing, and MITM—
each with statistically distinct signatures so that downstream classifiers can
learn meaningful decision boundaries.

Typical usage::

    from data.synthetic_generator import SyntheticIIoMTGenerator

    gen = SyntheticIIoMTGenerator()
    df  = gen.generate_combined_dataset(num_samples=50_000)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project-root resolution
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "settings.yaml"


def _load_config(config_path: Path = _DEFAULT_CONFIG) -> Dict[str, Any]:
    """Load and return the YAML configuration dictionary.

    Args:
        config_path: Absolute or relative path to ``settings.yaml``.

    Returns:
        Parsed YAML dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg: Dict[str, Any] = yaml.safe_load(fh)
    logger.info("Configuration loaded from %s", config_path)
    return cfg


# ===================================================================
# CICIoMT2024 feature schema (~46 columns)
# ===================================================================
CICIOMT_FEATURES: List[str] = [
    "flow_duration",
    "header_length",
    "protocol_type",
    "rate",
    "syn_flag_number",
    "ack_flag_number",
    "fin_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "urg_flag_number",
    # Protocol one-hots
    "HTTP",
    "HTTPS",
    "DNS",
    "Telnet",
    "SMTP",
    "SSH",
    "IRC",
    "TCP",
    "UDP",
    "DHCP",
    "ARP",
    "ICMP",
    "IGMP",
    "IPv",
    "LLC",
    # Aggregate statistics
    "tot_sum",
    "min",
    "max",
    "avg",
    "std",
    "tot_size",
    "IAT",
    "number",
    "magnitude",
    "radius",
    "covariance",
    "variance",
    "weight",
    # Packet-level
    "pkt_count",
    # Derived flow features (pad to ≈46)
    "fwd_pkt_len_mean",
    "bwd_pkt_len_mean",
    "flow_byts_per_sec",
    "flow_pkts_per_sec",
    "fwd_iat_mean",
    "bwd_iat_mean",
    "active_mean",
]

assert len(CICIOMT_FEATURES) == 46, (
    f"Expected 46 CICIoMT features, got {len(CICIOMT_FEATURES)}"
)

# ===================================================================
# Edge-IIoTset additional features (industrial / Modbus sensors)
# ===================================================================
_EDGE_EXTRA_FEATURES: List[str] = [
    "modbus_function_code",
    "modbus_length",
    "modbus_unit_id",
    "modbus_response_time",
    "sensor_temperature",
    "sensor_humidity",
    "sensor_pressure",
    "sensor_vibration",
    "sensor_voltage",
    "sensor_current",
    "plc_register_value",
    "plc_coil_status",
    "scada_command_type",
    "mqtt_topic_depth",
    "coap_response_code",
]

EDGE_IIOTSET_FEATURES: List[str] = CICIOMT_FEATURES + _EDGE_EXTRA_FEATURES
assert len(EDGE_IIOTSET_FEATURES) == 61, (
    f"Expected 61 Edge-IIoTset features, got {len(EDGE_IIOTSET_FEATURES)}"
)

# ===================================================================
# Attack-type statistical profiles
# ===================================================================
# Each profile maps feature-name → (mean, std) used by np.random.normal.
# Features not listed fall back to class-specific defaults.

_ATTACK_PROFILES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "Benign": {
        "flow_duration": (5000.0, 3000.0),
        "rate": (50.0, 30.0),
        "pkt_count": (20.0, 15.0),
        "syn_flag_number": (1.0, 0.5),
        "ack_flag_number": (10.0, 5.0),
        "fin_flag_number": (1.0, 0.5),
        "rst_flag_number": (0.1, 0.1),
        "IAT": (100.0, 50.0),
        "header_length": (40.0, 10.0),
        "tot_size": (1500.0, 800.0),
        "magnitude": (5.0, 2.0),
        "flow_byts_per_sec": (5000.0, 3000.0),
        "flow_pkts_per_sec": (20.0, 10.0),
    },
    "DDoS": {
        "flow_duration": (500.0, 200.0),       # very short bursts
        "rate": (5000.0, 2000.0),               # extremely high rate
        "pkt_count": (1000.0, 500.0),           # huge packet count
        "syn_flag_number": (500.0, 200.0),      # SYN flood signature
        "ack_flag_number": (2.0, 1.0),
        "fin_flag_number": (0.5, 0.3),
        "rst_flag_number": (50.0, 20.0),        # many resets
        "IAT": (0.5, 0.3),                      # near-zero inter-arrival
        "header_length": (40.0, 5.0),
        "tot_size": (40000.0, 15000.0),
        "magnitude": (50.0, 15.0),
        "flow_byts_per_sec": (500000.0, 200000.0),
        "flow_pkts_per_sec": (5000.0, 2000.0),
    },
    "DoS": {
        "flow_duration": (1000.0, 400.0),
        "rate": (2000.0, 800.0),
        "pkt_count": (500.0, 200.0),
        "syn_flag_number": (200.0, 80.0),
        "ack_flag_number": (5.0, 3.0),
        "fin_flag_number": (1.0, 0.5),
        "rst_flag_number": (30.0, 10.0),
        "IAT": (2.0, 1.0),
        "header_length": (40.0, 5.0),
        "tot_size": (20000.0, 8000.0),
        "magnitude": (30.0, 10.0),
        "flow_byts_per_sec": (200000.0, 80000.0),
        "flow_pkts_per_sec": (2000.0, 800.0),
    },
    "Reconnaissance": {
        "flow_duration": (200.0, 100.0),        # very short probes
        "rate": (500.0, 200.0),
        "pkt_count": (5.0, 3.0),                # few packets per probe
        "syn_flag_number": (3.0, 1.5),          # SYN-scan signature
        "ack_flag_number": (0.5, 0.3),
        "fin_flag_number": (0.2, 0.1),
        "rst_flag_number": (2.0, 1.0),          # probes get RSTs back
        "IAT": (5.0, 3.0),
        "header_length": (40.0, 5.0),
        "tot_size": (200.0, 100.0),             # very small payloads
        "magnitude": (2.0, 1.0),
        "flow_byts_per_sec": (10000.0, 5000.0),
        "flow_pkts_per_sec": (500.0, 200.0),
    },
    "Spoofing": {
        "flow_duration": (3000.0, 1500.0),
        "rate": (80.0, 40.0),
        "pkt_count": (30.0, 15.0),
        "syn_flag_number": (2.0, 1.0),
        "ack_flag_number": (15.0, 7.0),
        "fin_flag_number": (1.0, 0.5),
        "rst_flag_number": (0.5, 0.3),
        "IAT": (80.0, 40.0),
        "header_length": (120.0, 40.0),         # forged / oversized headers
        "tot_size": (3000.0, 1500.0),
        "magnitude": (8.0, 4.0),
        "flow_byts_per_sec": (8000.0, 4000.0),
        "flow_pkts_per_sec": (30.0, 15.0),
    },
    "MITM": {
        "flow_duration": (8000.0, 4000.0),      # longer interception
        "rate": (60.0, 25.0),
        "pkt_count": (40.0, 20.0),
        "syn_flag_number": (2.0, 1.0),
        "ack_flag_number": (25.0, 10.0),        # double ACKs (relay)
        "fin_flag_number": (2.0, 1.0),
        "rst_flag_number": (1.0, 0.5),
        "IAT": (300.0, 150.0),                  # irregular IAT (relay)
        "header_length": (60.0, 20.0),
        "tot_size": (4000.0, 2000.0),
        "magnitude": (10.0, 5.0),
        "flow_byts_per_sec": (6000.0, 3000.0),
        "flow_pkts_per_sec": (15.0, 8.0),
    },
}


class SyntheticIIoMTGenerator:
    """Generates synthetic IIoMT network-traffic DataFrames.

    The generator supports two benchmark schemas:

    * **CICIoMT2024** — 46 CICFlowMeter-style features.
    * **Edge-IIoTset** — 61 features (46 flow + 15 industrial/Modbus/sensor).

    Attack-class ratios are loaded from ``config/settings.yaml`` under
    ``data.synthetic``.

    Args:
        config_path: Path to the project configuration YAML.
            Defaults to ``<project_root>/config/settings.yaml``.

    Example::

        gen = SyntheticIIoMTGenerator()
        ciciomt_df = gen.generate_ciciomt_data(50_000)
        edge_df    = gen.generate_edge_iiotset_data(50_000)
        combined   = gen.generate_combined_dataset(50_000)
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self._config = _load_config(config_path or _DEFAULT_CONFIG)
        data_cfg = self._config.get("data", {})

        self._attack_types: List[str] = data_cfg.get(
            "attack_types",
            ["Benign", "DDoS", "DoS", "Reconnaissance", "Spoofing", "MITM"],
        )
        synth_cfg = data_cfg.get("synthetic", {})
        self._ratios: Dict[str, float] = {
            "Benign": synth_cfg.get("benign_ratio", 0.40),
            "DDoS": synth_cfg.get("ddos_ratio", 0.15),
            "DoS": synth_cfg.get("dos_ratio", 0.15),
            "Reconnaissance": synth_cfg.get("recon_ratio", 0.10),
            "Spoofing": synth_cfg.get("spoofing_ratio", 0.10),
            "MITM": synth_cfg.get("mitm_ratio", 0.10),
        }
        self._seed: int = data_cfg.get("random_seed", 42)
        self._rng = np.random.default_rng(self._seed)
        logger.info(
            "SyntheticIIoMTGenerator initialised — attack ratios: %s",
            self._ratios,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _class_sample_counts(self, num_samples: int) -> Dict[str, int]:
        """Compute per-class sample counts from configured ratios.

        Any residual samples (due to rounding) are assigned to *Benign*.

        Args:
            num_samples: Total number of samples to generate.

        Returns:
            Dictionary mapping attack-type name → sample count.
        """
        counts: Dict[str, int] = {}
        for label, ratio in self._ratios.items():
            counts[label] = int(num_samples * ratio)
        remainder = num_samples - sum(counts.values())
        counts["Benign"] = counts.get("Benign", 0) + remainder
        return counts

    def _generate_features(
        self,
        feature_names: List[str],
        attack_type: str,
        n: int,
    ) -> np.ndarray:
        """Generate *n* feature rows for a single attack type.

        For features that appear in ``_ATTACK_PROFILES[attack_type]`` the
        corresponding (mean, std) is used.  All other features receive a
        generic profile that still varies by class to preserve
        separability.

        Args:
            feature_names: Ordered list of feature column names.
            attack_type: One of the six supported attack labels.
            n: Number of samples.

        Returns:
            ``np.ndarray`` of shape ``(n, len(feature_names))``.
        """
        profile = _ATTACK_PROFILES.get(attack_type, _ATTACK_PROFILES["Benign"])
        # Deterministic but class-dependent seed offset for reproducibility
        class_offset = hash(attack_type) % 10_000
        data = np.empty((n, len(feature_names)), dtype=np.float64)

        for col_idx, feat in enumerate(feature_names):
            if feat in profile:
                mean, std = profile[feat]
            else:
                # Generic per-class variation derived from feature index
                mean = 10.0 + col_idx * 0.5 + class_offset * 0.01
                std = max(mean * 0.3, 1.0)

            col = self._rng.normal(loc=mean, scale=std, size=n)

            # Protocol one-hot columns should be binary 0/1
            if feat in {
                "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC",
                "TCP", "UDP", "DHCP", "ARP", "ICMP", "IGMP", "IPv", "LLC",
            }:
                col = (col > mean).astype(np.float64)
            # Modbus / PLC binary columns
            elif feat == "plc_coil_status":
                col = (col > mean).astype(np.float64)
            else:
                # Clamp to non-negative for count / size features
                np.clip(col, 0.0, None, out=col)

            data[:, col_idx] = col

        return data

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_ciciomt_data(self, num_samples: int = 100_000) -> pd.DataFrame:
        """Generate synthetic CICIoMT2024-style network-flow data.

        Args:
            num_samples: Total rows to generate across all classes.

        Returns:
            DataFrame with 46 numeric feature columns and a ``label``
            column (str).

        Raises:
            ValueError: If *num_samples* ≤ 0.
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")

        counts = self._class_sample_counts(num_samples)
        frames: List[pd.DataFrame] = []

        for attack_type, n in counts.items():
            if n == 0:
                continue
            arr = self._generate_features(CICIOMT_FEATURES, attack_type, n)
            df = pd.DataFrame(arr, columns=CICIOMT_FEATURES)
            df["label"] = attack_type
            frames.append(df)
            logger.debug("CICIoMT — generated %d samples for '%s'", n, attack_type)

        result = pd.concat(frames, ignore_index=True)
        result = result.sample(frac=1.0, random_state=self._seed).reset_index(
            drop=True
        )
        logger.info(
            "CICIoMT2024 dataset generated — %d samples, %d features",
            len(result),
            len(CICIOMT_FEATURES),
        )
        return result

    def generate_edge_iiotset_data(
        self, num_samples: int = 100_000
    ) -> pd.DataFrame:
        """Generate synthetic Edge-IIoTset-style data with Modbus features.

        Args:
            num_samples: Total rows to generate across all classes.

        Returns:
            DataFrame with 61 numeric feature columns and a ``label``
            column (str).

        Raises:
            ValueError: If *num_samples* ≤ 0.
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")

        counts = self._class_sample_counts(num_samples)
        frames: List[pd.DataFrame] = []

        for attack_type, n in counts.items():
            if n == 0:
                continue
            arr = self._generate_features(EDGE_IIOTSET_FEATURES, attack_type, n)
            df = pd.DataFrame(arr, columns=EDGE_IIOTSET_FEATURES)

            # Post-process industrial-specific columns for realism
            df["modbus_function_code"] = self._rng.integers(1, 17, size=n)
            df["modbus_unit_id"] = self._rng.integers(1, 248, size=n)
            df["sensor_temperature"] = self._rng.normal(36.5, 2.0, size=n)
            df["sensor_humidity"] = np.clip(
                self._rng.normal(55.0, 15.0, size=n), 0.0, 100.0
            )
            df["scada_command_type"] = self._rng.integers(0, 5, size=n)

            df["label"] = attack_type
            frames.append(df)
            logger.debug(
                "Edge-IIoTset — generated %d samples for '%s'", n, attack_type
            )

        result = pd.concat(frames, ignore_index=True)
        result = result.sample(frac=1.0, random_state=self._seed).reset_index(
            drop=True
        )
        logger.info(
            "Edge-IIoTset dataset generated — %d samples, %d features",
            len(result),
            len(EDGE_IIOTSET_FEATURES),
        )
        return result

    def generate_combined_dataset(
        self, num_samples: int = 100_000
    ) -> pd.DataFrame:
        """Generate a merged dataset with a unified label scheme.

        CICIoMT features are generated for the first half of *num_samples*,
        Edge-IIoTset features for the second half.  The two halves are then
        merged on the **common** CICIoMT columns; Edge-IIoTset-only columns
        are filled with ``0.0`` for the CICIoMT rows.

        Args:
            num_samples: Total combined rows.

        Returns:
            Unified DataFrame with ``label`` column and all 61 feature
            columns (Edge-IIoTset superset).

        Raises:
            ValueError: If *num_samples* ≤ 0.
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")

        n_cic = num_samples // 2
        n_edge = num_samples - n_cic

        df_cic = self.generate_ciciomt_data(n_cic)
        df_edge = self.generate_edge_iiotset_data(n_edge)

        # Align columns — CICIoMT rows get 0.0 for Edge-only features
        combined = pd.concat([df_cic, df_edge], ignore_index=True, sort=False)
        for col in EDGE_IIOTSET_FEATURES:
            if col not in combined.columns:
                combined[col] = 0.0
        combined.fillna(0.0, inplace=True)

        # Reorder: all features first, then label
        ordered_cols = EDGE_IIOTSET_FEATURES + ["label"]
        combined = combined[[c for c in ordered_cols if c in combined.columns]]

        combined = combined.sample(frac=1.0, random_state=self._seed).reset_index(
            drop=True
        )
        logger.info(
            "Combined dataset generated — %d samples, columns: %s",
            len(combined),
            list(combined.columns[:5]),
        )
        return combined


# ===================================================================
# Quick smoke-test
# ===================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    gen = SyntheticIIoMTGenerator()

    df1 = gen.generate_ciciomt_data(1_000)
    print(f"CICIoMT shape: {df1.shape}")
    print(df1["label"].value_counts())

    df2 = gen.generate_edge_iiotset_data(1_000)
    print(f"\nEdge-IIoTset shape: {df2.shape}")
    print(df2["label"].value_counts())

    df3 = gen.generate_combined_dataset(2_000)
    print(f"\nCombined shape: {df3.shape}")
    print(df3["label"].value_counts())
