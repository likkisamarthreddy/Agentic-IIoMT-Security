"""
Comprehensive metrics collection for IIoMT agentic security evaluation.

Collects, stores, and analyses all performance metrics required to
reproduce the paper's benchmark tables:

- **Latency**: τ_edge, τ_agent, T_ttm (time-to-mitigation)
- **Resources**: per-agent RAM, CPU, MQTT bandwidth
- **Detection**: true/predicted labels with confidence for per-attack
  accuracy, precision, recall, F1, and false-positive rate (FPR)

All samples are timestamped for time-series analysis and CSV export.

Typical usage::

    collector = MetricsCollector()
    collector.record_edge_latency("edge-1", 2.4)
    collector.record_detection(true_label=1, predicted_label=1, confidence=0.97)
    summary = collector.get_summary()
    collector.export_csv(Path("results/metrics.csv"))
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paper target thresholds (Section V of the research paper)
# ---------------------------------------------------------------------------
_PAPER_TARGETS: Dict[str, Dict[str, float]] = {
    "tau_edge_ms": {"target": 3.0, "op": "le"},
    "tau_agent_ms": {"target": 180.0, "op": "le"},
    "t_ttm_ms": {"target": 250.0, "op": "le"},
    "peak_ram_mb": {"target": 45.0, "op": "le"},
    "model_size_mb": {"target": 15.0, "op": "le"},
    "overall_accuracy": {"target": 0.95, "op": "ge"},
}


# ---------------------------------------------------------------------------
# Timestamped sample containers
# ---------------------------------------------------------------------------
@dataclass
class _TimedSample:
    """A single metric sample with a monotonic timestamp."""

    timestamp: float
    value: float
    label: str = ""


@dataclass
class _DetectionSample:
    """A single detection result for confusion-matrix construction."""

    timestamp: float
    true_label: int
    predicted_label: int
    confidence: float


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------
class MetricsCollector:
    """Centralised metrics store for the IIoMT evaluation pipeline.

    Thread-safe: all ``record_*`` methods acquire an internal lock so
    the collector can be shared across edge-node and gateway threads.
    """

    def __init__(self) -> None:
        self._edge_latencies: Dict[str, List[_TimedSample]] = {}
        self._agent_latencies: List[_TimedSample] = []
        self._ttm_samples: List[_TimedSample] = []
        self._memory_samples: Dict[str, List[_TimedSample]] = {}
        self._cpu_samples: Dict[str, List[_TimedSample]] = {}
        self._bandwidth_samples: List[_TimedSample] = []
        self._detections: List[_DetectionSample] = []

        import threading
        self._lock = threading.Lock()

        logger.info("MetricsCollector initialised")

    # -- Recording ----------------------------------------------------------

    def record_edge_latency(self, agent_id: str, latency_ms: float) -> None:
        """Record an edge-agent inference latency (τ_edge).

        Args:
            agent_id: Identifier of the edge node (e.g. ``"edge-1"``).
            latency_ms: Measured latency in milliseconds.
        """
        sample = _TimedSample(timestamp=time.monotonic(), value=latency_ms)
        with self._lock:
            self._edge_latencies.setdefault(agent_id, []).append(sample)

    def record_agent_latency(self, latency_ms: float) -> None:
        """Record a gateway reasoning-loop latency (τ_agent).

        Args:
            latency_ms: End-to-end reasoning time in milliseconds.
        """
        sample = _TimedSample(timestamp=time.monotonic(), value=latency_ms)
        with self._lock:
            self._agent_latencies.append(sample)

    def record_ttm(self, ttm_ms: float) -> None:
        """Record a total time-to-mitigation measurement (T_ttm).

        Args:
            ttm_ms: Time from threat detection to mitigation action in ms.
        """
        sample = _TimedSample(timestamp=time.monotonic(), value=ttm_ms)
        with self._lock:
            self._ttm_samples.append(sample)

    def record_memory(self, agent_id: str, memory_mb: float) -> None:
        """Record a RAM-usage sample for a specific agent.

        Args:
            agent_id: Agent identifier.
            memory_mb: Resident memory in megabytes.
        """
        sample = _TimedSample(timestamp=time.monotonic(), value=memory_mb)
        with self._lock:
            self._memory_samples.setdefault(agent_id, []).append(sample)

    def record_cpu(self, agent_id: str, cpu_percent: float) -> None:
        """Record a CPU-usage sample for a specific agent.

        Args:
            agent_id: Agent identifier.
            cpu_percent: CPU utilisation as a percentage (0–100+).
        """
        sample = _TimedSample(timestamp=time.monotonic(), value=cpu_percent)
        with self._lock:
            self._cpu_samples.setdefault(agent_id, []).append(sample)

    def record_bandwidth(self, bytes_sent: int) -> None:
        """Record an MQTT bandwidth sample.

        Args:
            bytes_sent: Number of bytes transmitted in this sample.
        """
        sample = _TimedSample(
            timestamp=time.monotonic(), value=float(bytes_sent)
        )
        with self._lock:
            self._bandwidth_samples.append(sample)

    def record_detection(
        self,
        true_label: int,
        predicted_label: int,
        confidence: float,
    ) -> None:
        """Record a single classification result.

        Args:
            true_label: Ground-truth class index.
            predicted_label: Model-predicted class index.
            confidence: Classifier softmax confidence (0–1).
        """
        sample = _DetectionSample(
            timestamp=time.monotonic(),
            true_label=true_label,
            predicted_label=predicted_label,
            confidence=confidence,
        )
        with self._lock:
            self._detections.append(sample)

    # -- Summaries ----------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """Return a comprehensive metrics summary dictionary.

        Returns:
            Dictionary containing mean/p50/p95/p99/max for each latency
            family, peak/mean resource usage, total bandwidth, and overall
            detection accuracy.
        """
        with self._lock:
            summary: Dict[str, Any] = {}

            # — Edge latencies (aggregated across all agents) —
            all_edge = [
                s.value for samples in self._edge_latencies.values()
                for s in samples
            ]
            summary["edge_latency"] = self._describe(all_edge, "τ_edge (ms)")

            # — Per-agent edge latencies —
            per_agent: Dict[str, Dict[str, float]] = {}
            for aid, samples in self._edge_latencies.items():
                vals = [s.value for s in samples]
                per_agent[aid] = self._describe(vals, f"τ_edge {aid}")
            summary["edge_latency_per_agent"] = per_agent

            # — Agent latency —
            agent_vals = [s.value for s in self._agent_latencies]
            summary["agent_latency"] = self._describe(agent_vals, "τ_agent (ms)")

            # — TTM —
            ttm_vals = [s.value for s in self._ttm_samples]
            summary["ttm"] = self._describe(ttm_vals, "T_ttm (ms)")

            # — Memory —
            mem_summary: Dict[str, Dict[str, float]] = {}
            for aid, samples in self._memory_samples.items():
                vals = [s.value for s in samples]
                mem_summary[aid] = self._describe(vals, f"RAM {aid} (MB)")
            summary["memory"] = mem_summary

            # — CPU —
            cpu_summary: Dict[str, Dict[str, float]] = {}
            for aid, samples in self._cpu_samples.items():
                vals = [s.value for s in samples]
                cpu_summary[aid] = self._describe(vals, f"CPU {aid} (%)")
            summary["cpu"] = cpu_summary

            # — Bandwidth —
            bw_vals = [s.value for s in self._bandwidth_samples]
            summary["bandwidth"] = {
                "total_bytes": sum(bw_vals),
                "sample_count": len(bw_vals),
                "mean_bytes_per_sample": (
                    float(np.mean(bw_vals)) if bw_vals else 0.0
                ),
            }

            # — Detection accuracy —
            if self._detections:
                true = np.array([d.true_label for d in self._detections])
                pred = np.array([d.predicted_label for d in self._detections])
                summary["detection"] = {
                    "total_samples": len(self._detections),
                    "overall_accuracy": float(np.mean(true == pred)),
                    "mean_confidence": float(
                        np.mean([d.confidence for d in self._detections])
                    ),
                }
            else:
                summary["detection"] = {
                    "total_samples": 0,
                    "overall_accuracy": 0.0,
                    "mean_confidence": 0.0,
                }

            return summary

    def get_per_attack_metrics(
        self,
        label_mapping: Dict[int, str],
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-attack-type metrics (Table 1 in the paper).

        Args:
            label_mapping: Map from integer class index to human-readable
                attack name, e.g. ``{0: "Benign", 1: "DDoS", …}``.

        Returns:
            Nested dict ``{attack_name: {accuracy, precision, recall,
            f1, fpr, support}}`` where *fpr* is the false-positive rate
            for that specific class.
        """
        with self._lock:
            if not self._detections:
                logger.warning("No detection samples recorded")
                return {}

            true = np.array([d.true_label for d in self._detections])
            pred = np.array([d.predicted_label for d in self._detections])

        classes = sorted(label_mapping.keys())
        results: Dict[str, Dict[str, float]] = {}

        for cls in classes:
            name = label_mapping[cls]
            tp = int(np.sum((true == cls) & (pred == cls)))
            fp = int(np.sum((true != cls) & (pred == cls)))
            fn = int(np.sum((true == cls) & (pred != cls)))
            tn = int(np.sum((true != cls) & (pred != cls)))
            support = int(np.sum(true == cls))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            accuracy = tp / support if support > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            results[name] = {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "fpr": round(fpr, 4),
                "support": support,
            }

        return results

    # -- Export -------------------------------------------------------------

    def export_csv(self, filepath: Path) -> None:
        """Export all raw metric samples to a CSV file.

        Creates separate sections for each metric family, delineated by
        a header row.  Suitable for downstream analysis in pandas or
        spreadsheet tools.

        Args:
            filepath: Destination CSV path (parent dirs created if needed).
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with self._lock, open(filepath, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)

            # — Edge latencies —
            writer.writerow(["## Edge Latencies"])
            writer.writerow(["timestamp", "agent_id", "latency_ms"])
            for aid, samples in self._edge_latencies.items():
                for s in samples:
                    writer.writerow([s.timestamp, aid, s.value])
            writer.writerow([])

            # — Agent latencies —
            writer.writerow(["## Agent Latencies"])
            writer.writerow(["timestamp", "latency_ms"])
            for s in self._agent_latencies:
                writer.writerow([s.timestamp, s.value])
            writer.writerow([])

            # — TTM —
            writer.writerow(["## Time-to-Mitigation"])
            writer.writerow(["timestamp", "ttm_ms"])
            for s in self._ttm_samples:
                writer.writerow([s.timestamp, s.value])
            writer.writerow([])

            # — Memory —
            writer.writerow(["## Memory Usage"])
            writer.writerow(["timestamp", "agent_id", "memory_mb"])
            for aid, samples in self._memory_samples.items():
                for s in samples:
                    writer.writerow([s.timestamp, aid, s.value])
            writer.writerow([])

            # — CPU —
            writer.writerow(["## CPU Usage"])
            writer.writerow(["timestamp", "agent_id", "cpu_percent"])
            for aid, samples in self._cpu_samples.items():
                for s in samples:
                    writer.writerow([s.timestamp, aid, s.value])
            writer.writerow([])

            # — Bandwidth —
            writer.writerow(["## Bandwidth"])
            writer.writerow(["timestamp", "bytes_sent"])
            for s in self._bandwidth_samples:
                writer.writerow([s.timestamp, s.value])
            writer.writerow([])

            # — Detections —
            writer.writerow(["## Detections"])
            writer.writerow(
                ["timestamp", "true_label", "predicted_label", "confidence"]
            )
            for d in self._detections:
                writer.writerow(
                    [d.timestamp, d.true_label, d.predicted_label, d.confidence]
                )

        logger.info("Metrics exported to %s", filepath)

    # -- Target comparison --------------------------------------------------

    def check_targets(self) -> Dict[str, Dict[str, Any]]:
        """Compare collected metrics against paper target thresholds.

        Returns:
            Dict mapping metric name → ``{measured, target, op, passed}``.
        """
        summary = self.get_summary()
        results: Dict[str, Dict[str, Any]] = {}

        # Map summary keys to paper targets
        metric_map: Dict[str, Optional[float]] = {
            "tau_edge_ms": summary["edge_latency"].get("mean"),
            "tau_agent_ms": summary["agent_latency"].get("mean"),
            "t_ttm_ms": summary["ttm"].get("mean"),
            "overall_accuracy": summary["detection"].get("overall_accuracy"),
        }

        # Add peak RAM from memory samples
        all_mem = [
            s.value
            for samples in self._memory_samples.values()
            for s in samples
        ]
        if all_mem:
            metric_map["peak_ram_mb"] = float(np.max(all_mem))

        for metric_name, target_info in _PAPER_TARGETS.items():
            measured = metric_map.get(metric_name)
            if measured is None:
                results[metric_name] = {
                    "measured": None,
                    "target": target_info["target"],
                    "op": target_info["op"],
                    "passed": None,
                    "status": "NO_DATA",
                }
                continue

            target_val = target_info["target"]
            op = target_info["op"]
            if op == "le":
                passed = measured <= target_val
            elif op == "ge":
                passed = measured >= target_val
            else:
                passed = False

            results[metric_name] = {
                "measured": round(measured, 4),
                "target": target_val,
                "op": op,
                "passed": passed,
                "status": "PASS" if passed else "FAIL",
            }

        return results

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _describe(
        values: List[float], label: str = ""
    ) -> Dict[str, float]:
        """Compute descriptive statistics for a list of values.

        Args:
            values: Numeric sample list.
            label: Human-readable label (logged on empty input).

        Returns:
            Dict with keys ``count, mean, std, min, p50, p95, p99, max``.
        """
        if not values:
            logger.debug("No samples for %s", label)
            return {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "max": 0.0,
            }
        arr = np.array(values)
        return {
            "count": len(arr),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(np.max(arr)),
        }

    def __repr__(self) -> str:  # noqa: D105
        with self._lock:
            n_edge = sum(len(v) for v in self._edge_latencies.values())
            n_det = len(self._detections)
        return (
            f"<MetricsCollector edge_samples={n_edge} "
            f"detections={n_det}>"
        )


# ---------------------------------------------------------------------------
# Governance Metrics (AAIF Paper)
# ---------------------------------------------------------------------------

def compute_ecr(policy_ok_count: int, total_constrained: int) -> float:
    """Ethical Compliance Rate (ECR)"""
    return policy_ok_count / total_constrained if total_constrained > 0 else 1.0

def compute_fer(false_escalation_count: int, total_escalations: int) -> float:
    """False Escalation Rate (FER)"""
    return false_escalation_count / total_escalations if total_escalations > 0 else 0.0

def compute_gci(compliance_ratios: List[float], weights: List[float]) -> float:
    """Governance Compliance Index (GCI)"""
    if not compliance_ratios or not weights or sum(weights) == 0:
        return 0.0
    return sum(c * w for c, w in zip(compliance_ratios, weights)) / sum(weights)

def compute_ri2(ecr_segments: List[float]) -> float:
    """Resilience Index (RI2)"""
    if not ecr_segments:
        return 1.0
    return max(0.0, 1.0 - float(np.var(ecr_segments)))

def compute_cas(current_ecr: float, fer: float, previous_ecr: float = None, delta_t: float = 1.0) -> float:
    """Cyber-Adaptive Score (CAS)"""
    if previous_ecr is None or delta_t <= 0:
        return current_ecr * (1.0 - fer)
    rate_of_change = (current_ecr - previous_ecr) / delta_t
    return current_ecr * (1.0 - fer) + rate_of_change

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    mc = MetricsCollector()

    # Simulate some samples
    import random
    random.seed(42)

    for i in range(100):
        mc.record_edge_latency("edge-1", random.gauss(2.5, 0.4))
        mc.record_edge_latency("edge-2", random.gauss(2.7, 0.5))
        mc.record_agent_latency(random.gauss(150, 20))
        mc.record_ttm(random.gauss(200, 30))
        mc.record_memory("edge-1", random.gauss(38, 5))
        mc.record_cpu("edge-1", random.gauss(25, 8))
        mc.record_bandwidth(random.randint(100, 5000))

        true_lbl = random.choice([0, 1, 2, 3, 4, 5])
        pred_lbl = true_lbl if random.random() < 0.96 else random.randint(0, 5)
        mc.record_detection(true_lbl, pred_lbl, random.uniform(0.8, 1.0))

    summary = mc.get_summary()
    logger.info("Summary: %s", summary)

    label_map = {
        0: "Benign", 1: "DDoS", 2: "DoS",
        3: "Reconnaissance", 4: "Spoofing", 5: "MITM",
    }
    per_attack = mc.get_per_attack_metrics(label_map)
    for name, metrics in per_attack.items():
        logger.info("  %s: %s", name, metrics)

    targets = mc.check_targets()
    for name, result in targets.items():
        logger.info("  %s: %s", name, result)

    mc.export_csv(Path("results/metrics_smoke_test.csv"))
    logger.info("Smoke test complete ✓")
