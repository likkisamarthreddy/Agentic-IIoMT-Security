"""
Windows-compatible network emulator for IIoMT multi-node topology.

Replaces Linux-only Mininet with a pure-Python threading-based emulator
that simulates edge-node and gateway agents communicating over MQTT.
Each virtual node runs in its own thread with simulated resource
constraints (memory, CPU) monitored via psutil.

Typical usage::

    from infrastructure.network_emulator import NetworkEmulator

    emulator = NetworkEmulator(config)
    emulator.add_edge_node("edge-1", edge_agent_1)
    emulator.add_edge_node("edge-2", edge_agent_2)
    emulator.set_gateway(gateway_agent)
    emulator.start()
    # … run evaluation …
    emulator.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import psutil
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "settings.yaml"


def _load_config(path: Path | None = None) -> Dict[str, Any]:
    """Load and return the YAML configuration dictionary.

    Args:
        path: Optional explicit path to settings.yaml.

    Returns:
        Parsed YAML dictionary.
    """
    config_path = path or _DEFAULT_CONFIG
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class NodeMetrics:
    """Runtime metrics for a single virtual node."""

    node_id: str
    memory_mb: float = 0.0
    memory_limit_mb: float = 128.0
    cpu_percent: float = 0.0
    cpu_limit: float = 0.5
    packets_sent: int = 0
    packets_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    uptime_sec: float = 0.0
    warnings: List[str] = field(default_factory=list)


@dataclass
class NetworkStats:
    """Aggregate network-level statistics."""

    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    total_packets: int = 0
    avg_latency_ms: float = 0.0
    peak_latency_ms: float = 0.0
    active_nodes: int = 0


# ---------------------------------------------------------------------------
# VirtualNode
# ---------------------------------------------------------------------------
class VirtualNode:
    """A simulated network node that runs an agent in its own thread.

    Each ``VirtualNode`` wraps a callable *agent* function and executes
    it inside a daemon thread.  While the agent runs, a lightweight
    monitor thread samples the host process for memory / CPU usage and
    emits warnings when the configured limits are approached.

    Args:
        node_id: Unique identifier for this node (e.g. ``"edge-1"``).
        agent_callable: The function to execute (should block until
            stopped).
        memory_limit_mb: Simulated memory ceiling in MB.
        cpu_limit: Simulated CPU core fraction (e.g. ``0.5``).
    """

    def __init__(
        self,
        node_id: str,
        agent_callable: Callable[..., Any],
        memory_limit_mb: float = 128.0,
        cpu_limit: float = 0.5,
    ) -> None:
        self.node_id = node_id
        self._agent_callable = agent_callable
        self.memory_limit_mb = memory_limit_mb
        self.cpu_limit = cpu_limit

        self._thread: Optional[threading.Thread] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._start_time: float = 0.0
        self._metrics = NodeMetrics(
            node_id=node_id,
            memory_limit_mb=memory_limit_mb,
            cpu_limit=cpu_limit,
        )
        self._lock = threading.Lock()

    # -- public API ---------------------------------------------------------

    def start(self) -> None:
        """Launch the agent and its resource-monitor thread."""
        logger.info("Starting virtual node %s", self.node_id)
        self._stop_event.clear()
        self._start_time = time.monotonic()

        self._thread = threading.Thread(
            target=self._run_agent,
            name=f"node-{self.node_id}",
            daemon=True,
        )
        self._monitor_thread = threading.Thread(
            target=self._run_monitor,
            name=f"monitor-{self.node_id}",
            daemon=True,
        )
        self._thread.start()
        self._monitor_thread.start()

    def stop(self) -> None:
        """Signal the node and monitor threads to stop."""
        logger.info("Stopping virtual node %s", self.node_id)
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=3.0)

    @property
    def is_alive(self) -> bool:
        """Return ``True`` if the agent thread is running."""
        return self._thread is not None and self._thread.is_alive()

    def get_metrics(self) -> NodeMetrics:
        """Return a snapshot of current node metrics."""
        with self._lock:
            self._metrics.uptime_sec = time.monotonic() - self._start_time
            return NodeMetrics(**self._metrics.__dict__)

    def record_packet_sent(self, size_bytes: int) -> None:
        """Increment outgoing-packet counters (thread-safe)."""
        with self._lock:
            self._metrics.packets_sent += 1
            self._metrics.bytes_sent += size_bytes

    def record_packet_received(self, size_bytes: int) -> None:
        """Increment incoming-packet counters (thread-safe)."""
        with self._lock:
            self._metrics.packets_received += 1
            self._metrics.bytes_received += size_bytes

    # -- internal -----------------------------------------------------------

    def _run_agent(self) -> None:
        """Execute the wrapped agent callable."""
        try:
            self._agent_callable(self._stop_event)
        except Exception:
            logger.exception("Agent %s crashed", self.node_id)

    def _run_monitor(self) -> None:
        """Periodically sample host-process resource usage."""
        process = psutil.Process()
        while not self._stop_event.is_set():
            try:
                mem_info = process.memory_info()
                mem_mb = mem_info.rss / (1024 * 1024)
                cpu_pct = process.cpu_percent(interval=0.5)

                with self._lock:
                    self._metrics.memory_mb = mem_mb
                    self._metrics.cpu_percent = cpu_pct

                # Warn if approaching limits
                if mem_mb > self.memory_limit_mb * 0.85:
                    warning = (
                        f"Node {self.node_id}: memory {mem_mb:.1f} MB "
                        f"approaching limit {self.memory_limit_mb} MB"
                    )
                    logger.warning(warning)
                    with self._lock:
                        self._metrics.warnings.append(warning)

                if cpu_pct > self.cpu_limit * 100 * 0.90:
                    warning = (
                        f"Node {self.node_id}: CPU {cpu_pct:.1f}% "
                        f"approaching limit {self.cpu_limit * 100:.0f}%"
                    )
                    logger.warning(warning)
                    with self._lock:
                        self._metrics.warnings.append(warning)

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                logger.debug("Monitor lost process handle for %s", self.node_id)
                break

            self._stop_event.wait(timeout=2.0)


# ---------------------------------------------------------------------------
# NetworkEmulator
# ---------------------------------------------------------------------------
class NetworkEmulator:
    """Windows-compatible multi-node IIoMT topology emulator.

    Orchestrates ``VirtualNode`` instances communicating over a real
    (or local) MQTT broker, reproducing the Docker Compose topology
    without requiring Docker or Linux.

    Args:
        config: Optional pre-loaded settings dict.  When *None* the
            emulator reads ``config/settings.yaml`` automatically.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self._config: Dict[str, Any] = config or _load_config()
        self._edge_nodes: Dict[str, VirtualNode] = {}
        self._gateway: Optional[VirtualNode] = None
        self._running = False
        self._lock = threading.Lock()
        self._latency_samples: List[float] = []

        # Pull infrastructure limits from config
        infra = self._config.get("infrastructure", {})
        edge_cfg = infra.get("edge_containers", {})
        self._edge_memory_limit = float(
            str(edge_cfg.get("memory_limit", "128m")).rstrip("mM")
        )
        self._edge_cpu_limit = float(edge_cfg.get("cpu_limit", 0.5))

        gw_cfg = infra.get("gateway", {})
        self._gw_memory_limit = float(
            str(gw_cfg.get("memory_limit", "512m")).rstrip("mM")
        )
        self._gw_cpu_limit = float(gw_cfg.get("cpu_limit", 2.0))

        logger.info(
            "NetworkEmulator initialised — edge limits: %s MB / %s CPU, "
            "gateway limits: %s MB / %s CPU",
            self._edge_memory_limit,
            self._edge_cpu_limit,
            self._gw_memory_limit,
            self._gw_cpu_limit,
        )

    # -- Node registration --------------------------------------------------

    def add_edge_node(
        self,
        node_id: str,
        agent: Callable[..., Any],
    ) -> None:
        """Register an edge-node agent.

        Args:
            node_id: Unique identifier (e.g. ``"edge-1"``).
            agent: Callable that accepts a ``threading.Event`` stop-flag
                and blocks until the flag is set.
        """
        node = VirtualNode(
            node_id=node_id,
            agent_callable=agent,
            memory_limit_mb=self._edge_memory_limit,
            cpu_limit=self._edge_cpu_limit,
        )
        self._edge_nodes[node_id] = node
        logger.info("Registered edge node %s", node_id)

    def set_gateway(self, agent: Callable[..., Any]) -> None:
        """Register the System 2 gateway agent.

        Args:
            agent: Callable that accepts a ``threading.Event`` stop-flag.
        """
        self._gateway = VirtualNode(
            node_id="gateway",
            agent_callable=agent,
            memory_limit_mb=self._gw_memory_limit,
            cpu_limit=self._gw_cpu_limit,
        )
        logger.info("Registered gateway node")

    # -- Lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Start all registered nodes in separate threads.

        Edge nodes are launched first, followed by the gateway so that
        it can immediately begin subscribing to edge-alert topics.
        """
        if self._running:
            logger.warning("Emulator already running")
            return

        logger.info(
            "Starting network emulator with %d edge node(s)",
            len(self._edge_nodes),
        )

        # Start edges
        for node in self._edge_nodes.values():
            node.start()

        # Brief settling period so edges connect to MQTT
        time.sleep(0.5)

        # Start gateway
        if self._gateway is not None:
            self._gateway.start()

        self._running = True
        logger.info("Network emulator running")

    def stop(self) -> None:
        """Gracefully shut down all nodes (gateway first)."""
        if not self._running:
            return

        logger.info("Stopping network emulator …")

        if self._gateway is not None:
            self._gateway.stop()

        for node in self._edge_nodes.values():
            node.stop()

        self._running = False
        logger.info("Network emulator stopped")

    # -- Metrics ------------------------------------------------------------

    def get_node_metrics(self, node_id: str) -> NodeMetrics:
        """Return resource-usage metrics for a specific node.

        Args:
            node_id: Identifier of the node (``"edge-1"``, ``"gateway"``…).

        Returns:
            ``NodeMetrics`` dataclass snapshot.

        Raises:
            KeyError: If *node_id* is not registered.
        """
        if node_id == "gateway" and self._gateway is not None:
            return self._gateway.get_metrics()
        if node_id in self._edge_nodes:
            return self._edge_nodes[node_id].get_metrics()
        raise KeyError(f"Unknown node: {node_id}")

    def get_network_metrics(self) -> NetworkStats:
        """Return aggregate network-level statistics.

        Returns:
            ``NetworkStats`` with bandwidth, latency, and active-node
            counts across the entire emulated topology.
        """
        stats = NetworkStats()
        all_nodes: List[VirtualNode] = list(self._edge_nodes.values())
        if self._gateway is not None:
            all_nodes.append(self._gateway)

        for node in all_nodes:
            m = node.get_metrics()
            stats.total_bytes_sent += m.bytes_sent
            stats.total_bytes_received += m.bytes_received
            stats.total_packets += m.packets_sent + m.packets_received
            if node.is_alive:
                stats.active_nodes += 1

        with self._lock:
            if self._latency_samples:
                stats.avg_latency_ms = sum(self._latency_samples) / len(
                    self._latency_samples
                )
                stats.peak_latency_ms = max(self._latency_samples)

        return stats

    def record_latency(self, latency_ms: float) -> None:
        """Record an observed inter-node latency sample.

        Args:
            latency_ms: Round-trip or one-way latency in milliseconds.
        """
        with self._lock:
            self._latency_samples.append(latency_ms)

    # -- Utility ------------------------------------------------------------

    @property
    def node_ids(self) -> List[str]:
        """Return identifiers of all registered nodes."""
        ids = list(self._edge_nodes.keys())
        if self._gateway is not None:
            ids.append("gateway")
        return ids

    @property
    def is_running(self) -> bool:
        """Return ``True`` if the emulator has been started."""
        return self._running

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"<NetworkEmulator edges={len(self._edge_nodes)} "
            f"gateway={'yes' if self._gateway else 'no'} "
            f"running={self._running}>"
        )


# ---------------------------------------------------------------------------
# Stand-alone smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    def _dummy_agent(stop_event: threading.Event) -> None:
        """Trivial agent that sleeps until stopped."""
        logger.info("Dummy agent started")
        while not stop_event.is_set():
            stop_event.wait(timeout=1.0)
        logger.info("Dummy agent stopped")

    emulator = NetworkEmulator()
    emulator.add_edge_node("edge-1", _dummy_agent)
    emulator.add_edge_node("edge-2", _dummy_agent)
    emulator.add_edge_node("edge-3", _dummy_agent)
    emulator.set_gateway(_dummy_agent)

    emulator.start()
    time.sleep(3)

    for nid in emulator.node_ids:
        m = emulator.get_node_metrics(nid)
        logger.info("  %s → mem=%.1f MB, cpu=%.1f%%", nid, m.memory_mb, m.cpu_percent)

    net = emulator.get_network_metrics()
    logger.info("Network: active=%d, bytes_sent=%d", net.active_nodes, net.total_bytes_sent)

    emulator.stop()
    logger.info("Smoke test complete ✓")
