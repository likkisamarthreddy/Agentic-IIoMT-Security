import logging
import json
import time
import psutil
from typing import Dict, Any, Optional
import paho.mqtt.client as mqtt

from .detection.kde_threshold import AdaptiveKDEThreshold
from .detection.emergency_brake import EmergencyBrake

logger = logging.getLogger("iimt.system1.edge_agent")

class EdgeAgent:
    """Main edge agent orchestrator for System 1 Reflex Layer."""
    
    def __init__(self, agent_id: str, model: Any, kde_threshold: AdaptiveKDEThreshold, 
                 emergency_brake: EmergencyBrake, mqtt_client: Optional[mqtt.Client] = None,
                 config: dict = None, metrics_collector: Any = None):
        self.agent_id = agent_id
        self.model = model
        self.kde_threshold = kde_threshold
        self.emergency_brake = emergency_brake
        self.mqtt_client = mqtt_client or mqtt.Client(client_id=f"edge-agent-{agent_id}")
        self.config = config or {}
        self.metrics_collector = metrics_collector

        # ------------------------------------------------------------------
        # Detect the model's input contract so we always feed the right shape.
        # ONNX sessions expose input shape as e.g. ['batch_size', seq, features];
        # a dimension may be an int (static) or a string (dynamic).
        # ------------------------------------------------------------------
        self._is_onnx = False
        self.expected_seq_len: Optional[int] = None
        self.expected_num_features: Optional[int] = None
        try:
            import onnxruntime as ort
            if isinstance(self.model, ort.InferenceSession):
                self._is_onnx = True
                in_shape = self.model.get_inputs()[0].shape
                if len(in_shape) == 3:
                    self.expected_seq_len = in_shape[1] if isinstance(in_shape[1], int) else None
                    self.expected_num_features = in_shape[2] if isinstance(in_shape[2], int) else None
        except Exception:
            pass

        # Resolve the buffer/sequence length: a static ONNX seq dim wins,
        # otherwise fall back to config, then a sensible default.
        cfg_seq = self.config.get("system1", {}).get("model", {}).get("sequence_length")
        if self.expected_seq_len and self.expected_seq_len > 0:
            self.sequence_length = int(self.expected_seq_len)
        elif cfg_seq:
            self.sequence_length = int(cfg_seq)
        else:
            self.sequence_length = 50

        import collections
        self.packet_buffer = collections.deque(maxlen=self.sequence_length)

        # KDE warm-up state: the threshold needs a batch of baseline scores
        # before it can decide anomalies. We bootstrap it on-line.
        self.kde_warmup_size = int(
            self.config.get("system1", {}).get("kde", {}).get("warmup_samples", 100)
        )
        self._warmup_scores: list = []
        self._inference_failures = 0

        self.total_packets = 0
        self.anomalies_detected = 0
        self.active = False
        
        # Setup MQTT
        if self.mqtt_client:
            self.mqtt_client.on_message = self._on_message
            
            # Use settings or defaults
            broker_host = self.config.get("mqtt", {}).get("broker_host", "localhost")
            broker_port = self.config.get("mqtt", {}).get("broker_port", 1883)
            self.traffic_topic = f"iimt/traffic/{agent_id}"
            self.alert_topic = self.config.get("mqtt", {}).get("topics", {}).get("edge_alerts", "iimt/edge/alerts")
            self.cmd_topic = self.config.get("mqtt", {}).get("topics", {}).get("gateway_commands", "iimt/gateway/commands")
            
            try:
                self.mqtt_client.connect(broker_host, broker_port)
                logger.info(f"Edge Agent {agent_id} connected to MQTT broker at {broker_host}:{broker_port}")
            except Exception as e:
                logger.warning(f"Edge Agent {agent_id} could not connect to MQTT broker: {e}. Running in standalone mode.")

    def start(self):
        """Starts the edge agent processing loop."""
        self.active = True
        if self.mqtt_client and self.mqtt_client.is_connected():
            self.mqtt_client.subscribe(self.traffic_topic)
            self.mqtt_client.subscribe(self.cmd_topic)
            self.mqtt_client.loop_start()
            logger.info(f"Edge Agent {self.agent_id} started listening on {self.traffic_topic}")
            
    def stop(self):
        """Stops the edge agent."""
        self.active = False
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        logger.info(f"Edge Agent {self.agent_id} stopped.")

    def process_packet(self, packet_data: dict) -> Optional[Dict[str, Any]]:
        """Buffers packets, runs sequence inference, checks KDE, triggers brake if needed."""
        self.total_packets += 1
        
        # Assume packet_data["features"] is already a preprocessed array of length `num_features`
        features = packet_data.get("features")
        
        if features is not None:
            self.packet_buffer.append(features)
            
        # Only run inference if we have a full sequence
        if len(self.packet_buffer) < self.sequence_length:
            return None
            
        # 1. Measure inference latency (τ_edge)
        import numpy as np
        start_ns = time.perf_counter_ns()

        try:
            # Raw buffer as (buffer_len, num_features)
            seq_array = np.array(self.packet_buffer, dtype=np.float32)

            if self._is_onnx:
                # Validate the feature count against the model contract.
                if (
                    self.expected_num_features
                    and seq_array.shape[1] != self.expected_num_features
                ):
                    raise ValueError(
                        f"Feature mismatch: input has {seq_array.shape[1]} features "
                        f"but model expects {self.expected_num_features}"
                    )

                # Match the model's expected sequence length. If the ONNX seq
                # dim is static (e.g. 1), slice/pad the buffer to fit it;
                # otherwise use the whole buffer.
                target_seq = self.expected_seq_len or seq_array.shape[0]
                seq_slice = seq_array[-target_seq:]
                if seq_slice.shape[0] < target_seq:
                    pad = np.repeat(
                        seq_slice[:1], target_seq - seq_slice.shape[0], axis=0
                    )
                    seq_slice = np.concatenate([pad, seq_slice], axis=0)

                inp = np.expand_dims(seq_slice, axis=0)  # (1, seq, features)
                ort_inputs = {self.model.get_inputs()[0].name: inp}
                logits = self.model.run(None, ort_inputs)[0]

                import scipy.special
                probs = scipy.special.softmax(logits, axis=1)
                score = float(1.0 - np.max(probs, axis=1)[0])
            else:
                # PyTorch path: feed (1, seq, features)
                import torch
                seq_tensor = torch.tensor(np.expand_dims(seq_array, axis=0))
                score = float(self.model.get_anomaly_score(seq_tensor).item())
        except Exception as e:
            # Never fabricate a detection score on failure — surface it instead.
            self._inference_failures += 1
            if self._inference_failures <= 5 or self._inference_failures % 100 == 0:
                logger.error(
                    "Inference failed (#%d): %s", self._inference_failures, e
                )
            return None

        latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        
        # 2. Check KDE dynamic threshold (with on-line warm-up / bootstrap)
        if not getattr(self.kde_threshold, "_is_initialized", False):
            self._warmup_scores.append(float(score))
            if len(self._warmup_scores) >= self.kde_warmup_size:
                try:
                    self.kde_threshold.initialize(
                        np.asarray(self._warmup_scores, dtype=np.float64)
                    )
                    logger.info(
                        "KDE threshold initialised from %d warm-up scores.",
                        len(self._warmup_scores),
                    )
                except Exception as e:
                    logger.warning("KDE warm-up initialisation failed: %s", e)
                    # Keep collecting; retry once enough fresh scores accrue.
                    self._warmup_scores = self._warmup_scores[-self.kde_warmup_size // 2 :]
            # During warm-up we cannot reliably decide anomalies.
            is_anomalous = False
        else:
            is_anomalous = self.kde_threshold.is_anomalous(score)
            # Feed verified-normal scores back for on-line threshold adaptation.
            if not is_anomalous:
                try:
                    self.kde_threshold.update(score, is_normal=True)
                except Exception:
                    pass

        result = {
            "packet_id": packet_data.get("packet_id", self.total_packets),
            "device_id": packet_data.get("device_id", "unknown"),
            "score": float(score),
            "is_anomalous": is_anomalous,
            "latency_ms": latency_ms,
            "brake_triggered": False
        }
        
        if is_anomalous:
            self.anomalies_detected += 1
            
            # 3. Emergency Brake (SDN micro-mitigation)
            brake_action = self.emergency_brake.evaluate([score])
            if brake_action:
                result["brake_triggered"] = True
                result["brake_action"] = brake_action.action
                logger.warning(f"[EMERGENCY BRAKE] {brake_action.action} triggered for device {result['device_id']}")
                self._apply_sdn_mitigation(brake_action.action)
                
            # 4. Forward Alert to Gateway (System 2)
            self._forward_alert(result)
            
        return result

    def _apply_sdn_mitigation(self, action: str):
        """Applies physical SDN mitigation using Linux Traffic Control (tc).

        ``tc`` only exists on Linux, so on other platforms this logs the
        intended action instead of issuing a no-op command that would error.
        Success is only reported when the command actually returns 0.
        """
        import platform
        import shutil
        import subprocess

        if platform.system() != "Linux" or shutil.which("tc") is None:
            logger.info(
                "[SDN-SIM] '%s' requested (tc unavailable on this platform; "
                "simulated only).", action
            )
            return

        try:
            if "THROTTLE" in action:
                cmd = ["tc", "qdisc", "replace", "dev", "eth0", "root", "tbf",
                       "rate", "1mbit", "burst", "32kbit", "latency", "400ms"]
            elif "DROP" in action:
                cmd = ["tc", "qdisc", "replace", "dev", "eth0", "root",
                       "netem", "loss", "100%"]
            else:
                return

            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if proc.returncode == 0:
                logger.info("Applied physical tc rule on eth0: %s", action)
            else:
                logger.error(
                    "tc rule for '%s' failed (rc=%d): %s",
                    action, proc.returncode, proc.stderr.strip(),
                )
        except Exception as e:
            logger.error(f"Failed to apply SDN mitigation: {e}")

    def _on_message(self, client, userdata, msg):
        """MQTT callback for incoming traffic or gateway commands."""
        try:
            if msg.topic == self.traffic_topic:
                data = json.loads(msg.payload.decode())
                self.process_packet(data)
            elif msg.topic == self.cmd_topic:
                cmd = json.loads(msg.payload.decode())
                # Only process commands for our managed devices
                logger.info(f"Received gateway command: {cmd}")
        except Exception as e:
            logger.error(f"Error parsing MQTT message: {e}")

    def _forward_alert(self, alert_data: dict):
        """Publishes alert to System 2 gateway topic."""
        if not self.mqtt_client or not self.mqtt_client.is_connected():
            return
            
        try:
            payload = json.dumps({
                "source_agent": self.agent_id,
                "timestamp": time.time(),
                **alert_data
            })
            self.mqtt_client.publish(self.alert_topic, payload)
        except Exception as e:
            logger.error(f"Failed to forward alert: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Returns metrics dictionary including τ_edge, memory, CPU."""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        cpu_percent = process.cpu_percent(interval=0.1)
        
        return {
            "total_packets": self.total_packets,
            "anomalies_detected": self.anomalies_detected,
            "memory_mb": memory_mb,
            "cpu_percent": cpu_percent
        }

if __name__ == "__main__":
    import os
    import torch
    import yaml
    from pathlib import Path
    from system1.models.cnn_bigru import CNNBiGRU

    logging.basicConfig(level=logging.INFO)
    logger.info("Initializing Edge Agent container...")

    # Load config
    config_path = Path("config/settings.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    agent_id = os.environ.get("AGENT_ID", "edge-1")
    mqtt_host = os.environ.get("MQTT_HOST", "localhost")
    mqtt_port = int(os.environ.get("MQTT_PORT", "1883"))
    
    # Update config with env vars
    if "mqtt" not in config:
        config["mqtt"] = {}
    config["mqtt"]["broker_host"] = mqtt_host
    config["mqtt"]["broker_port"] = mqtt_port

    # Load Model (ONNX INT8 preferred for edge)
    import onnxruntime as ort
    model_path_onnx = Path("checkpoints/cnn_bigru_int8.onnx")
    
    if model_path_onnx.exists():
        sess_options = ort.SessionOptions()
        model = ort.InferenceSession(str(model_path_onnx), sess_options, providers=["CPUExecutionProvider"])
        logger.info(f"Loaded ONNX INT8 model from {model_path_onnx}")
    else:
        logger.warning(f"ONNX model not found at {model_path_onnx}. Falling back to PyTorch FP32 if available.")
        model = CNNBiGRU(
            num_features=config["data"]["num_features"],
            num_classes=6,  # 6 macro classes
            config_path=config_path
        )
        fp32_path = Path("checkpoints/cnn_bigru_fp32.pt")
        if fp32_path.exists():
            model.load_state_dict(torch.load(fp32_path, weights_only=True))
            logger.info(f"Loaded PyTorch FP32 model from {fp32_path}")
        model.eval()

    # Load KDE and Brake
    kde = AdaptiveKDEThreshold(config_path=config_path)
    brake = EmergencyBrake(kde_threshold=kde, config_path=config_path)

    # Start Agent
    agent = EdgeAgent(
        agent_id=agent_id,
        model=model,
        kde_threshold=kde,
        emergency_brake=brake,
        config=config
    )
    
    try:
        agent.start()
        logger.info(f"Edge Agent {agent_id} is running and waiting for traffic...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        agent.stop()
        logger.info("Agent stopped by user.")
