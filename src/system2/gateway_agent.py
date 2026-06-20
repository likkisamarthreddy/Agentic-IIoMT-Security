import logging
import json
import time
from typing import Dict, Any
import paho.mqtt.client as mqtt

from .reasoning.context_fusion import ContextFusionEngine
from .reasoning.symbolic_rules import SymbolicRuleEngine
from .reasoning.reason_act_loop import ReActLoop
from .mitigation.action_playbook import ActionPlaybook
from .mitigation.sdn_controller import SDNController

logger = logging.getLogger("iimt.system2.gateway")

class GatewayAgent:
    """Main gateway orchestrator for System 2 Reasoning Engine."""
    
    def __init__(self, config: dict, metrics_collector=None):
        self.config = config
        self.metrics_collector = metrics_collector
        self.active_mitigations = 0
        
        from .reasoning.slm_interface import SLMInterface
        
        # Initialize components
        policies_path = "config/safety_policies.yaml" # Assuming run from project root
        self.context_engine = ContextFusionEngine(
            alpha=config.get("system2", {}).get("risk_metric", {}).get("alpha", 0.5),
            beta=config.get("system2", {}).get("risk_metric", {}).get("beta", 0.3),
            gamma=config.get("system2", {}).get("risk_metric", {}).get("gamma", 0.2)
        )
        self.rule_engine = SymbolicRuleEngine(policies_path)
        self.playbook = ActionPlaybook(policies_path)
        self.sdn_controller = SDNController()
        
        # SLM configuration
        slm_config = config.get("system2", {}).get("slm", {})
        if "ollama_host" not in slm_config:
            slm_config["ollama_host"] = f"http://{config.get('system2', {}).get('ollama_hostname', 'ollama')}:11434"
        slm_config["enabled"] = True
        self.slm_interface = SLMInterface(slm_config)
        
        self.react_loop = ReActLoop(
            context_engine=self.context_engine,
            rule_engine=self.rule_engine,
            action_playbook=self.playbook,
            config=config.get("system2", {}).get("reasoning", {}),
            slm_interface=self.slm_interface
        )
        
        # MQTT Setup
        self.mqtt_client = mqtt.Client(client_id="gateway-agent")
        self.mqtt_client.on_message = self._on_mqtt_message
        
        broker_host = config.get("mqtt", {}).get("broker_host", "localhost")
        broker_port = config.get("mqtt", {}).get("broker_port", 1883)
        self.alert_topic = config.get("mqtt", {}).get("topics", {}).get("edge_alerts", "iimt/edge/alerts")
        self.cmd_topic = config.get("mqtt", {}).get("topics", {}).get("gateway_commands", "iimt/gateway/commands")
        self.hitl_topic = config.get("mqtt", {}).get("topics", {}).get("hitl_notifications", "iimt/hitl/notifications")
        self.override_topic = config.get("mqtt", {}).get("topics", {}).get("hitl_overrides", "iimt/gateway/overrides")
        
        try:
            self.mqtt_client.connect(broker_host, broker_port)
            logger.info(f"Gateway Agent connected to MQTT broker at {broker_host}:{broker_port}")
        except Exception as e:
            logger.warning(f"Could not connect to MQTT broker: {e}. Will run in standalone mode.")

    def start(self):
        """Starts the gateway agent processing loop."""
        self.mqtt_client.subscribe(self.alert_topic)
        self.mqtt_client.subscribe(self.override_topic)
        self.mqtt_client.loop_start()
        logger.info("Gateway Agent started and listening for alerts and overrides.")

    def stop(self):
        """Stops the gateway agent."""
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        logger.info("Gateway Agent stopped.")

    def process_alert(self, alert_event: dict):
        """Process an alert using the ReAct loop."""
        logger.info(f"Processing alert from {alert_event.get('source_agent', 'unknown')}: {alert_event.get('alert_id')}")
        
        # Run ReAct Loop
        result = self.react_loop.execute(alert_event)
        
        # Apply Mitigation
        if result.action_taken != "LOG_ONLY":
            action_params = {
                "device_id": alert_event.get("device_id"),
                "action_name": result.action_taken
            }
            rule_id = self.sdn_controller.apply_rule(action_params)
            self._publish_mitigation(alert_event.get("device_id"), action_params)
            self.active_mitigations += 1
            
        # Notify HITL
        self._notify_hitl(alert_event, result.action_taken, result.explanation_nl)
        
        # Record Metrics
        if self.metrics_collector:
            self.metrics_collector.record_agent_latency(result.latency_ms)
            
    def _on_mqtt_message(self, client, userdata, msg):
        """Central MQTT callback dispatcher."""
        topic = msg.topic
        try:
            payload = json.loads(msg.payload.decode())
            if topic == self.alert_topic:
                self.process_alert(payload)
            elif topic == self.override_topic:
                self.handle_hitl_override(payload)
        except Exception as e:
            logger.error(f"Error parsing MQTT message on topic {topic}: {e}")

    def _publish_mitigation(self, device_id: str, action: dict):
        """Publishes mitigation commands back to edge agents."""
        try:
            payload = json.dumps(action)
            self.mqtt_client.publish(self.cmd_topic, payload)
        except Exception as e:
            logger.error(f"Failed to publish mitigation command: {e}")

    def _notify_hitl(self, alert: dict, action: str, explanation: str):
        """Publishes notification for the HITL dashboard."""
        if not self.mqtt_client or not self.mqtt_client.is_connected():
            return
            
        try:
            payload = json.dumps({
                "timestamp": time.time(),
                "device_id": alert.get("device_id"),
                "edge_agent": alert.get("edge_agent"),
                "anomaly_score": alert.get("anomaly_score"),
                "predicted_class": alert.get("predicted_class"),
                "action_taken": action,
                "reasoning": explanation
            })
            self.mqtt_client.publish(self.hitl_topic, payload)
        except Exception as e:
            logger.error(f"Failed to publish HITL notification: {e}")

    def handle_hitl_override(self, override_data: dict):
        """Processes clinician override (approve/reject/modify)."""
        logger.info(f"Handling HITL Override: {override_data}")
        device_id = override_data.get("device_id")
        action = override_data.get("hitl_action")
        
        if action == "REJECT":
            logger.info(f"Clinician rejected AI action for {device_id}. Rolling back...")
            self.sdn_controller.rollback_all_rules(device_id)
            self._publish_mitigation(device_id, {"device_id": device_id, "action_name": "ROLLBACK"})
            self.active_mitigations = max(0, self.active_mitigations - 1)
        elif action == "MODIFY":
            new_action = override_data.get("new_mitigation")
            logger.info(f"Clinician modified AI action for {device_id} to {new_action}. Applying...")
            self.sdn_controller.rollback_all_rules(device_id)
            action_params = {"device_id": device_id, "action_name": new_action}
            self.sdn_controller.apply_rule(action_params)
            self._publish_mitigation(device_id, action_params)
        elif action == "APPROVE":
            logger.info(f"Clinician approved AI action for {device_id}.")

    def get_metrics(self) -> Dict[str, Any]:
        """Returns gateway metrics."""
        return {
            "active_mitigations": self.active_mitigations
        }

if __name__ == "__main__":
    import os
    import yaml
    import time
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)
    logger.info("Initializing Gateway Agent container...")

    # Load config
    config_path = Path("config/settings.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    mqtt_host = os.environ.get("MQTT_HOST", "localhost")
    mqtt_port = int(os.environ.get("MQTT_PORT", "1883"))
    
    if "mqtt" not in config:
        config["mqtt"] = {}
    config["mqtt"]["broker_host"] = mqtt_host
    config["mqtt"]["broker_port"] = mqtt_port

    agent = GatewayAgent(config=config)
    
    try:
        agent.start()
        logger.info("Gateway Agent is running and listening for edge alerts...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        agent.stop()
        logger.info("Gateway Agent stopped by user.")
