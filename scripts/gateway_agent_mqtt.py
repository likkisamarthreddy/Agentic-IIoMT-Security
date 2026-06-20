import time
import json
import logging
# Mock import for paho mqtt to avoid dependency issues on standard environments
try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("Warning: paho-mqtt not installed. Using mock MQTT client.")
    class mqtt:
        class Client:
            def connect(self, *args, **kwargs): pass
            def loop_start(self): pass
            def publish(self, topic, payload):
                print(f"[MQTT SEND] {topic}: {payload}")

from transformers import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("GatewayAgent")

class GatewayAgent:
    def __init__(self, model_path="./slm_gateway_agent_lora"):
        logger.info(f"Loading SLM Gateway Reasoner from {model_path}...")
        try:
            self.slm = pipeline("text-generation", model=model_path, device_map="auto")
        except Exception as e:
            logger.warning(f"Failed to load fine-tuned SLM. Using mock pipeline. Error: {e}")
            self.slm = lambda prompt, **kwargs: [{"generated_text": prompt + " [Action] throttle. [Reasoning] Override applied."}]
            
        self.safety_rules = {
            "infusion_pump": ["throttle", "alert_doctor"], # CANNOT fully isolate
            "heart_monitor": ["read_only", "alert_nurse"],
            "hvac_controller": ["isolate", "reboot", "ignore"],
            "smart_lighting": ["isolate", "ignore"]
        }
        
        # MQTT Setup
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.connect("localhost", 1883)
        self.mqtt_client.loop_start()

    def send_mqtt_command(self, device_id, action):
        topic = f"iot/command/{device_id}"
        payload = json.dumps({"action": action, "timestamp": time.time()})
        self.mqtt_client.publish(topic, payload)
        logger.info(f"Published mitigation: {action} to {topic}")

    def on_alert_received(self, device_id, alert_type, alert_confidence, criticality, historical_density, context_vitals):
        logger.info("="*50)
        logger.info(f"ALERT RECEIVED: {alert_type} on {device_id} (Confidence: {alert_confidence:.2f})")
        
        # Step 1: Context Fusion (Equation 1)
        risk = (0.6 * alert_confidence) + (0.3 * criticality) + (0.1 * historical_density)
        logger.info(f"Calculated Risk Score: {risk:.2f}")
        
        # If confidence is low, trigger HITL
        if alert_confidence < 0.70:
            logger.warning(f"Confidence {alert_confidence*100:.1f}% < 70%. Triggering Human-In-The-Loop (HITL) interface.")
            self.send_mqtt_command(device_id, "alert_admin")
            return
            
        # Step 2: Generate Plan using SLM
        criticality_str = "High" if criticality > 0.7 else "Low"
        prompt = (f"[System] You are a clinical safety agent. Alert: {alert_type} detected on {device_id}. "
                  f"Criticality: {criticality_str}. Patient Vitals: {context_vitals}. "
                  f"Available Actions: [throttle, isolate, reconfigure, ignore].")
                  
        logger.info("Prompting SLM Reasoner...")
        t0 = time.time()
        
        response = self.slm(prompt, max_new_tokens=50)[0]['generated_text']
        
        logger.info(f"SLM Reasoning Time: {(time.time()-t0)*1000:.2f}ms")
        logger.info(f"SLM Response: {response}")
        
        # Step 3: Symbolic Validation (The "Neuro-Symbolic" safety net)
        action_to_take = "ignore"
        if "[Action]" in response:
            try:
                action_to_take = response.split("[Action]")[1].split(".")[0].strip()
            except IndexError:
                pass
                
        # Enforce rule: never isolate life-critical devices
        if action_to_take == "isolate" and device_id in ["infusion_pump", "heart_monitor"]:
            logger.error(f"SYMBOLIC OVERRIDE TRIGGERED! SLM attempted unsafe action: {action_to_take} on {device_id}")
            action_to_take = "throttle" # Safe fallback
        
        self.send_mqtt_command(device_id, action_to_take)
        logger.info("="*50)

if __name__ == "__main__":
    agent = GatewayAgent()
    
    # Mock some incoming alerts from System 1
    time.sleep(1)
    agent.on_alert_received(
        device_id="infusion_pump",
        alert_type="DDoS",
        alert_confidence=0.95,
        criticality=0.9,
        historical_density=0.8,
        context_vitals="Stable"
    )
    
    time.sleep(1)
    agent.on_alert_received(
        device_id="hvac_controller",
        alert_type="Spoofing",
        alert_confidence=0.98,
        criticality=0.2,
        historical_density=0.1,
        context_vitals="N/A"
    )
    
    time.sleep(1)
    agent.on_alert_received(
        device_id="heart_monitor",
        alert_type="Man-in-the-Middle",
        alert_confidence=0.65, # Should trigger HITL
        criticality=0.9,
        historical_density=0.2,
        context_vitals="Fluctuating"
    )
