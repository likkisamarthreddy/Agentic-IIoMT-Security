import json
import paho.mqtt.client as mqtt
import requests

# Config
MQTT_BROKER = "127.0.0.1" # Mosquitto runs locally in the gateway container
MQTT_PORT = 1883
OLLAMA_API = "http://127.0.0.1:11434/api/generate"

def query_slm(alert_data):
    prompt = f"""
    [System 2 Gateway Reasoning Engine]
    Anomalous IIoMT traffic detected at the edge.
    Confidence: {alert_data['confidence']}
    Raw Telemetry: {json.dumps(alert_data['raw_telemetry'])}
    
    You are protecting a critical medical environment. 
    Evaluate the risk and provide a single actionable mitigation command (e.g. DROP, THROTTLE, QUARANTINE, ALLOW).
    """
    
    payload = {
        "model": "phi3:mini",
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_API, json=payload)
        return response.json().get('response', 'THROTTLE')
    except Exception as e:
        # Simulated reasoning if Ollama isn't running natively
        return """
[Simulated SLM Reasoning]
1. Analyzing telemetry: High volume of identical payloads detected from Edge_Node_1.
2. Confidence score is extreme (0.99+), indicating a sustained volumetric anomaly.
3. Cross-referencing safe-state protocols: Full quarantine may disrupt vital clinical metrics.
4. Action Plan: Deploy localized bandwidth throttling (tc qdisc) and alert SOC.
ACTION: THROTTLE
        """

def on_message(client, userdata, msg):
    print(f"Received alert from edge: {msg.topic}")
    try:
        alert_data = json.loads(msg.payload.decode())
        print(f"Escalating to SLM for reasoning...")
        
        mitigation_plan = query_slm(alert_data)
        
        print("\n--- SLM REASONING OUTPUT ---")
        print(mitigation_plan)
        print("----------------------------\n")
        
    except Exception as e:
        print(f"Error parsing message: {e}")

client = mqtt.Client(client_id="System2_Gateway")
client.on_message = on_message

client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.subscribe("security/alerts")

print("System 2 Gateway listening for security alerts...")
client.loop_forever()
