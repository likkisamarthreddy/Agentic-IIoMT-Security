import os
import json
import socket
import paho.mqtt.client as mqtt
import onnxruntime as ort
import numpy as np

# Config
UDP_IP = "0.0.0.0"
UDP_PORT = 1883
GATEWAY_IP = "127.0.0.1"
MQTT_PORT = 1883
MODEL_PATH = "expert_bigru_int8.onnx"

# Load INT8 Model (Will run at 0.64ms on CPU per your benchmarks)
if os.path.exists(MODEL_PATH):
    session = ort.InferenceSession(MODEL_PATH)
else:
    print(f"Waiting for {MODEL_PATH} from Kaggle Phase 1...")

# MQTT Client for System 2 Gateway communication
mqtt_client = mqtt.Client(client_id="Edge_Node_1")
mqtt_client.connect(GATEWAY_IP, MQTT_PORT, 60)

# UDP Socket for receiving telemetry
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"System 1 Edge Agent listening on UDP {UDP_PORT}...")

# Rolling sequence buffer (Simplified for 1 sequence length)
sequence_buffer = []

while True:
    data, addr = sock.recvfrom(4096)
    try:
        telemetry = json.loads(data.decode('utf-8'))
        
        # In a real environment, extract the 46 features here.
        # For emulation, we simulate the processed tensor:
        dummy_tensor = np.zeros((1, 32, 281), dtype=np.float32)
        
        if 'session' in locals():
            outputs = session.run(None, {'input': dummy_tensor})
            # Simple threshold check
            s1_prob = np.exp(outputs[0][0][1]) / sum(np.exp(outputs[0][0]))
            
            if s1_prob > 0.85:
                print(f"ANOMALY DETECTED (S1 Confidence: {s1_prob:.2f}) -> Triggering Emergency Brake!")
                # Local TC throttle command (simulated)
                os.system("tc qdisc add dev eth0 root tbf rate 1mbit burst 32kbit latency 400ms")
                
                # Escalate to System 2
                alert_payload = {
                    "source": "Edge_Node_1",
                    "confidence": float(s1_prob),
                    "raw_telemetry": telemetry
                }
                mqtt_client.publish("security/alerts", json.dumps(alert_payload))
                
    except Exception as e:
        print(f"Error processing packet: {e}")
