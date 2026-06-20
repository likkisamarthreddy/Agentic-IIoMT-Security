import paho.mqtt.client as mqtt
import json

payload = {
    "device_id": "sensor-1",
    "src_ip": "10.0.0.50",
    "features": [10.0] * 46
}

client = mqtt.Client()
client.connect("localhost", 1883, 60)
client.publish("iimt/traffic/edge-1", json.dumps(payload))
client.publish("iimt/traffic/edge-1", json.dumps(payload))
client.publish("iimt/traffic/edge-1", json.dumps(payload))
client.disconnect()
print("Sent test payloads to trigger anomaly!")
