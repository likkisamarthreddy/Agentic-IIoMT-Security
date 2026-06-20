from flask import Flask, render_template, request, jsonify
import paho.mqtt.client as mqtt
import json
import threading
import time
import os

app = Flask(__name__)

MQTT_HOST = os.environ.get("MQTT_HOST", "localhost")
MQTT_PORT = int(os.environ.get("MQTT_PORT", 1883))

# In-memory store for demo
notifications = []

mqtt_client = mqtt.Client(client_id="hitl-dashboard")

def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker with code {rc}")
    client.subscribe("iimt/hitl/notifications")

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        # prepend to keep newest first
        notifications.insert(0, data)
        # keep last 50
        if len(notifications) > 50:
            notifications.pop()
    except Exception as e:
        print(f"Error parsing MQTT message: {e}")

mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

# Run MQTT in background
def mqtt_loop():
    while True:
        try:
            mqtt_client.connect(MQTT_HOST, MQTT_PORT, 60)
            mqtt_client.loop_forever()
        except Exception as e:
            print(f"MQTT connection failed: {e}. Retrying in 5s...")
            time.sleep(5)

threading.Thread(target=mqtt_loop, daemon=True).start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/notifications')
def get_notifications():
    return jsonify(notifications)

@app.route('/api/override', methods=['POST'])
def submit_override():
    data = request.json
    device_id = data.get('device_id')
    action = data.get('action') # "APPROVE", "REJECT", "MODIFY"
    new_mitigation = data.get('new_mitigation')
    
    override_payload = {
        "timestamp": time.time(),
        "device_id": device_id,
        "hitl_action": action,
        "new_mitigation": new_mitigation,
        "issuer": "clinician"
    }
    
    # Publish override back to gateway
    mqtt_client.publish("iimt/gateway/overrides", json.dumps(override_payload))
    
    # Update local state so it appears handled
    for notif in notifications:
        if notif.get("device_id") == device_id:
            notif["status"] = action
            if new_mitigation:
                notif["action_taken"] = new_mitigation
    
    return jsonify({"success": True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
