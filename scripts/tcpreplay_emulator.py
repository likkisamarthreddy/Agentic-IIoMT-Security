import os
import time
import json
import random
import paho.mqtt.client as mqtt

MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
TARGET_PPS = int(os.getenv("TARGET_PPS", 500))
NUM_FEATURES = int(os.getenv("NUM_FEATURES", 97))

def main():
    client = mqtt.Client("tcpreplay_emulator")
    client.connect(MQTT_HOST, MQTT_PORT, 60)
    client.loop_start()

    print(f"Connected to {MQTT_HOST}:{MQTT_PORT}. Streaming at {TARGET_PPS} packets/sec.")
    sleep_time = 1.0 / TARGET_PPS

    agents = ["edge-1", "edge-2", "edge-3"]
    
    packet_id = 0
    try:
        while True:
            # Simulate a packet with NUM_FEATURES
            features = [random.random() for _ in range(NUM_FEATURES)]
            
            payload = {
                "device_id": f"dev_{random.randint(1, 100)}",
                "src_ip": f"10.0.0.{random.randint(2, 254)}",
                "features": features
            }
            
            target_agent = agents[packet_id % len(agents)]
            topic = f"iimt/traffic/{target_agent}"
            
            client.publish(topic, json.dumps(payload))
            
            packet_id += 1
            if packet_id % 1000 == 0:
                print(f"Streamed {packet_id} packets...")
                
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("Stopping stream.")
    finally:
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main()
