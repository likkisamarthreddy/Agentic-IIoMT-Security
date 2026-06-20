"""
CSV Traffic Replay Tool (tcpreplay alternative for CSV datasets)
================================================================

This tool blasts CICIoMT2024 CSV feature vectors over MQTT/TCP to
the Mininet edge nodes, accurately simulating a live traffic stream.
It dynamically throttles the send rate to hit the 500 pkt/sec target
defined in Phase 1 of the paper.
"""

import time
import json
import logging
import argparse
import pandas as pd
import paho.mqtt.client as mqtt

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

def replay_csv(csv_path: str, broker: str, port: int, target_rate: float, max_rows: int = None):
    logger.info(f"Loading dataset: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        return

    if max_rows and max_rows < len(df):
        df = df.head(max_rows)
        
    client = mqtt.Client(client_id="csv_replay_node")
    try:
        client.connect(broker, port, 60)
    except Exception as e:
        logger.error(f"Failed to connect to Mosquitto at {broker}:{port} - {e}")
        return
        
    client.loop_start()
    
    logger.info(f"Starting replay of {len(df)} packets at ~{target_rate} pkt/sec...")
    
    interval = 1.0 / target_rate
    start_time = time.time()
    sent = 0
    
    edge_agents = ["edge-1", "edge-2", "edge-3"]
    
    for _, row in df.iterrows():
        loop_start = time.time()
        
        # Convert row to JSON feature vector
        payload = row.to_json()
        
        # Publish to the edge nodes round-robin
        target_agent = edge_agents[sent % len(edge_agents)]
        client.publish(f"iimt/traffic/{target_agent}", payload, qos=0)
        sent += 1
        
        # Throttle to maintain target_rate
        elapsed = time.time() - loop_start
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
            
        if sent % 500 == 0:
            actual_rate = sent / (time.time() - start_time)
            logger.info(f"Replayed {sent} packets... Current rate: {actual_rate:.2f} pkt/sec")

    client.loop_stop()
    client.disconnect()
    
    total_time = time.time() - start_time
    final_rate = sent / total_time
    logger.info("=" * 50)
    logger.info(f"REPLAY COMPLETE")
    logger.info(f"Total packets: {sent}")
    logger.info(f"Total time:    {total_time:.2f} seconds")
    logger.info(f"Avg rate:      {final_rate:.2f} pkt/sec")
    logger.info("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSV Traffic Replay")
    parser.add_argument("--csv", type=str, required=True, help="Path to CICIoMT2024 test CSV")
    parser.add_argument("--broker", type=str, default="127.0.0.1", help="MQTT Broker IP")
    parser.add_argument("--port", type=int, default=1883, help="MQTT Broker Port")
    parser.add_argument("--rate", type=float, default=500.0, help="Target packets per second")
    parser.add_argument("--max_rows", type=int, default=10000, help="Max rows to replay")
    
    args = parser.parse_args()
    replay_csv(args.csv, args.broker, args.port, args.rate, args.max_rows)
