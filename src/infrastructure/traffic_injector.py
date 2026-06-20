import os
import json
import time
import socket
import pandas as pd
from tqdm import tqdm

GATEWAY_IP = "127.0.0.1"
GATEWAY_PORT = 1883
RATE_LIMIT_PKTS_PER_SEC = 500

def simulate_tcpreplay(csv_path):
    print(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    print(f"Streaming data at {RATE_LIMIT_PKTS_PER_SEC} packets/sec to {GATEWAY_IP}:{GATEWAY_PORT}...")
    delay = 1.0 / RATE_LIMIT_PKTS_PER_SEC
    
    for i, row in tqdm(df.iterrows(), total=len(df)):
        payload = json.dumps(row.to_dict()).encode('utf-8')
        sock.sendto(payload, (GATEWAY_IP, GATEWAY_PORT))
        time.sleep(delay)
        
    print("Streaming complete.")

if __name__ == "__main__":
    csv_file = r"C:\Users\user\Desktop\Agentic AI\CICIOMT24\test\test.csv"
    if os.path.exists(csv_file):
        simulate_tcpreplay(csv_file)
    else:
        print(f"Please provide a valid dataset CSV file at {csv_file}")
