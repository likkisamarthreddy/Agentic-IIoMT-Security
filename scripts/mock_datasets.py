import pandas as pd
import numpy as np
import os

os.makedirs("DATASETS/BoT-IoT", exist_ok=True)
os.makedirs("DATASETS/UNSW-NB15", exist_ok=True)

# BoT-IoT Mock (Needs a 'category' or 'attack' column, some categorical like 'proto', 'state')
n = 100
bot_iot_df = pd.DataFrame({
    'pkSeqID': range(n),
    'proto': np.random.choice(['tcp', 'udp', 'icmp'], n),
    'saddr': ['192.168.1.1']*n,
    'sport': np.random.randint(1024, 65535, n),
    'daddr': ['192.168.1.2']*n,
    'dport': np.random.randint(1024, 65535, n),
    'seq': range(n),
    'stddev': np.random.random(n),
    'N_IN_Conn_P_SrcIP': np.random.randint(1, 100, n),
    'min': np.random.random(n),
    'state_number': np.random.randint(1, 5, n),
    'mean': np.random.random(n),
    'N_IN_Conn_P_DstIP': np.random.randint(1, 100, n),
    'drate': np.random.random(n),
    'srate': np.random.random(n),
    'max': np.random.random(n),
    'category': np.random.choice(['DDoS', 'DoS', 'Reconnaissance', 'Normal'], n)
})
bot_iot_df.to_csv("DATASETS/BoT-IoT/mock.csv", index=False)

# UNSW-NB15 Mock (Needs 'attack_cat', 'id', 'proto', 'service', 'state')
unsw_df = pd.DataFrame({
    'id': range(n),
    'dur': np.random.random(n),
    'proto': np.random.choice(['tcp', 'udp', 'icmp'], n),
    'service': np.random.choice(['http', 'ftp', 'dns', '-'], n),
    'state': np.random.choice(['FIN', 'INT', 'CON'], n),
    'spkts': np.random.randint(1, 100, n),
    'dpkts': np.random.randint(1, 100, n),
    'sbytes': np.random.randint(100, 10000, n),
    'dbytes': np.random.randint(100, 10000, n),
    'rate': np.random.random(n),
    'sttl': np.random.randint(1, 255, n),
    'dttl': np.random.randint(1, 255, n),
    'attack_cat': np.random.choice(['Normal', 'Fuzzers', 'Analysis', 'Backdoors'], n)
})
unsw_df.to_csv("DATASETS/UNSW-NB15/mock.csv", index=False)

print("Mock datasets created successfully.")
