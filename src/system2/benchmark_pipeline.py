import sys
import os
import time
import json
import logging
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from system2.gateway_agent import GatewayAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("iimt.benchmark")
        
def run_benchmark():
    logger.info("Initializing System 2 Benchmark Pipeline (3-Stage Edition)...")
    
    config = {
        "system2": {
            "risk_metric": {"alpha": 0.5, "beta": 0.3, "gamma": 0.2},
            "ollama_hostname": "localhost",
            "reasoning": {"max_iterations": 3}
        },
        "mqtt": {"broker_host": "localhost", "broker_port": 1883}
    }
    
    try:
        agent = GatewayAgent(config=config)
    except Exception as e:
        logger.error(f"Failed to initialize GatewayAgent: {e}")
        return

    # Mock the MQTT network delay (uplink and downlink)
    mock_network_delay = 0.005 # 5ms up, 5ms down = 10ms total
    
    # We will simulate the latency of the entire Edge Pipeline (Stage 0 + Stage 1 + Stage 2)
    # IsolationForest prediction on CPU takes ~0.1ms
    # CNN-BiGRU inference takes ~0.8ms
    # Total edge latency approximately 1.0ms
    mock_tau_edge = 0.0010 
    
    test_alerts = [
        {
            "alert_id": "AL-1001",
            "device_id": "sensor-1",
            "source_agent": "edge-node-1",
            "predicted_class": "MITM",
            "anomaly_score": 0.98,
            "flow_features": {"packets_per_sec": 500, "avg_payload": 100},
            "timestamp": time.time()
        },
        {
            "alert_id": "AL-1002",
            "device_id": "sensor-2",
            "source_agent": "edge-node-2",
            "predicted_class": "Spoofing",
            "anomaly_score": 0.96,
            "flow_features": {"packets_per_sec": 50, "avg_payload": 64},
            "timestamp": time.time()
        }
    ]

    logger.info("Running ReAct Loop simulation with perf_counter()...")
    results = []
    
    for alert in test_alerts:
        logger.info(f"--- Processing simulated alert: {alert['predicted_class']} ---")
        
        # Patch SDN controller to act instantly so we measure SLM + reasoning correctly
        original_apply = agent.sdn_controller.apply_rule
        def mock_apply(params):
            time.sleep(0.005) # Simulate 5ms SDN controller application time
            return original_apply(params)
        agent.sdn_controller.apply_rule = mock_apply
        
        t0 = time.perf_counter()
        
        # [1] Edge inference (Stage 0 + Stage 1 + Stage 2)
        time.sleep(mock_tau_edge)
        t1 = time.perf_counter()
        
        # [2] Network uplink
        time.sleep(mock_network_delay)
        t2 = time.perf_counter()
        
        # [3] Gateway Reasoning (SLM ReAct)
        agent.process_alert(alert)
        t3 = time.perf_counter()
        
        # [4] Action Execution & Network downlink
        time.sleep(mock_network_delay)
        t4 = time.perf_counter()
        
        tau_edge = (t1 - t0) * 1000
        tau_comm = ((t2 - t1) + (t4 - t3)) * 1000  # Uplink + Downlink
        tau_agent = (t3 - t2) * 1000
        
        tau_action = 5.0 
        tau_agent_clean = tau_agent - tau_action
        
        T_ttm = (t4 - t0) * 1000
        
        results.append({
            "alert": alert["predicted_class"],
            "tau_edge": tau_edge,
            "tau_agent": tau_agent_clean,
            "tau_comm": tau_comm,
            "tau_action": tau_action,
            "T_ttm": T_ttm
        })

    print("\n" + "="*80)
    print("                 SYSTEM 2 END-TO-END BENCHMARK RESULTS")
    print("="*80)
    print(f"| {'Attack Class':<15} | {'τ_edge (ms)':<11} | {'τ_comm (ms)':<11} | {'τ_agent (ms)':<12} | {'τ_action (ms)':<13} | {'T_ttm (ms)':<10} |")
    print("|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['alert']:<15} | {r['tau_edge']:>11.2f} | {r['tau_comm']:>11.2f} | {r['tau_agent']:>12.2f} | {r['tau_action']:>13.2f} | {r['T_ttm']:>10.2f} |")
    print("="*80)
    
if __name__ == "__main__":
    run_benchmark()
