# --- repo bootstrap: make src/ importable + anchor CWD to repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _os.path.join(_ROOT, "src") not in _sys.path:
    _sys.path.insert(0, _os.path.join(_ROOT, "src"))
_os.chdir(_ROOT)
# --- end bootstrap ---

import os
import time
import json
import yaml
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s")
logger = logging.getLogger("inference_test")

def load_config():
    with open("config/settings.yaml", "r") as f:
        return yaml.safe_load(f)

def run_test():
    logger.info("========================================================")
    logger.info("  REAL DATASET DEPLOYMENT INFERENCE TEST")
    logger.info("========================================================")

    # 1. Setup Models & Engines
    config = load_config()
    
    # System 1 Edge Agent dependencies
    import onnxruntime as ort
    from system1.edge_agent import EdgeAgent
    from system1.detection.kde_threshold import AdaptiveKDEThreshold
    from system1.detection.emergency_brake import EmergencyBrake
    
    # We will use the optimized expert model trained on the real dataset
    model_path = Path("system1/expert_bigru_int8.onnx")
    if not model_path.exists():
        model_path = Path("checkpoints/cnn_bigru_int8.onnx")
    if not model_path.exists():
        model_path = Path("checkpoints/cnn_bigru_fp32.onnx")
        
    logger.info(f"Loading Edge Model: {model_path}")
    sess_options = ort.SessionOptions()
    model = ort.InferenceSession(str(model_path), sess_options, providers=["CPUExecutionProvider"])
    
    kde = AdaptiveKDEThreshold(config_path=Path("config/settings.yaml"))
    brake = EmergencyBrake(kde_threshold=kde, config_path=Path("config/settings.yaml"))
    
    # We pass None for MQTT client so it runs in standalone mode
    edge_agent = EdgeAgent(agent_id="edge-test-node", model=model, kde_threshold=kde, emergency_brake=brake, config=config)

    # System 2 Gateway Dependencies
    from system2.reasoning.context_fusion import ContextFusionEngine
    from system2.reasoning.reason_act_loop import ReActLoop
    from system2.reasoning.symbolic_rules import SymbolicRuleEngine
    from system2.mitigation.action_playbook import ActionPlaybook
    
    policies_path = Path("config/safety_policies.yaml")
    context_engine = ContextFusionEngine(**config["system2"]["risk_metric"], config_path=Path("config/settings.yaml"))
    rule_engine = SymbolicRuleEngine(policies_path)
    playbook = ActionPlaybook(policies_path)
    react_loop = ReActLoop(context_engine=context_engine, rule_engine=rule_engine, action_playbook=playbook, config=config["system2"]["reasoning"])

    # 2. Load a subset of REAL traffic
    test_csv_path = Path("CICIOMT24/test/test.csv")
    if not test_csv_path.exists():
        logger.error(f"Cannot find {test_csv_path}")
        return
        
    logger.info(f"Loading real packets from {test_csv_path}...")
    # Load 5000 rows randomly to find some attacks
    df = pd.read_csv(test_csv_path, nrows=20000)
    
    # Mapping for display
    def map_label(label_str):
        if pd.isna(label_str): return "Benign"
        s = str(label_str).lower()
        if "benign" in s or "normal" in s: return "Benign"
        elif "arp" in s or "mitm" in s or "poison" in s: return "MITM"
        elif "ddos" in s: return "DDoS"
        elif "dos" in s: return "DoS"
        elif "recon" in s or "scan" in s: return "Reconnaissance"
        elif "spoof" in s or "xss" in s or "sql" in s or "injection" in s: return "Spoofing"
        return "Benign"
        
    df['macro_class'] = df['label'].apply(map_label)
    
    # Pick one Benign and one Attack packet sequence
    attack_samples = df[df['macro_class'] != "Benign"]
    benign_samples = df[df['macro_class'] == "Benign"]
    
    if len(attack_samples) == 0:
        logger.warning("No attacks found in first 20000 rows. Re-run or increase subset.")
        return

    # Select specific features (same as preprocessor)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 46:
        numeric_cols = numeric_cols[:46]
        
    def simulate_packet_stream(samples_df, device_type, desc):
        logger.info(f"\n--- Simulating Stream: {desc} on {device_type} ---")
        
        # We need a sequence of packets (default length 50)
        seq_len = config.get("system1", {}).get("model", {}).get("sequence_length", 50)
        
        # Make sure we have enough rows
        if len(samples_df) < seq_len:
            logger.warning(f"Not enough samples for sequence (need {seq_len})")
            return
            
        sequence = samples_df.iloc[:seq_len]
        actual_label = sequence.iloc[-1]['macro_class']
        
        logger.info(f"Streaming {seq_len} packets into Edge Node...")
        
        result = None
        for i in range(seq_len):
            features = sequence.iloc[i][numeric_cols].values.astype(np.float32)
            # Normalise vaguely
            features = np.nan_to_num(features, 0)
            
            # Form packet
            packet = {
                "packet_id": i,
                "device_id": "dev-001",
                "device_type": device_type,
                "features": features.tolist()
            }
            
            # Feed to edge node
            res = edge_agent.process_packet(packet)
            if res:
                result = res
                
        if not result:
            logger.error("No result from edge agent.")
            return
            
        logger.info(f"Edge Node Inference Complete. Latency: {result['latency_ms']:.3f} ms")
        logger.info(f"Anomaly Score: {result['score']:.4f} (Is Anomalous: {result['is_anomalous']})")
        logger.info(f"Actual Traffic Class: {actual_label}")
        
        if result['is_anomalous']:
            logger.info(">> THREAT DETECTED! << Edge node is forwarding alert to Gateway (System 2)...")
            alert = {
                "alert_id": f"ALT-{int(time.time())}",
                "device_id": "dev-001",
                "device_type": device_type,
                "attack_type": actual_label, # Simulated classification
                "classifier_confidence": float(result['score']),
                "anomaly_score": float(result['score']),
                "timestamp": time.time(),
                "source_agent": "edge-test-node"
            }
            
            logger.info("\n[SYSTEM 2 GATEWAY REASONING LOOP]")
            react_res = react_loop.execute(alert)
            logger.info(f"Risk Score Computed: {react_res.risk_score:.4f}")
            logger.info(f"Action Taken: {react_res.action_taken}")
            logger.info(f"Explanation: {react_res.explanation_nl}")
        else:
            logger.info("Traffic is clean. No mitigation required.")

    # 1. Normal Traffic Simulation
    simulate_packet_stream(benign_samples, "patient_monitor", "Normal Telemetry")
    
    # 2. Attack Traffic Simulation
    simulate_packet_stream(attack_samples, "infusion_pump", "Real Attack Packet Stream")

if __name__ == "__main__":
    run_test()
