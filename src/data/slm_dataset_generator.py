"""
SLM Reasoning Dataset Generator.

Generates a synthetic dataset of 500+ Reason-and-Act examples for fine-tuning
the System 2 Small Language Model (Phi-3-mini). The dataset maps System 1
alerts (along with contextual information like patient vitals, historical logs)
to System 2 mitigation playbooks.
"""

import json
import random
from pathlib import Path

def generate_slm_dataset(output_path: str, num_samples: int = 600):
    attack_types = ["DDoS", "DoS", "Reconnaissance", "Spoofing", "MITM", "Benign"]
    devices = [
        "Infusion_Pump_A", "Patient_Monitor_B", "MRI_Scanner_C", 
        "Ventilator_D", "Modbus_Sensor_E", "PLC_Controller_F"
    ]
    
    dataset = []
    
    for i in range(num_samples):
        attack = random.choice(attack_types)
        device = random.choice(devices)
        
        # Contextual metrics
        confidence = round(random.uniform(0.6, 0.99), 2)
        kde_score = round(random.uniform(0.01, 5.0), 2)
        criticality = random.choice(["Low", "Medium", "High", "Critical"])
        
        # Scenario generation
        if attack == "Benign":
            instruction = (
                f"System 1 reported a Benign traffic pattern from {device}. "
                f"Confidence: {confidence}. KDE Anomaly Score: {kde_score}. "
                f"Device Criticality: {criticality}."
            )
            output = (
                f"Thought: The traffic is classified as Benign and the KDE score ({kde_score}) "
                f"is within normal operational bounds. No immediate threat is detected.\n"
                f"Action: Route to Continuous Context Event Base for baseline monitoring.\n"
                f"Mitigation: None."
            )
        elif attack in ["DDoS", "DoS"]:
            instruction = (
                f"System 1 detected a {attack} attack from {device}. "
                f"Confidence: {confidence}. KDE Anomaly Score: {kde_score}. "
                f"Device Criticality: {criticality}. High SYN/ACK packet rate observed."
            )
            if criticality in ["High", "Critical"]:
                output = (
                    f"Thought: A {attack} attack is underway. The device ({device}) has "
                    f"a {criticality} criticality. A full quarantine may disrupt critical operations. "
                    f"An Emergency Brake throttle is required while maintaining primary telemetry.\n"
                    f"Action: Execute SDN Traffic Throttling rule.\n"
                    f"Mitigation: Throttle unauthenticated packets on {device} by 80%. Escalate to Human-in-the-Loop."
                )
            else:
                output = (
                    f"Thought: A {attack} attack is underway. The device ({device}) has "
                    f"a {criticality} criticality. We can safely isolate the device to protect the subnet.\n"
                    f"Action: Execute L2 Network Isolation.\n"
                    f"Mitigation: Block all inbound/outbound traffic on {device} switch port."
                )
        elif attack == "Spoofing":
            instruction = (
                f"System 1 detected MAC/IP Spoofing originating from {device}. "
                f"Confidence: {confidence}. KDE Anomaly Score: {kde_score}. "
                f"Device Criticality: {criticality}. Header lengths are abnormal."
            )
            output = (
                f"Thought: Device identity ({device}) is compromised via spoofing. "
                f"We must trigger a re-authentication protocol to verify device identity.\n"
                f"Action: Cryptographic Re-authentication.\n"
                f"Mitigation: Force 802.1X re-auth on switch port. If failed, segment into quarantine VLAN."
            )
        elif attack == "MITM":
            instruction = (
                f"System 1 detected a Man-in-the-Middle (MITM) attack near {device}. "
                f"Confidence: {confidence}. KDE Anomaly Score: {kde_score}. "
                f"Device Criticality: {criticality}. Irregular IAT and double ACKs detected."
            )
            output = (
                f"Thought: Communication integrity for {device} is compromised by MITM interception. "
                f"We must establish a secure overlay and notify the operator.\n"
                f"Action: SDN Micro-segmentation.\n"
                f"Mitigation: Enforce IPsec/TLS strict mode on {device}. Alert clinical operator for manual override."
            )
        else: # Reconnaissance
            instruction = (
                f"System 1 detected {attack} scanning from {device}. "
                f"Confidence: {confidence}. KDE Anomaly Score: {kde_score}. "
                f"Device Criticality: {criticality}."
            )
            output = (
                f"Thought: The device {device} is scanning the network. This indicates potential malware infection "
                f"attempting lateral movement.\n"
                f"Action: Read-Only Telemetry restriction.\n"
                f"Mitigation: Block outbound TCP/UDP handshakes from {device}. Allow only MQTT/CoAP read telemetry."
            )
            
        dataset.append({"instruction": instruction, "output": output})
        
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_file, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
            
    print(f"Generated {len(dataset)} SLM reasoning examples at {out_file}")

if __name__ == "__main__":
    generate_slm_dataset("data/iiomt_reasoning_dataset.jsonl", 600)
