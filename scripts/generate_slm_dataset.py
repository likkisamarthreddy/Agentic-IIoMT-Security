import json
import random

def generate_dataset(num_samples=5000, output_file="slm_synthetic_data.jsonl"):
    devices = [
        {"id": "infusion_pump", "criticality": "High", "rules": ["throttle", "alert_doctor"]},
        {"id": "heart_monitor", "criticality": "High", "rules": ["read_only", "alert_nurse"]},
        {"id": "hvac_controller", "criticality": "Low", "rules": ["isolate", "reboot", "ignore"]},
        {"id": "smart_lighting", "criticality": "Low", "rules": ["isolate", "ignore"]}
    ]
    
    alerts = ["DDoS", "Spoofing", "Man-in-the-Middle", "Port Scan"]
    vitals = ["Stable", "Critical", "Fluctuating", "Normal"]
    
    with open(output_file, "w") as f:
        for _ in range(num_samples):
            device = random.choice(devices)
            alert = random.choice(alerts)
            vital = random.choice(vitals)
            
            # Context fusion mock score
            confidence = random.uniform(0.6, 0.99)
            
            prompt = (f"[System] You are a clinical safety agent. Alert: {alert} detected on {device['id']}. "
                      f"Criticality: {device['criticality']}. Patient Vitals: {vital}. "
                      f"Available Actions: [throttle, isolate, reconfigure, ignore].")
            
            if device["id"] == "infusion_pump":
                if vital in ["Critical", "Fluctuating"]:
                    action = "throttle"
                    reasoning = f"Patient is {vital}. Full isolation could be fatal. Throttling traffic to maintain essential telemetry."
                else:
                    action = "throttle"
                    reasoning = f"Patient is {vital}. Isolation disrupts infusion logs. Throttling reduces {alert} impact while maintaining connection."
            elif device["id"] == "heart_monitor":
                action = "read_only"
                reasoning = f"Heart monitor must not be interrupted. Setting to read-only prevents {alert} manipulation."
            else:
                if confidence > 0.8:
                    action = "isolate"
                    reasoning = f"High confidence {alert} on low-criticality {device['id']}. Isolating to protect network."
                else:
                    action = "ignore"
                    reasoning = f"Low confidence {alert} on {device['id']}. Ignoring to prevent false positive disruption."

            response = f"[Action] {action}. [Reasoning] {reasoning}"
            
            f.write(json.dumps({"prompt": prompt, "response": response}) + "\n")
    print(f"Generated {num_samples} synthetic samples in {output_file}")

if __name__ == "__main__":
    generate_dataset()
