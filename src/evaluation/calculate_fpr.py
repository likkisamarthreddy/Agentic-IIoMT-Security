import json
from pathlib import Path
from typing import Dict

def calculate_fpr(results_path: str):
    path = Path(results_path)
    if not path.exists():
        print(f"Error: Could not find {path}")
        return

    with open(path, "r") as f:
        data = json.load(f)

    cm = data.get("test_results", {}).get("confusion_matrix")
    if not cm:
        print("No confusion matrix found in results.")
        return

    # Labels map from our preprocessing
    # 0: Benign, 1: DDoS, 2: DoS, 3: MITM, 4: Reconnaissance, 5: Spoofing
    labels = ["Benign", "DDoS", "DoS", "MITM", "Reconnaissance", "Spoofing"]
    
    print("==========================================================")
    print(" False Positive Rate (FPR) Analysis (Test Set: ~1.1M pkts)")
    print("==========================================================")
    print(f"{'Class':<15} | {'TN':>8} | {'FP':>8} | {'FN':>8} | {'TP':>8} | {'FPR %':>8}")
    print("-" * 65)

    num_classes = len(labels)
    total_samples = sum(sum(row) for row in cm)

    for i in range(num_classes):
        # True Positives: diagonal element
        tp = cm[i][i]
        
        # False Positives: sum of column i, minus TP
        fp = sum(cm[j][i] for j in range(num_classes)) - tp
        
        # False Negatives: sum of row i, minus TP
        fn = sum(cm[i]) - tp
        
        # True Negatives: total samples - (TP + FP + FN)
        tn = total_samples - (tp + fp + fn)

        # False Positive Rate = FP / (FP + TN)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        print(f"{labels[i]:<15} | {tn:>8} | {fp:>8} | {fn:>8} | {tp:>8} | {fpr*100:>7.4f}%")

if __name__ == "__main__":
    calculate_fpr("checkpoints/training_results.json")
