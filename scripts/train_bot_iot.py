import argparse
import logging
import sys
import json
from pathlib import Path
import os
import torch
import numpy as np

# --- repo bootstrap: make src/ importable ---
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
os.chdir(_ROOT)
# --- end bootstrap ---

from data.bot_iot_loader import BotIoTLoader
from system1.models.cnn_bigru import CNNBiGRU
from system1.training.trainer import ModelTrainer
from system1.quantization.quantizer import ModelQuantizer
from evaluation.metrics_collector import compute_ecr, compute_fer, compute_gci, compute_ri2, compute_cas
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("train_bot_iot")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to BoT-IoT CSV file")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    config_path = Path("config/settings.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("=" * 65)
    logger.info("  Training on BoT-IoT Dataset")
    logger.info("=" * 65)

    # 1. Load Data
    logger.info(f"Loading data from {args.csv}")
    loader = BotIoTLoader(config_path)
    df = loader.load_csv(args.csv)

    # 2. Preprocess
    result = loader.prepare_pipeline(df, test_ratio=0.2, window_size=1)
    X_train, X_test, y_train, y_test, label_mapping = result
    
    train_loader, test_loader = loader.get_dataloaders(
        X_train, X_test, y_train, y_test, batch_size=256
    )

    # 3. Model
    num_features = X_train.shape[-1]
    num_classes = len(label_mapping)
    model = CNNBiGRU(num_features=num_features, num_classes=num_classes, config_path=config_path)

    # 4. Train
    trainer = ModelTrainer(model, config["system1"]["training"])
    trainer.train(train_loader, test_loader, epochs=args.epochs)

    # 5. Evaluate
    report = trainer.evaluate(model, test_loader)
    logger.info(f"FP32 Accuracy: {report.get('accuracy', 'N/A')}")

    # Agentic Governance Metrics
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Try to find benign class, if not assume 0
    benign_cls = label_mapping.get("Normal", label_mapping.get("Benign", 0))
    policy_ok_count = 0
    total_constrained = len(all_preds)
    total_escalations = 0
    false_escalation_count = 0
    
    for i in range(total_constrained):
        if np.max(all_probs[i]) >= 0.85:
            policy_ok_count += 1
        if all_preds[i] != benign_cls:
            total_escalations += 1
            if all_targets[i] == benign_cls:
                false_escalation_count += 1

    ecr = compute_ecr(policy_ok_count, total_constrained)
    fer = compute_fer(false_escalation_count, total_escalations)
    gci = compute_gci([ecr, 1.0 - fer], [0.6, 0.4])
    
    logger.info("--- Agentic Governance Metrics ---")
    logger.info(f"  Ethical Compliance Rate (ECR): {ecr:.4f}")
    logger.info(f"  False Escalation Rate (FER): {fer:.4f}")
    logger.info(f"  Governance Compliance Index (GCI): {gci:.4f}")

    logger.info("Evaluation Complete.")

if __name__ == "__main__":
    main()
