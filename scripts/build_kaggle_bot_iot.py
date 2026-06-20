import json
import os

# anchor CWD to repo root so the relative source paths below resolve
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

files_to_merge = [
    "src/data/preprocessor.py",
    "src/data/bot_iot_loader.py",
    "src/system1/models/cnn_bigru.py",
    "src/system1/training/trainer.py",
    "src/evaluation/metrics_collector.py"
]

header = """# ==========================================
# Kaggle Training Script for IIoMT CNN-BiGRU (BoT-IoT)
# ==========================================
# 1. Upload this script to a Kaggle Notebook.
# 2. Attach the BoT-IoT dataset (vigneshvenkateswaran/bot-iot).
# 3. Run the cell!

import os
import time
import copy
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("kaggle_bot_iot")

# --- Default Configurations (Hardcoded to remove YAML dependency) ---
CONFIG = {
    "data": {
        "window_size": 1,
        "test_ratio": 0.2,
        "random_seed": 42,
        "num_features": 46
    },
    "system1": {
        "training": {
            "batch_size": 2048,
            "epochs": 10,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "patience": 10,
            "scheduler_factor": 0.5,
            "scheduler_patience": 5
        },
        "model": {
            "conv1_filters": 64,
            "conv2_filters": 128,
            "kernel_size": 3,
            "pool_size": 2,
            "gru_hidden_size": 64,
            "gru_num_layers": 2,
            "fc1_size": 64,
            "dropout_rate": 0.3
        }
    }
}

"""

main_block = """
# =====================================================================
# KAGGLE EXECUTION BLOCK
# =====================================================================
if __name__ == "__main__":
    import glob
    
    # Kaggle dataset directory
    KAGGLE_DIR = "/kaggle/input/datasets/vigneshvenkateswaran/bot-iot"
    if not os.path.exists(KAGGLE_DIR):
        KAGGLE_DIR = "/kaggle/input/bot-iot" # fallback
        if not os.path.exists(KAGGLE_DIR):
            logger.error(f"Dataset not found at {KAGGLE_DIR}. Please attach the dataset to the Kaggle notebook.")
            exit(1)
            
    # Find a CSV file to load. BoT-IoT usually has a 10-best features or full version.
    csv_files = glob.glob(f"{KAGGLE_DIR}/**/*.csv", recursive=True)
    if not csv_files:
        logger.error("No CSV files found in the dataset directory.")
        exit(1)
        
    DATASET_PATH = csv_files[0]
    logger.info(f"Using dataset file: {DATASET_PATH}")
    
    logger.info("Initializing Preprocessor...")
    prep = BotIoTLoader(config_path=None) 
    prep._config = CONFIG
    
    logger.info("Loading CSV...")
    df = prep.load_csv(DATASET_PATH)
    
    logger.info("Preprocessing...")
    X_train, X_test, y_train, y_test, label_mapping = prep.prepare_pipeline(
        df, test_ratio=CONFIG["data"]["test_ratio"], window_size=1
    )
    
    train_loader, test_loader = prep.get_dataloaders(
        X_train, X_test, y_train, y_test, batch_size=CONFIG["system1"]["training"]["batch_size"]
    )
    
    logger.info("Initializing Model...")
    num_classes = len(label_mapping)
    num_features = X_train.shape[-1]
    
    model = CNNBiGRU(
        num_features=num_features,
        num_classes=num_classes,
        config_path=None
    )
    model._config = CONFIG["system1"]["model"]
    
    logger.info("Initializing Trainer...")
    trainer = ModelTrainer(model, CONFIG["system1"]["training"])
    
    logger.info("Starting Training...")
    trainer.train(train_loader, test_loader, epochs=CONFIG["system1"]["training"]["epochs"])
    
    logger.info("Evaluating...")
    report = trainer.evaluate(model, test_loader)
    logger.info(f"FP32 Accuracy: {report.get('accuracy', 'N/A')}")
    
    # ---------------------------------------------------------
    # AGENTIC GOVERNANCE METRICS
    # ---------------------------------------------------------
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
    logger.info("=========================================")
    logger.info("EVALUATION COMPLETE")
"""

def strip_imports(code):
    lines = code.split("\\n")
    cleaned = []
    for line in lines:
        if line.startswith("import ") or line.startswith("from ") or line.startswith("from __future__"):
            # keep numpy and torch stuff
            if "numpy" in line or "torch" in line or "pandas" in line:
                pass
            else:
                continue
        # Remove yaml loading
        if "yaml.safe_load" in line:
            continue
        # Remove relative imports from within the project
        if line.startswith("from data.") or line.startswith("from system1.") or line.startswith("from evaluation."):
            continue
        cleaned.append(line)
    return "\\n".join(cleaned)

with open("kaggle_bot_iot_training.py", "w", encoding="utf-8") as f_out:
    f_out.write(header)
    for file in files_to_merge:
        with open(file, "r", encoding="utf-8") as f_in:
            code = f_in.read()
            f_out.write(strip_imports(code))
            f_out.write("\\n\\n")
    
    f_out.write(main_block)

print("kaggle_bot_iot_training.py generated successfully.")
