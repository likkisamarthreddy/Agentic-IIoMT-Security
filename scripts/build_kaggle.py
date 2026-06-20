import json
import os

# anchor CWD to repo root so the relative source paths below resolve
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

files_to_merge = [
    "src/data/preprocessor.py",
    "src/system1/models/cnn_bigru.py",
    "src/system1/training/trainer.py"
]

header = """# ==========================================
# Kaggle Training Script for IIoMT CNN-BiGRU
# ==========================================
# 1. Upload this script to a Kaggle Notebook.
# 2. Attach the CICIoMT2024 dataset.
# 3. Change DATASET_PATH below to point to the CSV.
# 4. Run the cell!

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
logger = logging.getLogger("kaggle_trainer")

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
            "epochs": 50,
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
    # CHANGE THIS to your Kaggle dataset path!
    # Example: "/kaggle/input/ciciomt2024/train/train.csv"
    DATASET_PATH = "/kaggle/input/ciciomt2024/train/train.csv"
    
    if not os.path.exists(DATASET_PATH):
        logger.error(f"Dataset not found at {DATASET_PATH}. Please attach the dataset to the Kaggle notebook.")
        exit(1)
        
    logger.info("Initializing Preprocessor...")
    prep = DataPreprocessor(config_path=None) # We use the hardcoded CONFIG if config_path is None
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
    model = CNNBiGRU(
        num_features=CONFIG["data"]["num_features"],
        num_classes=num_classes,
        config_path=None
    )
    model._config = CONFIG["system1"]["model"]
    
    logger.info("Initializing Trainer...")
    trainer = ModelTrainer(model, CONFIG["system1"]["training"])
    
    logger.info("Starting Training...")
    trainer.train(train_loader, test_loader, epochs=CONFIG["system1"]["training"]["epochs"])
    
    logger.info("Evaluating...")
    trainer.evaluate(model, test_loader)
    
    # Save the model
    os.makedirs("checkpoints", exist_ok=True)
    trainer.save_checkpoint(model, "checkpoints/cnn_bigru_fp32.pt")
    
    with open("checkpoints/label_mapping.json", "w") as f:
        native_mapping = {k: int(v) for k, v in label_mapping.items()}
        json.dump(native_mapping, f, indent=2)
        
    logger.info("Training complete! You can now download checkpoints/cnn_bigru_fp32.pt and checkpoints/label_mapping.json")

"""

def strip_imports(code):
    lines = code.split("\\n")
    cleaned = []
    for line in lines:
        if line.startswith("import ") or line.startswith("from ") or line.startswith("from __future__"):
            continue
        # Remove yaml loading since we hardcoded it
        if "yaml.safe_load" in line:
            continue
        cleaned.append(line)
    return "\\n".join(cleaned)

with open("kaggle_training.py", "w", encoding="utf-8") as f_out:
    f_out.write(header)
    for file in files_to_merge:
        with open(file, "r", encoding="utf-8") as f_in:
            code = f_in.read()
            f_out.write(strip_imports(code))
            f_out.write("\\n\\n")
    
    f_out.write(main_block)

print("kaggle_training.py generated successfully.")
