# ==========================================
# Kaggle Training Script for IIoMT CNN-BiGRU (BoT-IoT)
# Memory-optimized to avoid Kaggle OOM restarts on large data.
# ==========================================
# 1. Upload this script to a Kaggle Notebook.
# 2. Attach the BoT-IoT dataset (vigneshvenkateswaran/bot-iot).
# 3. Run the cell!

import os
import gc
import glob
import logging
from typing import Dict, List, Tuple

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

# --- Default Configurations ---
CONFIG = {
    "data": {
        "window_size": 1,
        "test_ratio": 0.2,
        "random_seed": 42,
        "num_features": 46,
        # Memory controls --------------------------------------------------
        # Max rows to KEEP in memory after subsampling (None = keep all read).
        "max_rows": 500_000,
        # Rows read per chunk while streaming the CSV from disk.
        "chunk_size": 200_000,
        # Fraction of each chunk to keep (1.0 = keep all). Lower this if you
        # still hit OOM, e.g. 0.5 keeps half the rows.
        "sample_frac": 1.0,
        # Categorical columns with more unique values than this are LABEL-encoded
        # (one integer column) instead of one-hot encoded. One-hot encoding
        # high-cardinality columns like IPs/ports creates millions of columns
        # and is the #1 cause of OOM crashes on BoT-IoT.
        "max_onehot_cardinality": 50,
    },
    "system1": {
        "training": {
            "batch_size": 1024,
            "epochs": 10,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
        },
        "model": {
            "conv1_filters": 64,
            "conv2_filters": 128,
            "kernel_size": 3,
            "pool_size": 2,
            "gru_hidden_size": 64,
            "gru_num_layers": 2,
            "fc1_size": 64,
            "dropout_rate": 0.3,
        },
    },
}


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memory footprint by downcasting numeric columns in place."""
    for col in df.select_dtypes(include=["float"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["integer"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


class BotIoTLoader:
    def __init__(self) -> None:
        self._test_ratio = CONFIG["data"]["test_ratio"]
        self._window_size = CONFIG["data"]["window_size"]
        self._random_seed = CONFIG["data"]["random_seed"]
        self._max_rows = CONFIG["data"]["max_rows"]
        self._chunk_size = CONFIG["data"]["chunk_size"]
        self._sample_frac = CONFIG["data"]["sample_frac"]
        self._max_onehot_cardinality = CONFIG["data"]["max_onehot_cardinality"]

    # ------------------------------------------------------------------
    # Streaming CSV load: read in chunks, downcast, subsample, then concat.
    # This keeps peak RAM far below a single giant read_csv().
    # ------------------------------------------------------------------
    def load_csv(self, filepath: str) -> pd.DataFrame:
        logger.info("Streaming CSV in chunks to limit peak memory...")
        kept: List[pd.DataFrame] = []
        total = 0
        for chunk in pd.read_csv(filepath, chunksize=self._chunk_size, low_memory=False):
            chunk = downcast(chunk)
            if self._sample_frac < 1.0:
                chunk = chunk.sample(frac=self._sample_frac, random_state=self._random_seed)
            kept.append(chunk)
            total += len(chunk)
            if self._max_rows is not None and total >= self._max_rows:
                logger.info(f"Reached max_rows cap ({self._max_rows}); stopping read.")
                break
        df = pd.concat(kept, ignore_index=True)
        del kept
        gc.collect()
        if self._max_rows is not None and len(df) > self._max_rows:
            df = df.sample(n=self._max_rows, random_state=self._random_seed).reset_index(drop=True)
        logger.info(f"Loaded dataframe shape: {df.shape}")
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(axis=1, how="all")
        df = df.fillna(0)
        df = df.drop_duplicates()
        logger.info(f"Shape after cleaning: {df.shape}")
        return df

    def encode_labels(self, df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
        le = LabelEncoder()
        df[label_col] = le.fit_transform(df[label_col].astype(str))
        mapping = dict(zip(le.classes_, (int(v) for v in le.transform(le.classes_))))
        return df, mapping

    def prepare_pipeline(self, df: pd.DataFrame, label_col: str = "category"):
        if label_col not in df.columns:
            if "attack" in df.columns:
                label_col = "attack"
            else:
                label_col = df.columns[-1]
        logger.info(f"Target label column: '{label_col}'")

        df = self.clean(df)
        # Drop identifiers AND high-cardinality address/port columns that would
        # explode under one-hot encoding (these carry little generalizable signal).
        drop_cols = [
            "pkSeqID", "seq", "stime", "ltime", "flgs", "flgs_number", "state_number",
            "saddr", "daddr", "sport", "dport", "smac", "dmac", "soui", "doui",
        ]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if label_col in categorical_cols:
            categorical_cols.remove(label_col)

        # Split categoricals by cardinality: low -> one-hot, high -> label-encode.
        onehot_cols, labelenc_cols = [], []
        for col in categorical_cols:
            if df[col].nunique() <= self._max_onehot_cardinality:
                onehot_cols.append(col)
            else:
                labelenc_cols.append(col)

        for col in labelenc_cols:
            logger.info(f"Label-encoding high-cardinality column '{col}' ({df[col].nunique()} uniques)")
            df[col] = LabelEncoder().fit_transform(df[col].astype(str)).astype(np.float32)

        if onehot_cols:
            logger.info(f"One-hot encoding low-cardinality columns: {onehot_cols}")
            df = pd.get_dummies(df, columns=onehot_cols, drop_first=True, dtype=np.float32)

        df, label_mapping = self.encode_labels(df, label_col=label_col)
        logger.info(f"Label mapping: {label_mapping}")

        feature_cols = [c for c in df.columns if c != label_col]
        # Extract arrays then free the dataframe immediately.
        X = df[feature_cols].to_numpy(dtype=np.float32)
        y = df[label_col].to_numpy(dtype=np.int64)
        del df
        gc.collect()

        class_counts = pd.Series(y).value_counts()
        valid_classes = class_counts[class_counts >= 2].index
        valid_mask = np.isin(y, valid_classes)
        if not valid_mask.all():
            dropped = len(y) - int(valid_mask.sum())
            logger.warning(f"Dropping {dropped} samples from singleton classes (for stratify).")
            X = X[valid_mask]
            y = y[valid_mask]
        del valid_mask
        gc.collect()

        if len(y) == 0:
            raise ValueError("Dataset is empty after cleaning!")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self._test_ratio, random_state=self._random_seed, stratify=y
        )
        del X, y
        gc.collect()

        # Scale in place to avoid an extra full-size copy.
        scaler = StandardScaler(copy=False)
        X_train = scaler.fit_transform(X_train).astype(np.float32, copy=False)
        X_test = scaler.transform(X_test).astype(np.float32, copy=False)

        X_train = X_train.reshape(-1, 1, X_train.shape[1])
        X_test = X_test.reshape(-1, 1, X_test.shape[1])
        return X_train, X_test, y_train, y_test, label_mapping

    def get_dataloaders(self, X_train, X_test, y_train, y_test, batch_size):
        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        return train_loader, test_loader


class CNNBiGRU(nn.Module):
    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        cfg = CONFIG["system1"]["model"]
        self.conv1 = nn.Conv1d(1, cfg["conv1_filters"], cfg["kernel_size"], padding=cfg["kernel_size"] // 2)
        self.bn1 = nn.BatchNorm1d(cfg["conv1_filters"])
        self.conv2 = nn.Conv1d(cfg["conv1_filters"], cfg["conv2_filters"], cfg["kernel_size"], padding=cfg["kernel_size"] // 2)
        self.bn2 = nn.BatchNorm1d(cfg["conv2_filters"])
        self.pool = nn.MaxPool1d(cfg["pool_size"])

        dummy = torch.zeros(1, 1, num_features)
        dummy = self.pool(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(dummy)))))))
        # After conv+pool, tensor shape is [B, C=conv2_filters, L=pooled_length].
        # We then transpose to [B, seq_len=L, input_size=C] before GRU.
        conv_channels = dummy.shape[1]
        conv_out_len = dummy.shape[2]

        self.gru = nn.GRU(conv_channels, cfg["gru_hidden_size"], cfg["gru_num_layers"], batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(cfg["gru_hidden_size"] * 2 * conv_out_len, cfg["fc1_size"])
        self.dropout = nn.Dropout(cfg["dropout_rate"])
        self.fc2 = nn.Linear(cfg["fc1_size"], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.transpose(1, 2)
        gru_out, _ = self.gru(x)
        x = gru_out.contiguous().view(gru_out.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class ModelTrainer:
    def __init__(self, model: nn.Module, config: dict):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, train_loader: DataLoader, epochs: int):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config["learning_rate"], weight_decay=self.config["weight_decay"]
        )
        for epoch in range(epochs):
            self.model.train()
            total_loss, correct = 0.0, 0
            for X, y in train_loader:
                X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                outputs = self.model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * X.size(0)
                correct += (outputs.argmax(1) == y).sum().item()
            n = len(train_loader.dataset)
            logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/n:.4f} | Acc: {correct/n:.4f}")

    def evaluate(self, data_loader: DataLoader) -> dict:
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in data_loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                correct += (outputs.argmax(1) == y).sum().item()
                total += y.size(0)
        return {"accuracy": correct / total}


def compute_ecr(policy_ok_count: int, total: int) -> float:
    return policy_ok_count / total if total > 0 else 1.0


def compute_fer(false_escalation_count: int, total_escalations: int) -> float:
    return false_escalation_count / total_escalations if total_escalations > 0 else 0.0


def compute_gci(ratios: List[float], weights: List[float]) -> float:
    if not ratios or not weights or sum(weights) == 0:
        return 0.0
    return sum(c * w for c, w in zip(ratios, weights)) / sum(weights)


# =====================================================================
# KAGGLE EXECUTION BLOCK
# =====================================================================
if __name__ == "__main__":
    KAGGLE_DIR = "/kaggle/input/datasets/vigneshvenkateswaran/bot-iot"
    if not os.path.exists(KAGGLE_DIR):
        KAGGLE_DIR = "/kaggle/input/bot-iot"
        if not os.path.exists(KAGGLE_DIR):
            logger.error("Dataset not found. Attach the dataset to the Kaggle notebook.")
            raise SystemExit(1)

    csv_files = glob.glob(f"{KAGGLE_DIR}/**/*.csv", recursive=True)
    if not csv_files:
        logger.error("No CSV files found.")
        raise SystemExit(1)

    DATASET_PATH = csv_files[0]
    logger.info(f"Using dataset file: {DATASET_PATH}")

    prep = BotIoTLoader()
    df = prep.load_csv(DATASET_PATH)
    X_train, X_test, y_train, y_test, label_mapping = prep.prepare_pipeline(df)

    train_loader, test_loader = prep.get_dataloaders(
        X_train, X_test, y_train, y_test, batch_size=CONFIG["system1"]["training"]["batch_size"]
    )

    model = CNNBiGRU(num_features=X_train.shape[-1], num_classes=len(label_mapping))
    trainer = ModelTrainer(model, CONFIG["system1"]["training"])

    logger.info("Starting training...")
    trainer.train(train_loader, epochs=CONFIG["system1"]["training"]["epochs"])

    report = trainer.evaluate(test_loader)
    logger.info(f"FP32 Accuracy: {report.get('accuracy', 'N/A')}")

    # ---------------------------------------------------------
    # AGENTIC GOVERNANCE METRICS
    # ---------------------------------------------------------
    device = trainer.device
    model.eval()
    all_preds, all_probs, all_targets = [], [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)

    benign_cls = label_mapping.get("Normal", label_mapping.get("Benign", 0))
    max_probs = all_probs.max(axis=1)
    policy_ok_count = int((max_probs >= 0.85).sum())
    escalations = all_preds != benign_cls
    total_escalations = int(escalations.sum())
    false_escalation_count = int((escalations & (all_targets == benign_cls)).sum())

    ecr = compute_ecr(policy_ok_count, len(all_preds))
    fer = compute_fer(false_escalation_count, total_escalations)
    gci = compute_gci([ecr, 1.0 - fer], [0.6, 0.4])

    logger.info("--- Agentic Governance Metrics ---")
    logger.info(f"  Ethical Compliance Rate (ECR): {ecr:.4f}")
    logger.info(f"  False Escalation Rate (FER): {fer:.4f}")
    logger.info(f"  Governance Compliance Index (GCI): {gci:.4f}")
    logger.info("=========================================")
    logger.info("EVALUATION COMPLETE")
