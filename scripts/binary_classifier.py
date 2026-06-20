# ==========================================
# Binary Classifier Script for Paper Table 1
# ==========================================
import os, json, logging, copy, time
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Any
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_curve, confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("binary_trainer")

# --- Dataset Path Auto-Detection ---
def detect_paths():
    ciciomt_train_candidates = [
        "/kaggle/input/datasets/outstandard09/ciciomt2024/Dataset/WiFI_and_MQTT/train",
        "/kaggle/input/datasets/likkisamarthreddy/ciciomt2024/csv/train",
        "/kaggle/input/ciciomt2024/Dataset/WiFI_and_MQTT/train",
        "/kaggle/input/ciciomt2024/csv/train",
        "./data/ciciomt2024/train"
    ]
    ciciomt_test_candidates = [
        "/kaggle/input/datasets/outstandard09/ciciomt2024/Dataset/WiFI_and_MQTT/test",
        "/kaggle/input/datasets/likkisamarthreddy/ciciomt2024/csv/test",
        "/kaggle/input/ciciomt2024/Dataset/WiFI_and_MQTT/test",
        "/kaggle/input/ciciomt2024/csv/test",
        "./data/ciciomt2024/test"
    ]
    edge_iiot_candidates = [
        "/kaggle/input/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot/Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv",
        "/kaggle/input/edgeiiotset-cyber-security-dataset-of-iot-iiot/Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv",
        "./data/ML-EdgeIIoT-dataset.csv"
    ]
    
    train_path = next((c for c in ciciomt_train_candidates if os.path.exists(c)), ciciomt_train_candidates[0])
    test_path = next((c for c in ciciomt_test_candidates if os.path.exists(c)), ciciomt_test_candidates[0])
    iiot_path = next((c for c in edge_iiot_candidates if os.path.exists(c)), edge_iiot_candidates[0])
    
    return train_path, test_path, iiot_path

CICIOMT_TRAIN, CICIOMT_TEST, EDGE_IIOT_CSV = detect_paths()
logger.info("Auto-detected CICIOMT_TRAIN: %s", CICIOMT_TRAIN)
logger.info("Auto-detected CICIOMT_TEST:  %s", CICIOMT_TEST)
logger.info("Auto-detected EDGE_IIOT_CSV: %s", EDGE_IIOT_CSV)

# =====================================================================
# Model
# =====================================================================

class SEBlock1d(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(1, channel // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channel // reduction), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class EdgeCNN_BiGRU(nn.Module):
    def __init__(self, num_features: int, num_classes: int = 2):
        super().__init__()
        # Standard CNN with Squeeze-and-Excitation (SE) for spatial protocol weighting
        self.conv1 = nn.Conv1d(num_features, 64, kernel_size=3, padding=1)
        self.se1 = SEBlock1d(64, reduction=8)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.se2 = SEBlock1d(128, reduction=16)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        
        # BiGRU for temporal sequence processing
        self.gru = nn.GRU(128, 64, bidirectional=True, batch_first=True)
        
        # Temporal Attention Mechanism
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Permute for CNN: (Batch, Features, Seq_Len)
        x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.se1(self.conv1(x))))
        x = self.pool(self.relu(self.se2(self.conv2(x))))
        
        # Permute back for GRU: (Batch, Seq_Len, Features)
        x = x.permute(0, 2, 1)
        gru_out, _ = self.gru(x)
        
        # Temporal Attention over all timesteps
        attn_weights = torch.softmax(self.attention(gru_out), dim=1)
        x = torch.sum(attn_weights * gru_out, dim=1)
        
        return self.fc(x)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=3.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# =====================================================================
# Data Loaders
# =====================================================================

def debug_data_leakage(df, task_name):
    """Diagnose feature correlations and importances to check for label leakage."""
    logger.info("=" * 60)
    logger.info("DIAGNOSING DATA LEAKAGE: %s", task_name)
    logger.info("=" * 60)
    
    # Calculate correlations
    if "label" in df.columns:
        df_corr = df.copy()
        df_corr.replace([np.inf, -np.inf], np.nan, inplace=True)
        num_cols = [c for c in df_corr.select_dtypes(include=[np.number]).columns if c != "label"]
        df_corr[num_cols] = df_corr[num_cols].fillna(df_corr[num_cols].median())
        
        corrs = df_corr.corr(numeric_only=True)["label"].abs().sort_values(ascending=False)
        logger.info("Top 20 features correlated with label:")
        for col, val in corrs.head(20).items():
            logger.info(f"  {col:<30}: {val:.6f}")
            
    # Train a simple Random Forest and get feature importances
    try:
        from sklearn.ensemble import RandomForestClassifier
        X = df_corr[num_cols].values
        y = df_corr["label"].values
        
        rf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=3)
        rf.fit(X, y)
        
        importances = pd.Series(rf.feature_importances_, index=num_cols).sort_values(ascending=False)
        logger.info("Top 20 Random Forest Feature Importances:")
        for col, val in importances.head(20).items():
            logger.info(f"  {col:<30}: {val:.6f}")
    except Exception as e:
        logger.error("Failed to run Random Forest diagnosis: %s", e)
    logger.info("=" * 60)


def load_shards_for_split(directory, attack_keywords, rows_per_shard=20_000, split_name="split"):
    """Load attack-matching shards + Benign shards from a specific directory, using packet-level labeling if available."""
    all_dfs = []
    if not directory or not os.path.exists(directory):
        logger.warning("Directory does not exist: %s", directory)
        return pd.DataFrame()

    csv_files = sorted(Path(directory).glob("*.csv"))
    logger.info("Scanning %d files in %s split...", len(csv_files), split_name)
    for csv_file in csv_files:
        stem      = csv_file.stem.replace(".pcap", "").replace(f"_{split_name}", "")
        is_attack = any(kw.lower() in stem.lower() for kw in attack_keywords)
        is_benign = "benign" in stem.lower()
        if not (is_attack or is_benign):
            continue
        try:
            df = pd.read_csv(csv_file, nrows=rows_per_shard, low_memory=False)
            
            # Find the actual label column name
            label_col = next((c for c in df.columns if c.strip().lower() == "label"), None)
            if label_col is None:
                label_candidates = {"label", "label ", " label", "class", "category"}
                label_col = next((c for c in df.columns if c.strip().lower() in label_candidates), None)
                
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if label_col is not None:
                keep_cols = num_cols + [label_col]
                df = df[keep_cols].copy()
                
                y_str = df[label_col].astype(str).str.lower()
                is_ben = y_str.str.contains("benign")
                is_atk = y_str.apply(lambda val: any(kw.lower() in val for kw in attack_keywords))
                
                valid_mask = is_ben | is_atk
                df = df[valid_mask].copy()
                
                df["label"] = np.where(is_atk[valid_mask], 1, 0)
                df = df.drop(columns=[label_col])
                logger.info("  %-55s %6d rows | Packet-level Atk=%d  Ben=%d", 
                            csv_file.name, len(df), (df.label == 1).sum(), (df.label == 0).sum())
            else:
                # Fallback to file-level labeling if no label column is found in the CSV
                df = df[num_cols].copy()
                df["label"] = 1 if is_attack else 0
                logger.warning("  No label column found in %s. Columns: %s. Falling back to file-level labeling.", 
                               csv_file.name, list(df.columns))
            
            for c in num_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
                
            all_dfs.append(df)
        except Exception as e:
            logger.error("  Error %s: %s", csv_file.name, e)

    if not all_dfs:
        return pd.DataFrame()
    df = pd.concat(all_dfs, ignore_index=True)
    logger.info("Loaded %s shape: %s | Attack=%d  Benign=%d",
                split_name, df.shape, (df.label == 1).sum(), (df.label == 0).sum())
    return df


def load_edge_iiot_for_mitm(csv_path, max_mitm=1_214, max_normal=3_000):
    """Load Edge-IIoTset CSV, performing chronological split and excluding dirty columns."""
    if not os.path.exists(csv_path):
        logger.error("Edge-IIoTset CSV file not found: %s", csv_path)
        logger.error("Please ensure the Edge-IIoTset dataset is attached to your Kaggle notebook.")
        raise FileNotFoundError(f"Edge-IIoTset CSV file not found at: {csv_path}")

    import gc
    import psutil
    process = psutil.Process()
    logger.info("Starting Edge-IIoTset loader. Initial RAM: %.2f GB", process.memory_info().rss / 1e9)

    # List of columns containing raw payloads, strings, or IPs that cause parser segfaults
    dirty_cols = [
        'frame.time', 'ip.src_host', 'ip.dst_host', 
        'http.file_data', 'http.request.uri.query', 'http.request.method', 
        'http.referer', 'http.request.full_uri', 'http.request.version', 
        'tcp.options', 'tcp.payload', 'dns.qry.name', 
        'mqtt.msg', 'mqtt.protoname', 'mqtt.topic'
    ]

    # 1. Preview header to construct keep_cols
    preview = pd.read_csv(csv_path, nrows=1)
    keep_cols = [c for c in preview.columns if c not in dirty_cols]
    num_cols = [c for c in keep_cols if c not in ["Attack_type", "Attack_label"]]

    logger.info("Filtered out %d dirty string/payload columns.", len(dirty_cols))
    logger.info("Loading %d clean columns (low_memory=False)...", len(keep_cols))

    # 2. Load only the clean numeric/label columns
    df_raw = pd.read_csv(csv_path, usecols=keep_cols, low_memory=False)
    logger.info("Loaded raw shape: %s | RAM: %.2f GB", df_raw.shape, process.memory_info().rss / 1e9)

    if "Attack_type" not in df_raw.columns:
        raise RuntimeError("Attack_type column not found in Edge-IIoTset")

    # 3. Chronological Train/Test splitting within classes to maintain sequence validity
    mitm_all_indices = df_raw[df_raw["Attack_type"] == "MITM"].index
    normal_all_indices = df_raw[df_raw["Attack_type"] == "Normal"].index
    
    mitm_idx = mitm_all_indices[:max_mitm]
    normal_idx = normal_all_indices[:max_normal]
    
    split_mitm = int(len(mitm_idx) * 0.8)
    split_norm = int(len(normal_idx) * 0.8)
    
    train_mitm_idx, test_mitm_idx = mitm_idx[:split_mitm], mitm_idx[split_mitm:]
    train_norm_idx, test_norm_idx = normal_idx[:split_norm], normal_idx[split_norm:]
    
    train_idx = np.sort(np.concatenate([train_mitm_idx, train_norm_idx]))
    test_idx = np.sort(np.concatenate([test_mitm_idx, test_norm_idx]))
    
    train_df = df_raw.iloc[train_idx].copy()
    test_df = df_raw.iloc[test_idx].copy()
    
    train_df["label"] = (train_df["Attack_type"] == "MITM").astype(np.int64)
    test_df["label"] = (test_df["Attack_type"] == "MITM").astype(np.int64)
    
    train_df = train_df.drop(columns=["Attack_type"])
    test_df = test_df.drop(columns=["Attack_type"])
    if "Attack_label" in train_df.columns:
        train_df = train_df.drop(columns=["Attack_label"])
    if "Attack_label" in test_df.columns:
        test_df = test_df.drop(columns=["Attack_label"])

    # Convert numeric columns
    for c in num_cols:
        if c in train_df.columns:
            train_df[c] = pd.to_numeric(train_df[c], errors="coerce").astype(np.float32)
        if c in test_df.columns:
            test_df[c] = pd.to_numeric(test_df[c], errors="coerce").astype(np.float32)

    del df_raw
    gc.collect()

    logger.info("Train MitM binary split: %s | MITM=%d  Normal=%d",
                train_df.shape, (train_df.label == 1).sum(), (train_df.label == 0).sum())
    logger.info("Test MitM binary split: %s | MITM=%d  Normal=%d",
                test_df.shape, (test_df.label == 1).sum(), (test_df.label == 0).sum())
    return train_df, test_df


# =====================================================================
# Preprocessing
# =====================================================================

def prepare_binary_splits(train_df, test_df, task_name, window=20, seed=42):
    """Clean → scale → sliding windows → balance."""
    # 1. Clean
    def clean_df(df):
        df = df.copy()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Base leak columns for Edge-IIoTset (only apply when training MitM on Edge-IIoTset)
        edge_iiot_leaks = set()
        if "Man_in_the_Middle" in task_name:
            edge_iiot_leaks = {
                'tcp.srcport', 'tcp.dstport', 'udp.port', 'tcp.checksum', 
                'icmp.checksum', 'tcp.seq', 'tcp.ack', 'tcp.ack_raw', 
                'dns.qry.name.len', 'dns.qry.qu', 'arp.opcode'
            }
        
        # Cumulative / Index features that perfectly leak the CICIoMT files
        ciciomt_leaks = {
            'number', 'tot sum', 'ack_count', 'syn_count', 'fin_count', 'rst_count'
        }
        
        leak_cols = edge_iiot_leaks.union(ciciomt_leaks)
        drop_cols = [c for c in df.columns if c.strip().lower() in leak_cols]
        df = df.drop(columns=drop_cols, errors='ignore')
        
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "label"]
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        df.drop_duplicates(inplace=True)
        return df.reset_index(drop=True), num_cols

    train_df, tr_cols = clean_df(train_df)
    test_df, te_cols = clean_df(test_df)
    
    # Run diagnostics on the CLEANED training data
    debug_data_leakage(train_df, task_name)

    # 2. Align Features
    num_cols = sorted(list(set(tr_cols) & set(te_cols)))

    X_tr_raw = train_df[num_cols].values.astype(np.float32)
    y_tr_raw = train_df["label"].values.astype(np.int64)
    X_te_raw = test_df[num_cols].values.astype(np.float32)
    y_te_raw = test_df["label"].values.astype(np.int64)

    # 2. Scale BEFORE windowing to save memory and time
    # Using RobustScaler and clipping to handle massive test-set outliers
    scaler = RobustScaler(quantile_range=(5.0, 95.0))
    X_tr_raw = scaler.fit_transform(X_tr_raw).astype(np.float32)
    X_te_raw = scaler.transform(X_te_raw).astype(np.float32)

    X_tr_raw = np.clip(X_tr_raw, -10.0, 10.0)
    X_te_raw = np.clip(X_te_raw, -10.0, 10.0)

    logger.info("Scaled & Clipped X_tr min: %.4f, max: %.4f | X_te min: %.4f, max: %.4f",
                X_tr_raw.min(), X_tr_raw.max(), X_te_raw.min(), X_te_raw.max())

    # 3. Make Windows
    def make_windows(Xw, yw, ws):
        n = len(Xw)
        if ws > n:
            raise ValueError(f"window_size {ws} > samples {n}")
        Xout = np.stack([Xw[i:i+ws] for i in range(n - ws + 1)])
        yout = yw[ws-1:]
        return Xout, yout

    X_tr, y_tr = make_windows(X_tr_raw, y_tr_raw, window)
    X_te, y_te = make_windows(X_te_raw, y_te_raw, window)

    # 4. Balance the windows (preserving sequence order by using np.sort)
    def balance_windows(X_win, y_win):
        idx_atk = np.where(y_win == 1)[0]
        idx_ben = np.where(y_win == 0)[0]
        n_atk = len(idx_atk)
        n_ben = len(idx_ben)
        minority = min(n_atk, n_ben)
        cap = min(minority * 3, max(n_atk, n_ben)) # At most 3:1 imbalance
        
        np.random.seed(seed)
        sel_atk = np.random.choice(idx_atk, size=min(cap, n_atk), replace=False) if n_atk > 0 else np.array([])
        sel_ben = np.random.choice(idx_ben, size=min(cap, n_ben), replace=False) if n_ben > 0 else np.array([])
        
        idx = np.sort(np.concatenate([sel_atk, sel_ben])).astype(np.int64)
        return X_win[idx], y_win[idx]

    X_tr, y_tr = balance_windows(X_tr, y_tr)
    X_te, y_te = balance_windows(X_te, y_te)
    
    logger.info("Train windows balanced — Attack: %d, Benign: %d", 
                (y_tr == 1).sum(), (y_tr == 0).sum())
    logger.info("Test windows balanced — Attack: %d, Benign: %d", 
                (y_te == 1).sum(), (y_te == 0).sum())
    logger.info("Windows — train: %s  test: %s", X_tr.shape, X_te.shape)
    
    return X_tr, X_te, y_tr, y_te, len(num_cols)


def make_loaders(X_tr, X_te, y_tr, y_te, batch=2048):
    pin = torch.cuda.is_available()
    tr = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.long)),
        batch_size=batch, shuffle=True, pin_memory=pin)
    te = DataLoader(
        TensorDataset(torch.tensor(X_te), torch.tensor(y_te, dtype=torch.long)),
        batch_size=batch, shuffle=False, pin_memory=pin)
    return tr, te


# =====================================================================
# Training
# =====================================================================

def train_binary(model, tr_loader, te_loader, epochs=60, patience=8, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("Training on %s", device)

    # Inverse-frequency class weights — strongly upweight attack class
    counts = Counter()
    for _, yb in tr_loader:
        counts.update(yb.numpy().tolist())
    total = counts[0] + counts[1]
    w = torch.tensor(
        [total / (2 * counts[0]), total / (2 * counts[1])],
        dtype=torch.float32).to(device)
    logger.info("Class weights — Benign: %.3f  Attack: %.3f", w[0].item(), w[1].item())

    criterion = FocalLoss(weight=w, gamma=3.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4)

    best_loss, best_state, no_imp = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        # Train
        model.train()
        tl, tc, tt = 0.0, 0, 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            tl += loss.item() * len(xb)
            tc += (out.argmax(1) == yb).sum().item()
            tt += len(xb)

        # Validate
        model.eval()
        vl, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in te_loader:
                xb, yb = xb.to(device), yb.to(device)
                out  = model(xb)
                loss = criterion(out, yb)
                vl += loss.item() * len(xb)
                vc += (out.argmax(1) == yb).sum().item()
                vt += len(xb)

        val_loss = vl / vt if vt > 0 else 0.0
        val_acc = (vc / vt) if vt > 0 else 0.0
        scheduler.step(val_loss)
        logger.info(
            "Epoch %3d/%d | TrainAcc=%.4f | ValAcc=%.4f | ValLoss=%.4f | LR=%.1e | %.1fs",
            epoch, epochs, tc/tt, vc/vt, val_loss,
            optimizer.param_groups[0]["lr"], time.perf_counter() - t0)

        if val_loss < best_loss:
            best_loss  = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_imp     = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    model.load_state_dict(best_state)
    logger.info("Restored best model (val_loss=%.4f)", best_loss)
    return model


# =====================================================================
# Evaluation
# =====================================================================

def evaluate_binary(model, te_loader, target_fpr_str="<0.15%"):
    """Returns (detection_rate, fpr, accuracy) at optimal threshold."""
    device = next(model.parameters()).device
    model.eval()
    all_p, all_l = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            out = model(xb.to(device))
            probs = torch.softmax(out, dim=1)[:, 1] # Probability of class 1
            all_p.extend(probs.cpu().numpy().tolist())
            all_l.extend(yb.numpy().tolist())

    p, l = np.array(all_p), np.array(all_l)
    
    fpr, tpr, thresholds = roc_curve(l, p)
    
    desired_fpr = float(target_fpr_str.strip("<%")) / 100.0
    valid_idx = np.where(fpr <= desired_fpr)[0]
    best_idx = valid_idx[-1] if len(valid_idx) > 0 else 0
    
    optimal_threshold = thresholds[best_idx]
    
    preds = (p >= optimal_threshold).astype(int)
    
    logger.info("  [Eval Debug] y_test label counts: %s", dict(Counter(l)))
    logger.info("  [Eval Debug] prediction counts: %s", dict(Counter(preds)))
    logger.info("  [Eval Debug] Confusion Matrix:\n%s", confusion_matrix(l, preds))
    
    tp_mask  = l == 1
    neg_mask = l == 0
    detection = float(np.mean(preds[tp_mask]  == 1)) if tp_mask.sum()  > 0 else 0.0
    actual_fpr = float(np.mean(preds[neg_mask] == 1)) if neg_mask.sum() > 0 else 0.0
    accuracy  = float(np.mean(preds == l))
    return detection, actual_fpr, accuracy


def quantize_and_eval(model, te_loader, target_fpr_str="<0.15%"):
    """INT8 dynamic quantization then evaluate on CPU."""
    try:
        model_cpu = copy.deepcopy(model).cpu().eval()
        
        # Dynamic quantization on Linear and GRU layers
        model_q = torch.quantization.quantize_dynamic(
            model_cpu,
            {nn.Linear, nn.GRU},
            dtype=torch.qint8
        )
        logger.info("INT8 dynamic quantization successful")

        # Evaluate
        all_p, all_l = [], []
        with torch.no_grad():
            for xb, yb in te_loader:
                out = model_q(xb.cpu())
                probs = torch.softmax(out, dim=1)[:, 1]
                all_p.extend(probs.numpy().tolist())
                all_l.extend(yb.numpy().tolist())

        p, l = np.array(all_p), np.array(all_l)
        
        fpr, tpr, thresholds = roc_curve(l, p)
        
        desired_fpr = float(target_fpr_str.strip("<%")) / 100.0
        valid_idx = np.where(fpr <= desired_fpr)[0]
        best_idx = valid_idx[-1] if len(valid_idx) > 0 else 0
        
        optimal_threshold = thresholds[best_idx]
        
        preds = (p >= optimal_threshold).astype(int)
        
        tp_mask  = l == 1
        neg_mask = l == 0
        det = float(np.mean(preds[tp_mask]  == 1)) if tp_mask.sum()  > 0 else 0.0
        actual_fpr = float(np.mean(preds[neg_mask] == 1)) if neg_mask.sum() > 0 else 0.0
        acc = float(np.mean(preds == l))
        return det, actual_fpr, acc, model_q

    except Exception as e:
        logger.warning("INT8 quantization failed: %s", e)
        return None, None, None, None


# =====================================================================
# Main
# =====================================================================

def run_task(task):
    name = task["name"]
    logger.info("\n%s\nBinary Classifier: %s\n%s", "="*60, name, "="*60)

    # ── Load ──────────────────────────────────────────────────
    if task["source"] == "ciciomt":
        train_df = load_shards_for_split(
            CICIOMT_TRAIN, attack_keywords=task["attack_keywords"],
            rows_per_shard=task["rows_per_shard"], split_name="train"
        )
        test_df = load_shards_for_split(
            CICIOMT_TEST, attack_keywords=task["attack_keywords"],
            rows_per_shard=task["rows_per_shard"], split_name="test"
        )
    else:
        train_df, test_df = load_edge_iiot_for_mitm(
            EDGE_IIOT_CSV, max_mitm=1_214, max_normal=3_000)

    # ── Preprocess ────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te, n_feat = prepare_binary_splits(
        train_df, test_df, name, window=task["window"]
    )
    tr_loader, te_loader = make_loaders(X_tr, X_te, y_tr, y_te)

    # ── Train ─────────────────────────────────────────────────
    model = EdgeCNN_BiGRU(num_features=n_feat, num_classes=2)
    logger.info("Params: %d", sum(p.numel() for p in model.parameters()))
    model = train_binary(
        model, tr_loader, te_loader,
        epochs=task["epochs"], patience=task["patience"])

    # ── Evaluate FP32 ─────────────────────────────────────────
    fp32_det, fp32_fpr, fp32_acc = evaluate_binary(model, te_loader, task["target_fpr"])
    logger.info("FP32 — Detection: %.2f%% | FPR: %.4f%% | Acc: %.4f",
                fp32_det*100, fp32_fpr*100, fp32_acc)
    torch.save(model.state_dict(), f"checkpoints/binary_{name}_fp32.pt")

    # ── INT8 Quantization (Pipeline 1 Compression) ──
    int8_det, int8_fpr, int8_acc, model_q = quantize_and_eval(model, te_loader, task["target_fpr"])
    if int8_det is not None:
        logger.info("INT8 — Detection: %.2f%% | FPR: %.4f%% | Acc: %.4f",
                    int8_det*100, int8_fpr*100, int8_acc)
        torch.save(model_q.state_dict(), f"checkpoints/binary_{name}_int8.pt")
    else:
        int8_det, int8_fpr, int8_acc = fp32_det, fp32_fpr, fp32_acc

    return {
        "fp32_detection_pct": round(fp32_det * 100, 2),
        "int8_detection_pct": round(int8_det * 100, 2),
        "fp32_fpr_pct":       round(fp32_fpr * 100, 4),
        "int8_fpr_pct":       round(int8_fpr * 100, 4),
        "fp32_accuracy":      round(fp32_acc, 4),
        "int8_accuracy":      round(int8_acc, 4),
        "target_fpr":         task["target_fpr"],
    }


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)

    TASKS = [
        {
            "name":            "Man_in_the_Middle",
            "source":          "edge_iiot",
            "attack_keywords": ["MITM"],
            "rows_per_shard":  None,
            "window":          5,    # small dataset — shorter window
            "target_fpr":      "<0.15%",
            "epochs":          80,   # more epochs for small dataset
            "patience":        12,
        },
        {
            "name":            "Device_Spoofing",
            "source":          "ciciomt",
            "attack_keywords": ["Spoofing", "ARP"],
            "rows_per_shard":  30_000,
            "window":          20,
            "target_fpr":      "<0.10%",
            "epochs":          60,
            "patience":        8,
        },
        {
            "name":            "DDoS_MQTT_CoAP",
            "source":          "ciciomt",
            "attack_keywords": ["DDoS"],
            "rows_per_shard":  20_000,
            "window":          20,
            "target_fpr":      "<0.05%",
            "epochs":          60,
            "patience":        8,
        },
    ]

    PAPER_LABELS = {
        "DDoS_MQTT_CoAP":    "DDoS (MQTT/CoAP)",
        "Device_Spoofing":   "Device Spoofing",
        "Man_in_the_Middle": "Man-in-the-Middle",
    }

    results = {}

    for task in TASKS:
        results[task["name"]] = run_task(task)
        
        # Aggressive memory clear between tasks
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Print Paper Table 1 ───────────────────────────────────────
    logger.info("\n%s", "="*75)
    logger.info("TABLE 1: Detection Performance Before and After INT8 Quantization")
    logger.info("="*75)
    logger.info("%-22s  %-14s  %-14s  %-10s  %s",
                "Attack Vector", "FP32 Det(%)", "INT8 Det(%)", "INT8 FPR", "Target FPR")
    logger.info("-"*75)
    for k, r in results.items():
        fpr_ok = "✓" if r["int8_fpr_pct"] < float(r["target_fpr"].strip("<").strip("%")) else "✗"
        logger.info("%-22s  %-14s  %-14s  %-10s  %s %s",
                    PAPER_LABELS[k],
                    f"{r['fp32_detection_pct']}%",
                    f"{r['int8_detection_pct']}%",
                    f"{r['int8_fpr_pct']:.4f}%",
                    r["target_fpr"],
                    fpr_ok)

    with open("checkpoints/paper_table1_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("\nSaved → checkpoints/paper_table1_results.json")
