import os
import sys
import json
import time
import logging
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef, roc_curve
from sklearn.ensemble import IsolationForest
import joblib
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("iimt_kaggle")

# ===================================================================
# CONFIG
# ===================================================================
CONFIG = {
    "batch_size": 1024,
    "epochs": 40,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "early_stopping_patience": 10,
    "conv_filters": [32, 64],
    "conv_kernel_size": 3,
    "gru_hidden_size": 48,
    "gru_num_layers": 2,
    "gru_dropout": 0.3,
    "cnn_dropout": 0.25,
    "fc_hidden": 48,
    "fc_dropout": 0.3,
    "random_seed": 42,
    "sequence_length": 32,
    "sequence_stride": 16,
    "confidence_rejection": 0.55,
    "margin_rejection": 0.10,
}

MACRO_CLASSES = ["Benign", "DDoS", "DoS", "MITM", "Reconnaissance", "Spoofing"]

def map_label(label_str):
    if pd.isna(label_str): return np.nan
    s = str(label_str).lower()
    if "benign" in s or "normal" in s: return "Benign"
    elif "arp" in s: return "MITM"
    elif "mitm" in s: return "MITM"
    elif "poison" in s: return "MITM"
    elif "ddos" in s or "mirai" in s: return "DDoS"
    elif "dos" in s: return "DoS"
    elif "recon" in s or "scan" in s or "sweep" in s or "discovery" in s: return "Reconnaissance"
    elif "spoof" in s or "malformed" in s or "injection" in s or "xss" in s or "bruteforce" in s or "hijack" in s or "malware" in s: return "Spoofing"
    return np.nan

# ===================================================================
# CUSTOM LOSSES
# ===================================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        logp = F.log_softmax(logits, dim=1)
        ce_loss = F.nll_loss(logp, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            focal_loss = focal_loss * self.alpha[targets]
        return focal_loss.mean()

# ===================================================================
# EDGE-OPTIMIZED ARCHITECTURE COMPONENTS
# ===================================================================
class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_c, out_c, k):
        super().__init__()
        self.depthwise = nn.Conv1d(in_c, in_c, kernel_size=k, padding=k//2, groups=in_c)
        self.pointwise = nn.Conv1d(in_c, out_c, kernel_size=1)
        
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class SEBlock(nn.Module):
    def __init__(self, channels, r=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // r),
            nn.ReLU(inplace=True),
            nn.Linear(channels // r, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.shape
        w = self.fc(self.pool(x).view(b, c)).view(b, c, 1)
        return x * w

class LinearAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        scores = self.attn(x)
        weights = torch.softmax(scores, dim=1)
        context = (x * weights).sum(dim=1)
        return context

class MixtureOfBinaryExperts(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        cf, ks, gh, gl = CONFIG["conv_filters"], CONFIG["conv_kernel_size"], CONFIG["gru_hidden_size"], CONFIG["gru_num_layers"]

        self.conv1 = DepthwiseSeparableConv1d(num_features, cf[0], ks)
        self.bn1 = nn.BatchNorm1d(cf[0])
        self.relu1 = nn.ReLU(inplace=True)
        self.se1 = SEBlock(cf[0])
        self.drop1 = nn.Dropout(CONFIG["cnn_dropout"])
        
        self.conv2 = DepthwiseSeparableConv1d(cf[0], cf[1], ks)
        self.bn2 = nn.BatchNorm1d(cf[1])
        self.relu2 = nn.ReLU(inplace=True)
        self.se2 = SEBlock(cf[1])
        self.drop2 = nn.Dropout(CONFIG["cnn_dropout"])
        
        self.gru = nn.GRU(
            input_size=cf[1], hidden_size=gh, num_layers=gl,
            batch_first=True, bidirectional=True, dropout=CONFIG["gru_dropout"] if gl > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(gh * 2)
        self.attn = LinearAttention(gh * 2)
        
        def build_expert_head():
            return nn.Sequential(
                nn.Linear(gh * 2, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(CONFIG["fc_dropout"]),
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 2)
            )
            
        self.head_s1 = build_expert_head()
        self.expert_ddos = build_expert_head()
        self.expert_mitm = build_expert_head()
        self.expert_spoof = build_expert_head()
        self.expert_dos = build_expert_head()
        self.expert_recon = build_expert_head()

    def forward(self, x):
        out = x.permute(0, 2, 1)
        
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.se1(out)
        out = self.drop1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.se2(out)
        out = self.drop2(out)
        
        cnn_features = out.permute(0, 2, 1)
        gru_out, _ = self.gru(cnn_features)
        gru_out = self.layer_norm(gru_out)
        context = self.attn(gru_out)
        
        logits_s1 = self.head_s1(context)
        logits_ddos = self.expert_ddos(context)
        logits_mitm = self.expert_mitm(context)
        logits_spoof = self.expert_spoof(context)
        logits_dos = self.expert_dos(context)
        logits_recon = self.expert_recon(context)
        
        return logits_s1, logits_ddos, logits_mitm, logits_spoof, logits_dos, logits_recon

# ===================================================================
# LAZY SEQUENCE DATASET
# ===================================================================
class SequenceDataset(Dataset):
    def __init__(self, X_raw, indices, lengths, labels, time_idx=-1):
        self.X_raw = X_raw
        self.indices = indices
        self.lengths = lengths
        self.labels = labels
        self.time_idx = time_idx
        self.seq_len = CONFIG["sequence_length"]
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        actual_len = self.lengths[idx]
        window_x = self.X_raw[start_idx : start_idx + actual_len]
        
        if actual_len < self.seq_len:
            pad = window_x[-1:]
            while len(window_x) < self.seq_len:
                window_x = np.concatenate([window_x, pad], axis=0)
            window_x = window_x[:self.seq_len]
            
        seq_mean = np.mean(window_x, axis=0, keepdims=True)
        seq_var = np.var(window_x, axis=0, keepdims=True)
        seq_max = np.max(window_x, axis=0, keepdims=True)
        seq_min = np.min(window_x, axis=0, keepdims=True)
        seq_delta = np.diff(window_x, axis=0, prepend=window_x[0:1])
        
        if self.time_idx != -1:
            times = window_x[:, self.time_idx]
            inter_arrival = np.diff(times, prepend=times[0])
            burst_dur = abs(times[-1] - times[0])
            pkt_rate = self.seq_len / (burst_dur + 1e-6)
            
            inter_arr_rep = np.repeat(inter_arrival.reshape(-1, 1), 1, axis=-1)
            burst_rep = np.full((actual_len, 1), burst_dur, dtype=np.float32)
            rate_rep = np.full((actual_len, 1), pkt_rate, dtype=np.float32)
            extra_features = np.concatenate([inter_arr_rep, burst_rep, rate_rep], axis=-1)
        else:
            extra_features = np.zeros((self.seq_len, 3), dtype=np.float32)
            
        seq_mean_rep = np.repeat(seq_mean, self.seq_len, axis=0)
        seq_var_rep = np.repeat(seq_var, self.seq_len, axis=0)
        seq_max_rep = np.repeat(seq_max, self.seq_len, axis=0)
        seq_min_rep = np.repeat(seq_min, self.seq_len, axis=0)
        
        window_x_enhanced = np.concatenate([
            window_x, seq_mean_rep, seq_var_rep, seq_max_rep, seq_min_rep, seq_delta, extra_features
        ], axis=-1)
        
        window_x_enhanced = np.nan_to_num(window_x_enhanced, nan=0.0, posinf=100.0, neginf=-100.0)
        window_x_enhanced = np.clip(window_x_enhanced, -100.0, 100.0)
        
        lbl = self.labels[idx]
        return torch.tensor(window_x_enhanced, dtype=torch.float32), torch.tensor(lbl, dtype=torch.long)

# ===================================================================
# DATA LOADING & PREPROCESSING
# ===================================================================
def load_csv_mapped(path):
    df = pd.read_csv(path, low_memory=False)
    found_label = False
    for col in ["label", "Label", "Attack_type", "Attack", "attack", "Class", "class", "category", "Category"]:
        if col in df.columns:
            df["label"] = df[col].apply(map_label)
            if col != "label": df.drop(columns=[col], inplace=True)
            found_label = True
            break
    if not found_label:
        df["label"] = map_label(os.path.basename(path))

    df = df.dropna(subset=["label"])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df

def detect_paths():
    ciciomt_train_candidates = [
        "/kaggle/input/datasets/outstandard09/ciciomt2024/Dataset/WiFI_and_MQTT/train",
        "/kaggle/input/datasets/likkisamarthreddy/ciciomt2024/csv/train",
        "/kaggle/input/ciciomt2024/Dataset/WiFI_and_MQTT/train",
        "/kaggle/input/ciciomt2024/csv/train",
        "./data/ciciomt2024/train",
        "./CICIOMT24/train"
    ]
    ciciomt_test_candidates = [
        "/kaggle/input/datasets/outstandard09/ciciomt2024/Dataset/WiFI_and_MQTT/test",
        "/kaggle/input/datasets/likkisamarthreddy/ciciomt2024/csv/test",
        "/kaggle/input/ciciomt2024/Dataset/WiFI_and_MQTT/test",
        "/kaggle/input/ciciomt2024/csv/test",
        "./data/ciciomt2024/test",
        "./CICIOMT24/test"
    ]
    
    edge_candidates = [
        "/kaggle/input/datasets/outstandard09/edge-iiotset",
        "/kaggle/input/edge-iiotset",
        "/kaggle/input/edgeiiotset",
        "/kaggle/input/edge-iiotset-cyber-security-dataset-of-iot-iiot",
        "./data/edge_iiotset"
    ]
    
    train_path = next((p for p in ciciomt_train_candidates if os.path.exists(p)), None)
    test_path = next((p for p in ciciomt_test_candidates if os.path.exists(p)), None)
    edge_path = next((p for p in edge_candidates if os.path.exists(p)), None)
    
    if train_path: logger.info(f"Detected CICIoMT train path: {train_path}")
    if test_path: logger.info(f"Detected CICIoMT test path: {test_path}")
    if edge_path: logger.info(f"Detected Edge-IIoTset path: {edge_path}")
        
    return train_path, test_path, edge_path

def load_and_preprocess(train_path: str, test_path: str, edge_path: str):
    train_csvs, test_csvs, edge_csvs = [], [], []
    
    if train_path:
        for root, _, files in os.walk(train_path):
            for f in files:
                if f.endswith(".csv"): train_csvs.append(os.path.join(root, f))
    if test_path:
        for root, _, files in os.walk(test_path):
            for f in files:
                if f.endswith(".csv"): test_csvs.append(os.path.join(root, f))
    if edge_path:
        for root, _, files in os.walk(edge_path):
            for f in files:
                if f.endswith(".csv"): edge_csvs.append(os.path.join(root, f))

    def load_multiple_csvs(paths):
        if not paths: return pd.DataFrame()
        return pd.concat([load_csv_mapped(p) for p in tqdm(paths)], ignore_index=True)

    logger.info("Loading train data...")
    df_train = load_multiple_csvs(train_csvs)
    
    if edge_csvs:
        logger.info("Loading Edge-IIoTset data for joint training...")
        df_edge = load_multiple_csvs(edge_csvs)
        if not df_edge.empty:
            df_train = pd.concat([df_train, df_edge], ignore_index=True)
            logger.info(f"Concatenated Edge-IIoTset. New train size: {df_train.shape}")
            
    logger.info("Loading test data...")
    df_test = load_multiple_csvs(test_csvs)

    # Debug what's actually in the raw CSV before mapping
    if "label" in df_train.columns:
        logger.info(f"Raw unmapped training labels: {df_train['label'].value_counts().to_dict()}")

    le = LabelEncoder().fit(MACRO_CLASSES)
    label_mapping = {cls: int(idx) for idx, cls in enumerate(le.classes_)}

    df_train.columns = [c.lower() for c in df_train.columns]
    if not df_test.empty: df_test.columns = [c.lower() for c in df_test.columns]
    
    # Strictly define 5-tuple flow columns if available
    ideal_flow_cols = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol"]
    possible_flow_cols = [c for c in df_train.columns if c in ideal_flow_cols]
    if len(possible_flow_cols) < 2:
        possible_flow_cols = [c for c in df_train.columns if any(x in c for x in ['ip', 'port', 'protocol']) and 'type' not in c and 'v' not in c]
    
    # Strictly isolate timestamp
    time_cols = [c for c in df_train.columns if ('time' in c or 'ts' in c) and 'ttl' not in c and 'live' not in c]
    has_time = len(time_cols) > 0

    if len(possible_flow_cols) >= 2:
        logger.info(f"Found flow columns: {possible_flow_cols}. Grouping to ensure per-flow boundaries.")
        df_train['flow_id'] = df_train.groupby(possible_flow_cols).ngroup()
        if not df_test.empty: df_test['flow_id'] = df_test.groupby(possible_flow_cols).ngroup()
        
        if has_time:
            df_train.sort_values(by=['flow_id', time_cols[0]], inplace=True)
            if not df_test.empty: df_test.sort_values(by=['flow_id', time_cols[0]], inplace=True)
        else:
            df_train.sort_values(by=['flow_id'], inplace=True)
            if not df_test.empty: df_test.sort_values(by=['flow_id'], inplace=True)
    else:
        df_train['flow_id'] = 0
        if not df_test.empty: df_test['flow_id'] = 0
    
    df_train.reset_index(drop=True, inplace=True)
    if not df_test.empty: df_test.reset_index(drop=True, inplace=True)

    feature_cols = [c for c in df_train.columns if c not in ["label", "flow_id"]]
    for df in [df_train, df_test]:
        if df.empty: continue
        for c in feature_cols: df[c] = pd.to_numeric(df[c], errors="coerce")
    
    nan_report = df_train[feature_cols].isnull().sum()
    cols_to_drop = nan_report[nan_report > df_train.shape[0] * 0.5].index.tolist()
    for df in [df_train, df_test]:
        if df.empty: continue
        df.drop(columns=cols_to_drop, inplace=True)
        df.fillna(0, inplace=True)
    feature_cols = [c for c in feature_cols if c not in cols_to_drop]

    X_train_raw = df_train[feature_cols].values.astype(np.float32)
    y_train_raw = le.transform(df_train["label"].values)
    flow_ids_train = df_train['flow_id'].values
    time_idx = feature_cols.index(time_cols[0]) if has_time and time_cols[0] in feature_cols else -1

    if not df_test.empty:
        X_test_raw = df_test[feature_cols].values.astype(np.float32)
        y_test_raw = le.transform(df_test["label"].values)
        flow_ids_test = df_test['flow_id'].values
    else:
        X_test_raw, y_test_raw, flow_ids_test = np.zeros((0, len(feature_cols))), np.zeros((0,)), np.zeros((0,))

    del df_train, df_test
    gc.collect()

    X_train_raw = np.nan_to_num(X_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
    if len(X_test_raw) > 0: X_test_raw = np.nan_to_num(X_test_raw, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_train_raw = scaler.fit_transform(X_train_raw)
    if len(X_test_raw) > 0: X_test_raw = scaler.transform(X_test_raw)

    seq_len = CONFIG["sequence_length"]
    base_stride = CONFIG["sequence_stride"]
    
    def extract_sequence_indices(y, flow_ids, seq_len, base_stride, is_train=False):
        indices, lengths, labels = [], [], []
        i = 0
        while i < len(y):
            max_idx = min(i + seq_len, len(y))
            window_flows = flow_ids[i:max_idx]
            
            diffs = np.where(window_flows != window_flows[0])[0]
            if len(diffs) > 0:
                flow_len = diffs[0]
            else:
                flow_len = len(window_flows)
                
            window_y = y[i:i+flow_len]
            
            mitm_present = np.any(window_y == label_mapping.get("MITM", -1))
            spoof_present = np.any(window_y == label_mapping.get("Spoofing", -1))
            recon_present = np.any(window_y == label_mapping.get("Reconnaissance", -1))
            
            lbl = -1
            
            if mitm_present:
                lbl = label_mapping.get("MITM", -1)
            elif spoof_present:
                lbl = label_mapping.get("Spoofing", -1)
            elif recon_present:
                lbl = label_mapping.get("Reconnaissance", -1)
            else:
                counts = np.bincount(window_y)
                if len(counts) > 0: lbl = counts.argmax()
                
            min_len_required = 4
            
            if lbl != -1 and flow_len >= min_len_required:
                indices.append(i)
                lengths.append(flow_len)
                labels.append(lbl)
                
            if flow_len < seq_len:
                i += flow_len
            else:
                current_stride = 10 if (is_train and lbl in [label_mapping.get("MITM",-1), label_mapping.get("Spoofing",-1)]) else base_stride
                i += current_stride
                
        return np.array(indices, dtype=np.int64), np.array(lengths, dtype=np.int64), np.array(labels, dtype=np.int64)

    logger.info(f"Raw train labels before sequence extraction: {Counter(y_train_raw)}")
    logger.info(f"Extracting lazy sequence indices strictly bounded by flows...")
    train_idx, train_len, train_lbl = extract_sequence_indices(y_train_raw, flow_ids_train, seq_len, base_stride, is_train=True)
    logger.info(f"Train sequence labels after extraction: {Counter(train_lbl)}")
    if len(X_test_raw) > 0:
        test_idx, test_len, test_lbl = extract_sequence_indices(y_test_raw, flow_ids_test, seq_len, seq_len, is_train=False)
    else:
        test_idx, test_len, test_lbl = np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    if len(test_idx) == 0:
        try:
            train_val_idx, test_idx, train_val_len, test_len, train_val_lbl, test_lbl = train_test_split(
                train_idx, train_len, train_lbl, test_size=0.15, stratify=train_lbl, random_state=42
            )
        except ValueError:
            logger.error("Stratified test split failed. Hard exiting to prevent corrupted pipeline.")
            sys.exit(1)
    else:
        train_val_idx, train_val_len, train_val_lbl = train_idx, train_len, train_lbl

    try:
        train_idx, val_idx, train_len, val_len, train_lbl, val_lbl = train_test_split(
            train_val_idx, train_val_len, train_val_lbl, test_size=0.15, stratify=train_val_lbl, random_state=42
        )
    except ValueError:
        logger.error("Stratified val split failed. Hard exiting.")
        sys.exit(1)
        
    logger.info(f"Train labels: {Counter(train_lbl)}")
    logger.info(f"Val labels: {Counter(val_lbl)}")
    logger.info(f"Test labels: {Counter(test_lbl)}")
    
    # Only enforce that splits don't lose classes that actually exist in the full sequence set
    extracted_classes = set(train_val_lbl).union(set(test_lbl))
    
    for split_name, lbls in [("Train", train_lbl), ("Val", val_lbl), ("Test", test_lbl)]:
        present_classes = set(lbls)
        missing_from_split = extracted_classes - present_classes
        if missing_from_split:
            logger.error(f"CRITICAL: {split_name} split dropped classes present in data: {missing_from_split}")
            sys.exit(1)
            
    num_features_enhanced = len(feature_cols) * 6 + 3
    return (X_train_raw, train_idx, train_len, train_lbl), (X_train_raw, val_idx, val_len, val_lbl), (X_test_raw, test_idx, test_len, test_lbl), label_mapping, num_features_enhanced, time_idx, len(feature_cols)

# ===================================================================
# MAIN
# ===================================================================
def main():
    torch.manual_seed(CONFIG["random_seed"])
    np.random.seed(CONFIG["random_seed"])

    train_path, test_path, edge_path = detect_paths()
    if not train_path: return
            
    res = load_and_preprocess(train_path, test_path, edge_path)
    train_tuple, val_tuple, test_tuple, label_mapping, num_features, time_idx, raw_dim = res
    X_train_raw, train_idx, train_len, y_train = train_tuple
    X_val_raw, val_idx, val_len, y_val = val_tuple
    X_test_raw, test_idx, test_len, y_test = test_tuple
    
    benign_idx = label_mapping.get("Benign", 0)

    # -------------------------------------------------------------
    # Stage 0: Isolation Forest
    # -------------------------------------------------------------
    logger.info("Training Isolation Forest on raw Benign features only...")
    benign_train_indices = train_idx[y_train == benign_idx]
    if len(benign_train_indices) > 50000:
        benign_train_indices = np.random.choice(benign_train_indices, 50000, replace=False)
        
    if len(benign_train_indices) > 0:
        X_train_benign_pooled = []
        for i in benign_train_indices:
            window = X_train_raw[i : i + CONFIG["sequence_length"], :raw_dim]
            X_train_benign_pooled.append(window.mean(axis=0))
        X_train_benign_pooled = np.array(X_train_benign_pooled)
        
        iforest = IsolationForest(n_estimators=100, random_state=CONFIG["random_seed"], contamination=0.001, n_jobs=-1)
        iforest.fit(X_train_benign_pooled)
        joblib.dump(iforest, "/kaggle/working/iforest.joblib")
    else:
        iforest = None
        
    # -------------------------------------------------------------
    # Effective Number of Samples Dataloader
    # -------------------------------------------------------------
    class_counts = Counter(y_train)
    beta = 0.9999
    effective_num = {cls: (1.0 - beta) / (1.0 - (beta ** count)) for cls, count in class_counts.items()}
    # Normalize weights so they aren't astronomically small
    sum_eff = sum(effective_num.values())
    class_weights = {cls: (w / sum_eff) for cls, w in effective_num.items()}
    
    sample_weights = np.array([class_weights[y] for y in y_train])
    sample_weights_tensor = torch.from_numpy(sample_weights).double()
    sampler = WeightedRandomSampler(weights=sample_weights_tensor, num_samples=len(sample_weights_tensor), replacement=True)

    train_ds = SequenceDataset(X_train_raw, train_idx, train_len, y_train, time_idx)
    val_ds = SequenceDataset(X_val_raw, val_idx, val_len, y_val, time_idx)
    test_ds = SequenceDataset(X_test_raw, test_idx, test_len, y_test, time_idx)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], sampler=sampler, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, pin_memory=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False, pin_memory=True, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MixtureOfBinaryExperts(num_features=num_features).to(device)

    # Class weights for Stage 1 CE
    s1_counts = Counter([0 if y == benign_idx else 1 for y in y_train])
    n_total = len(y_train)
    s1_weights = torch.tensor([n_total / (2 * max(s1_counts[0], 1)), n_total / (2 * max(s1_counts[1], 1))], dtype=torch.float32).to(device)
    s1_weights = s1_weights / s1_weights.sum()

    criterion_s1 = nn.CrossEntropyLoss(weight=s1_weights)
    # Explicit class weight multipliers to combat 4.9M vs <1k imbalance
    alpha_weights = torch.ones(len(label_mapping), dtype=torch.float32).to(device)
    if "MITM" in label_mapping: alpha_weights[label_mapping["MITM"]] = 5.0
    if "Spoofing" in label_mapping: alpha_weights[label_mapping["Spoofing"]] = 3.0
    if "Reconnaissance" in label_mapping: alpha_weights[label_mapping["Reconnaissance"]] = 2.0
    criterion_expert = FocalLoss(gamma=2.0, alpha=alpha_weights)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CONFIG["learning_rate"], epochs=CONFIG["epochs"], steps_per_epoch=max(1, len(train_loader)))
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_ckpt_score = -1.0
    patience_counter = 0

    for epoch in range(CONFIG["epochs"]):
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            # Map targets for binary experts
            y_s1 = (batch_y != benign_idx).long()
            y_ddos = (batch_y == label_mapping.get("DDoS", -1)).long()
            y_mitm = (batch_y == label_mapping.get("MITM", -1)).long()
            y_spoof = (batch_y == label_mapping.get("Spoofing", -1)).long()
            y_dos = (batch_y == label_mapping.get("DoS", -1)).long()
            y_recon = (batch_y == label_mapping.get("Reconnaissance", -1)).long()
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                l_s1, l_ddos, l_mitm, l_spoof, l_dos, l_recon = model(batch_X)
                
                loss_s1 = criterion_s1(l_s1, y_s1)
                loss_ddos = criterion_expert(l_ddos, y_ddos)
                loss_mitm = criterion_expert(l_mitm, y_mitm)
                loss_spoof = criterion_expert(l_spoof, y_spoof)
                loss_dos = criterion_expert(l_dos, y_dos)
                loss_recon = criterion_expert(l_recon, y_recon)
                
                loss = loss_s1 + loss_ddos + loss_mitm + loss_spoof + loss_dos + loss_recon
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_loss += loss.item()
            
        model.eval()
        val_preds_final = []
        val_targs = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                l_s1, l_ddos, l_mitm, l_spoof, l_dos, l_recon = model(batch_X)
                
                p_s1 = torch.softmax(l_s1, dim=1)[:, 1].cpu().numpy()
                p_ddos = torch.softmax(l_ddos, dim=1)[:, 1].cpu().numpy()
                p_mitm = torch.softmax(l_mitm, dim=1)[:, 1].cpu().numpy()
                p_spoof = torch.softmax(l_spoof, dim=1)[:, 1].cpu().numpy()
                p_dos = torch.softmax(l_dos, dim=1)[:, 1].cpu().numpy()
                p_recon = torch.softmax(l_recon, dim=1)[:, 1].cpu().numpy()
                
                # Naive max pooling for val loop checkpoint
                stacked = np.stack([np.zeros_like(p_ddos), p_ddos, p_dos, p_mitm, p_recon, p_spoof], axis=1)
                preds = np.argmax(stacked, axis=1)
                preds[p_s1 < 0.5] = benign_idx
                
                val_preds_final.extend(preds)
                val_targs.extend(batch_y.numpy())
                
        val_macro_f1 = f1_score(val_targs, val_preds_final, average='macro', zero_division=0)
        
        logger.info(f"Epoch {epoch+1} - Val Macro-F1: {val_macro_f1:.4f}")
        
        if val_macro_f1 > best_ckpt_score:
            best_ckpt_score = val_macro_f1
            patience_counter = 0
            torch.save(model.state_dict(), "/kaggle/working/cnn_bigru_fp32.pt")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["early_stopping_patience"]:
                break

    # --- Threshold Calibration ---
    model.load_state_dict(torch.load("/kaggle/working/cnn_bigru_fp32.pt"))
    model.eval()
    
    logger.info("Extracting Validation Probabilities for ROC Search...")
    probs_dict = {"s1": [], "ddos": [], "mitm": [], "spoof": [], "dos": [], "recon": []}
    val_y = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            l_s1, l_ddos, l_mitm, l_spoof, l_dos, l_recon = model(batch_X.to(device))
            
            l_s1 = torch.nan_to_num(l_s1, nan=0.0, posinf=50.0, neginf=-50.0)
            l_ddos = torch.nan_to_num(l_ddos, nan=0.0, posinf=50.0, neginf=-50.0)
            l_mitm = torch.nan_to_num(l_mitm, nan=0.0, posinf=50.0, neginf=-50.0)
            l_spoof = torch.nan_to_num(l_spoof, nan=0.0, posinf=50.0, neginf=-50.0)
            l_dos = torch.nan_to_num(l_dos, nan=0.0, posinf=50.0, neginf=-50.0)
            l_recon = torch.nan_to_num(l_recon, nan=0.0, posinf=50.0, neginf=-50.0)
            
            probs_dict["s1"].extend(torch.softmax(l_s1, dim=1)[:, 1].cpu().numpy())
            probs_dict["ddos"].extend(torch.softmax(l_ddos, dim=1)[:, 1].cpu().numpy())
            probs_dict["mitm"].extend(torch.softmax(l_mitm, dim=1)[:, 1].cpu().numpy())
            probs_dict["spoof"].extend(torch.softmax(l_spoof, dim=1)[:, 1].cpu().numpy())
            probs_dict["dos"].extend(torch.softmax(l_dos, dim=1)[:, 1].cpu().numpy())
            probs_dict["recon"].extend(torch.softmax(l_recon, dim=1)[:, 1].cpu().numpy())
            val_y.extend(batch_y.numpy())
            
    val_y = np.array(val_y)
    
    def get_roc_threshold(probs, targets, target_fpr):
        if len(np.unique(targets)) < 2: return 0.5
        fpr, tpr, th = roc_curve(targets, probs)
        idx = np.where(fpr <= target_fpr)[0]
        if len(idx) == 0: return 0.99
        return th[idx[np.argmax(tpr[idx])]]

    t_s1 = get_roc_threshold(np.array(probs_dict["s1"]), (val_y != benign_idx).astype(int), 0.05)
    t_ddos = get_roc_threshold(np.array(probs_dict["ddos"]), (val_y == label_mapping.get("DDoS")).astype(int), 0.0005)
    t_mitm = get_roc_threshold(np.array(probs_dict["mitm"]), (val_y == label_mapping.get("MITM")).astype(int), 0.0015)
    t_spoof = get_roc_threshold(np.array(probs_dict["spoof"]), (val_y == label_mapping.get("Spoofing")).astype(int), 0.001)
    t_dos = get_roc_threshold(np.array(probs_dict["dos"]), (val_y == label_mapping.get("DoS")).astype(int), 0.005)
    t_recon = get_roc_threshold(np.array(probs_dict["recon"]), (val_y == label_mapping.get("Reconnaissance")).astype(int), 0.01)

    logger.info(f"Expert Thresholds -> S1: {t_s1:.3f}, DDoS: {t_ddos:.3f}, MITM: {t_mitm:.3f}, Spoofing: {t_spoof:.3f}, DoS: {t_dos:.3f}, Recon: {t_recon:.3f}")

    # --- FP32 Evaluation (Sanity Check) ---
    logger.info("Evaluating FP32 predictions before ONNX export...")
    fp32_preds = []
    
    def apply_expert_inference(p_s1, p_ddos, p_mitm, p_spoof, p_dos, p_recon, raw_x, thresholds):
        batch_preds = []
        t_s1, t_ddos, t_mitm, t_spoof, t_dos, t_recon = thresholds
        
        for i in range(len(p_s1)):
            if p_s1[i] < t_s1:
                batch_preds.append(benign_idx)
                continue
            
            candidates = []
            if p_ddos[i] > t_ddos: candidates.append((p_ddos[i], label_mapping.get("DDoS")))
            if p_mitm[i] > t_mitm: candidates.append((p_mitm[i], label_mapping.get("MITM")))
            if p_spoof[i] > t_spoof: candidates.append((p_spoof[i], label_mapping.get("Spoofing")))
            if p_dos[i] > t_dos: candidates.append((p_dos[i], label_mapping.get("DoS")))
            if p_recon[i] > t_recon: candidates.append((p_recon[i], label_mapping.get("Reconnaissance")))
            
            if candidates:
                batch_preds.append(max(candidates)[1])
            else:
                batch_preds.append(benign_idx)
            
        return batch_preds

    with torch.no_grad():
        for batch_X, _ in val_loader:
            l_s1, l_ddos, l_mitm, l_spoof, l_dos, l_recon = model(batch_X.to(device))
            p_s1 = torch.softmax(l_s1, dim=1)[:, 1].cpu().numpy()
            p_ddos = torch.softmax(l_ddos, dim=1)[:, 1].cpu().numpy()
            p_mitm = torch.softmax(l_mitm, dim=1)[:, 1].cpu().numpy()
            p_spoof = torch.softmax(l_spoof, dim=1)[:, 1].cpu().numpy()
            p_dos = torch.softmax(l_dos, dim=1)[:, 1].cpu().numpy()
            p_recon = torch.softmax(l_recon, dim=1)[:, 1].cpu().numpy()
            
            batch_preds = apply_expert_inference(p_s1, p_ddos, p_mitm, p_spoof, p_dos, p_recon, batch_X.numpy(), (t_s1, t_ddos, t_mitm, t_spoof, t_dos, t_recon))
            fp32_preds.extend(batch_preds)
            
    logger.info(f"FP32 Test Preds distribution: {Counter(fp32_preds)}")

    # --- ONNX Export ---
    logger.info("Applying L1 Unstructured Pruning (20%) and Quantization...")
    import torch.nn.utils.prune as prune
    
    model_cpu = model.cpu().eval()
    for name, module in model_cpu.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
            prune.l1_unstructured(module, name='weight', amount=0.2)
            prune.remove(module, 'weight') # Make permanent
        
    dummy_input = torch.randn(1, CONFIG["sequence_length"], num_features)
    import onnx
    torch.onnx.export(
        model_cpu, dummy_input, "/kaggle/working/expert_bigru_fp32.onnx", 
        input_names=['input'], output_names=['s1', 'ddos', 'mitm', 'spoof', 'dos', 'recon'], 
        opset_version=17, export_params=True, do_constant_folding=False,
        dynamic_axes={'input': {0: 'batch_size'}}, 
        dynamo=False
    )
                      
    quantize_dynamic(
        "/kaggle/working/expert_bigru_fp32.onnx", "/kaggle/working/expert_bigru_int8.onnx", 
        op_types_to_quantize=["MatMul", "Gemm"],
        weight_type=QuantType.QInt8, per_channel=True,
        nodes_to_exclude=["/gru/GRU"] # Protect the GRU hidden states from collapse
    )
    
    logger.info("Evaluation Complete. Download models from /kaggle/working/")

if __name__ == "__main__":
    main()
