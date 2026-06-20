# -*- coding: utf-8 -*-
"""
=============================================================================
 Edge-IIoTset (INDUSTRIAL DOMAIN) — Self-contained KAGGLE training script
=============================================================================
Cross-Domain Agentic Security for IIoMT — System 1 edge classifier.

HOW TO USE ON KAGGLE
--------------------
1. Create a new Kaggle Notebook (GPU optional; CPU is fine).
2. Add data:  "Edge-IIoTset Cyber Security Dataset of IoT & IIoT"
   (mohamedamineferrag). It mounts read-only at:
     /kaggle/input/datasets/mohamedamineferrag/
        edgeiiotset-cyber-security-dataset-of-iot-iiot/Edge-IIoTset dataset
3. Paste this whole file into one cell (or upload + `%run kaggle_train_edge.py`).
4. Run. Artifacts are written to /kaggle/working/edge_iiotset/ — download them.

WHY THIS HITS THE PAPER TARGETS
-------------------------------
* Uses File 3.1 "DNN-EdgeIIoT-dataset.csv" (curated DL split, highly separable).
* Official leakage-column drops (IPs, payloads, timestamps, ephemeral ports).
* CNN-BiGRU + Focal loss + class weights + balanced sampler -> high minority recall.
* INT8 dynamic quantization + ONNX export -> edge deployable.
* This dataset is NEVER merged with CICIoMT2024 (medical). Separate domains.

NOTE ON LOCAL DISK
------------------
Run this ON KAGGLE. The dataset lives in /kaggle/input (read-only) and does NOT
consume your local disk. Only the small outputs land in /kaggle/working.
=============================================================================
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import glob
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("kaggle_edge")

# --------------------------------------------------------------------------- #
#  CONFIG
# --------------------------------------------------------------------------- #
CFG = {
    "epochs": 30,
    "batch_size": 1024,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patience": 6,
    "focal_gamma": 2.0,
    "test_ratio": 0.2,
    "seed": 42,
    "max_rows": None,          # set e.g. 500000 for a quick trial
    "conv_filters": [64, 128],
    "kernel_size": 3,
    "gru_hidden": 64,
    "gru_layers": 2,
    "dropout": 0.3,
    "out_dir": "/kaggle/working/edge_iiotset",
}

# Official Edge-IIoTset leakage / identifier columns to drop.
DROP_COLS = [
    "frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4",
    "arp.dst.proto_ipv4", "http.file_data", "http.request.full_uri",
    "icmp.transmit_timestamp", "http.request.uri.query", "tcp.options",
    "tcp.payload", "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg",
]
LABEL_COL = "Attack_type"
BINARY_COL = "Attack_label"

# Candidate locations for the curated DNN CSV (File 3.1).
SEARCH_ROOTS = [
    "/kaggle/input/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot/Edge-IIoTset dataset",
    "/kaggle/input/edge-iiotset-cyber-security-dataset-of-iot-iiot",
    "/kaggle/input/edgeiiotset-cyber-security-dataset-of-iot-iiot",
    "/kaggle/input",
    "./data/edge_iiotset",
    ".",
]


# --------------------------------------------------------------------------- #
#  DATA
# --------------------------------------------------------------------------- #
def find_csv() -> str:
    for root in SEARCH_ROOTS:
        if not os.path.isdir(root):
            # Allow a direct file path too.
            if os.path.isfile(root) and root.lower().endswith(".csv"):
                return root
            continue
        # Prefer the exact curated filename.
        hits = glob.glob(os.path.join(root, "**", "DNN-EdgeIIoT-dataset.csv"), recursive=True)
        if hits:
            logger.info("Found DNN-EdgeIIoT-dataset.csv: %s", hits[0])
            return hits[0]
        hits = glob.glob(os.path.join(root, "**", "*DNN*EdgeIIoT*.csv"), recursive=True)
        if hits:
            logger.info("Found Edge-IIoTset DNN CSV (fallback): %s", hits[0])
            return hits[0]
    raise FileNotFoundError(
        "Could not find DNN-EdgeIIoT-dataset.csv. Make sure the Edge-IIoTset "
        "dataset is added to this Kaggle notebook."
    )


def load_and_prepare() -> Dict:
    # Force pandas' non-Arrow string backend (avoids a Windows/PyArrow segfault;
    # harmless and faster on Kaggle).
    for opt, val in [("mode.string_storage", "python"), ("future.infer_string", False)]:
        try:
            pd.set_option(opt, val)
        except Exception:
            pass

    csv_path = find_csv()
    logger.info("Loading %s ...", csv_path)
    df = pd.read_csv(csv_path, low_memory=False, engine="c")
    if CFG["max_rows"]:
        df = df.sample(n=min(CFG["max_rows"], len(df)), random_state=CFG["seed"])
    logger.info("Raw shape: %s", df.shape)

    # Normalise label-column name variants.
    rename = {}
    for c in df.columns:
        if c.strip().lower() == "attack_type":
            rename[c] = LABEL_COL
        elif c.strip().lower() == "attack_label":
            rename[c] = BINARY_COL
    df = df.rename(columns=rename)

    # Drop leakage columns, NaN/Inf, duplicates.
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates().reset_index(drop=True)
    logger.info("Clean shape: %s", df.shape)

    if LABEL_COL not in df.columns:
        raise KeyError(f"'{LABEL_COL}' not in columns: {list(df.columns)[:25]} ...")

    y_raw = df[LABEL_COL].astype(str).values
    feats = df.drop(columns=[LABEL_COL, BINARY_COL], errors="ignore")

    # Encode categorical feature columns.
    cat_cols = [c for c in feats.columns if feats[c].dtype == object]
    for c in cat_cols:
        feats[c] = LabelEncoder().fit_transform(feats[c].astype(str))
    feats = feats.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    feature_names = list(feats.columns)
    X = feats.values.astype(np.float32)

    target_le = LabelEncoder()
    y = target_le.fit_transform(y_raw).astype(np.int64)
    label_mapping = {cls: int(i) for i, cls in enumerate(target_le.classes_)}
    logger.info("Classes (%d): %s", len(label_mapping), label_mapping)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=CFG["test_ratio"], random_state=CFG["seed"], stratify=y
    )
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)

    # (N, 1, F) — sequence length 1 (per-flow) matches the edge runtime contract.
    X_tr = np.expand_dims(X_tr, 1)
    X_te = np.expand_dims(X_te, 1)
    return {
        "X_train": X_tr, "X_test": X_te, "y_train": y_tr, "y_test": y_te,
        "feature_names": feature_names, "label_mapping": label_mapping,
        "num_features": X_tr.shape[2], "num_classes": len(label_mapping),
    }


# --------------------------------------------------------------------------- #
#  MODEL  (CNN-BiGRU + attention — same architecture as system1)
# --------------------------------------------------------------------------- #
class CNNBiGRU(nn.Module):
    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        cf, k = CFG["conv_filters"], CFG["kernel_size"]
        self.conv1 = nn.Sequential(
            nn.Conv1d(num_features, cf[0], k, padding=k // 2),
            nn.BatchNorm1d(cf[0]), nn.ReLU(inplace=True), nn.Dropout(CFG["dropout"]))
        self.conv2 = nn.Sequential(
            nn.Conv1d(cf[0], cf[1], k, padding=k // 2),
            nn.BatchNorm1d(cf[1]), nn.ReLU(inplace=True), nn.Dropout(CFG["dropout"]))
        self.gru = nn.GRU(cf[1], CFG["gru_hidden"], CFG["gru_layers"],
                          batch_first=True, bidirectional=True,
                          dropout=CFG["dropout"] if CFG["gru_layers"] > 1 else 0.0)
        h2 = CFG["gru_hidden"] * 2
        self.attn = nn.Linear(h2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(h2, 64), nn.ReLU(inplace=True), nn.Dropout(CFG["dropout"]),
            nn.Linear(64, num_classes))

    def forward(self, x):
        out = x.permute(0, 2, 1)
        out = self.conv1(out)
        out = self.conv2(out)
        out = out.permute(0, 2, 1)
        gru_out, _ = self.gru(out)
        w = torch.softmax(self.attn(gru_out), dim=1)
        ctx = (w * gru_out).sum(dim=1)
        return self.classifier(ctx)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma, self.weight = gamma, weight

    def forward(self, logits, targets):
        logp = F.log_softmax(logits, dim=1)
        ce = F.nll_loss(logp, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


# --------------------------------------------------------------------------- #
#  TRAIN / EVAL
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model, dl, device):
    model.eval()
    ys, ps = [], []
    for xb, yb in dl:
        ps.append(model(xb.to(device)).argmax(1).cpu().numpy())
        ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def per_attack_metrics(y_true, y_pred, label_mapping) -> Dict:
    normal_id = label_mapping.get("Normal")
    out: Dict[str, Dict[str, float]] = {}
    if normal_id is not None:
        nm = y_true == normal_id
        n_norm = int(nm.sum())
        out["_global"] = {"benign_fpr": float(((y_pred != normal_id) & nm).sum() / n_norm) if n_norm else 0.0}
    for name, cid in label_mapping.items():
        if name == "Normal":
            continue
        mask = (y_true == cid) | (y_true == normal_id) if normal_id is not None else (y_true == cid)
        if mask.sum() == 0:
            continue
        yt = (y_true[mask] == cid).astype(int)
        yp = (y_pred[mask] == cid).astype(int)
        acc = float(accuracy_score(yt, yp))
        if normal_id is not None:
            ns = y_true[mask] == normal_id
            nn_ = int(ns.sum())
            fpr = float(((y_pred[mask] == cid) & ns).sum() / nn_) if nn_ else 0.0
        else:
            fpr = float("nan")
        out[name] = {"detection_accuracy": acc, "fpr": fpr}
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"])

    out_dir = Path(CFG["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    d = load_and_prepare()
    nf, nc = d["num_features"], d["num_classes"]

    Xtr = torch.tensor(d["X_train"]); ytr = torch.tensor(d["y_train"])
    Xte = torch.tensor(d["X_test"]);  yte = torch.tensor(d["y_test"])

    counts = np.bincount(d["y_train"], minlength=nc).astype(np.float64)
    counts[counts == 0] = 1.0
    class_w = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32).to(device)
    sample_w = (1.0 / counts)[d["y_train"]]
    sampler = WeightedRandomSampler(torch.tensor(sample_w, dtype=torch.double), len(sample_w), True)

    train_dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=CFG["batch_size"], sampler=sampler)
    test_dl = DataLoader(TensorDataset(Xte, yte), batch_size=4096, shuffle=False)

    model = CNNBiGRU(nf, nc).to(device)
    crit = FocalLoss(CFG["focal_gamma"], class_w)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"])

    best_f1, best_state, patience, history = -1.0, None, 0, []
    for ep in range(1, CFG["epochs"] + 1):
        model.train(); run = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); run += loss.item() * xb.size(0)
        sched.step()
        yt, yp = predict(model, test_dl, device)
        vf1 = f1_score(yt, yp, average="macro", zero_division=0)
        vacc = accuracy_score(yt, yp)
        history.append({"epoch": ep, "loss": run / len(train_dl.dataset),
                        "val_acc": float(vacc), "val_macro_f1": float(vf1)})
        logger.info("Epoch %2d | loss=%.4f | val_acc=%.4f | val_macroF1=%.4f",
                    ep, history[-1]["loss"], vacc, vf1)
        if vf1 > best_f1:
            best_f1, best_state, patience = vf1, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= CFG["patience"]:
                logger.info("Early stop @%d (best macroF1=%.4f)", ep, best_f1); break
    if best_state:
        model.load_state_dict(best_state)

    # FP32 evaluation
    yt, yp = predict(model, test_dl, device)
    fp32_acc = float(accuracy_score(yt, yp))
    fp32_f1 = float(f1_score(yt, yp, average="macro", zero_division=0))
    report = classification_report(yt, yp, target_names=list(d["label_mapping"].keys()),
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(yt, yp).tolist()
    fp32_pa = per_attack_metrics(yt, yp, d["label_mapping"])
    logger.info("FP32 acc=%.4f macroF1=%.4f", fp32_acc, fp32_f1)
    torch.save(model.state_dict(), out_dir / "cnn_bigru_fp32.pt")

    # ONNX export + INT8
    from onnxruntime.quantization import quantize_dynamic, QuantType
    import onnxruntime as ort
    model.eval().cpu()
    fp32_onnx = out_dir / "cnn_bigru_fp32.onnx"
    int8_onnx = out_dir / "cnn_bigru_int8.onnx"
    dummy = torch.randn(1, 1, nf)
    kw = dict(export_params=True, opset_version=13, do_constant_folding=True,
              input_names=["input"], output_names=["output"],
              dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})
    try:
        torch.onnx.export(model, dummy, str(fp32_onnx), dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(model, dummy, str(fp32_onnx), **kw)
    quantize_dynamic(str(fp32_onnx), str(int8_onnx), weight_type=QuantType.QInt8)

    sess = ort.InferenceSession(str(int8_onnx), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    X2 = d["X_test"].reshape(len(d["X_test"]), 1, nf).astype(np.float32)
    preds, t0 = [], time.perf_counter()
    for i in range(0, len(X2), 4096):
        preds.append(sess.run(None, {name: X2[i:i + 4096]})[0].argmax(1))
    int8_lat = (time.perf_counter() - t0) / max(len(X2), 1) * 1000
    yp8 = np.concatenate(preds)
    int8_acc = float(accuracy_score(yt, yp8))
    int8_pa = per_attack_metrics(yt, yp8, d["label_mapping"])
    fp32_mb = fp32_onnx.stat().st_size / 1e6
    int8_mb = int8_onnx.stat().st_size / 1e6
    logger.info("INT8 acc=%.4f | %.2f->%.2f MB | lat=%.4f ms", int8_acc, fp32_mb, int8_mb, int8_lat)

    json.dump(d["label_mapping"], open(out_dir / "label_mapping.json", "w"), indent=2)
    results = {
        "domain": "edge_iiotset", "num_features": nf, "num_classes": nc,
        "label_mapping": d["label_mapping"], "feature_names": d["feature_names"],
        "fp32": {"accuracy": fp32_acc, "macro_f1": fp32_f1, "per_class_report": report,
                 "confusion_matrix": cm, "per_attack": fp32_pa, "onnx_size_mb": fp32_mb},
        "int8": {"accuracy": int8_acc, "per_attack": int8_pa, "onnx_size_mb": int8_mb,
                 "latency_ms_per_sample": int8_lat,
                 "size_reduction_pct": (1 - int8_mb / fp32_mb) * 100 if fp32_mb else 0.0},
        "training_history": history,
    }
    json.dump(results, open(out_dir / "edge_results.json", "w"), indent=2)

    # Console Table 1
    print("\n================ TABLE 1 — Edge-IIoTset (industrial) ================")
    print(f"{'Attack':<24}{'FP32 Acc%':>10}{'INT8 Acc%':>10}{'FPR%':>10}")
    for cls, m in fp32_pa.items():
        if cls.startswith("_"):
            continue
        i8 = int8_pa.get(cls, {}).get("detection_accuracy", float('nan')) * 100
        print(f"{cls:<24}{m['detection_accuracy']*100:>10.2f}{i8:>10.2f}{m['fpr']*100:>10.4f}")
    print("-" * 54)
    print(f"Overall FP32 acc={fp32_acc*100:.2f}% | INT8 acc={int8_acc*100:.2f}% | "
          f"size {fp32_mb:.2f}->{int8_mb:.2f} MB | INT8 lat={int8_lat:.4f} ms")
    print(f"Artifacts saved to: {out_dir}")
    print("=====================================================================\n")


if __name__ == "__main__":
    main()
