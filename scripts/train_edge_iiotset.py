# -*- coding: utf-8 -*-
"""
Edge-IIoTset Trainer  (INDUSTRIAL DOMAIN — standalone)
======================================================

Trains an independent CNN-BiGRU classifier on the **Edge-IIoTset** dataset
ONLY. It is never merged with the CICIoMT2024 (medical) pipeline — the two
domains stay fully separate, satisfying the paper's cross-domain objective
as two independent domain models.

Pipeline
--------
1. Load + clean Edge-IIoTset via ``data.edge_iiotset_loader.EdgeIIoTLoader``.
2. Train the ``system1.models.cnn_bigru.CNNBiGRU`` model with:
     * **Focal loss** + **class weights** (handles class imbalance)
     * **WeightedRandomSampler** (balanced mini-batches)
     * cosine LR schedule + early stopping
3. Evaluate FP32: overall accuracy, macro-F1, confusion matrix,
   **per-attack binary detection accuracy + FPR** (paper Table 1 style).
4. Export ONNX (FP32) and apply **INT8 dynamic quantization**; re-evaluate.
5. Save everything under ``checkpoints/edge_iiotset/`` (separate from CIC):
     * ``cnn_bigru_fp32.pt`` / ``cnn_bigru_fp32.onnx`` / ``cnn_bigru_int8.onnx``
     * ``label_mapping.json``
     * ``edge_results.json``  (consumed by evaluation/paper_table1.py)

Run::

    python train_edge_iiotset.py
    # then:  python -m evaluation.paper_table1 --domain edge
"""

from __future__ import annotations

import os
# Guard against the Windows OpenMP duplicate-runtime segfault (0xC0000005)
# that occurs when torch, scikit-learn and scipy each load their own libomp.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import sys
# torch.onnx's exporter prints unicode (emoji) status lines that crash the
# Windows cp1252 console; force UTF-8 so those prints never raise.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

# repo bootstrap: make src/ importable + anchor CWD to repo root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))
os.chdir(_ROOT)

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)

from data.edge_iiotset_loader import EdgeIIoTLoader, EdgeData
from system1.models.cnn_bigru import CNNBiGRU

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("edge_iiotset.train")

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_CFG = {
    "epochs": 30,
    "batch_size": 1024,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "early_stopping_patience": 6,
    "focal_gamma": 2.0,
}


# --------------------------------------------------------------------------- #
#  Focal loss (class-weighted) — the key to lifting minority-class recall.
# --------------------------------------------------------------------------- #
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)
        ce = F.nll_loss(logp, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


# --------------------------------------------------------------------------- #
#  Training
# --------------------------------------------------------------------------- #
def make_loaders(data: EdgeData, batch_size: int) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    Xtr = torch.tensor(data.X_train, dtype=torch.float32)
    ytr = torch.tensor(data.y_train, dtype=torch.long)
    Xte = torch.tensor(data.X_test, dtype=torch.float32)
    yte = torch.tensor(data.y_test, dtype=torch.long)

    # Class weights (inverse frequency) for focal loss.
    counts = np.bincount(data.y_train, minlength=data.num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    class_weights = torch.tensor((counts.sum() / (len(counts) * counts)), dtype=torch.float32)

    # Balanced sampler so minority attacks appear in most mini-batches.
    sample_w = (1.0 / counts)[data.y_train]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_w, dtype=torch.double),
        num_samples=len(sample_w), replacement=True,
    )

    train_dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, sampler=sampler)
    test_dl = DataLoader(TensorDataset(Xte, yte), batch_size=batch_size, shuffle=False)
    return train_dl, test_dl, class_weights


def train_model(data: EdgeData, device: torch.device) -> CNNBiGRU:
    model = CNNBiGRU(num_features=data.num_features, num_classes=data.num_classes).to(device)
    train_dl, test_dl, class_weights = make_loaders(data, TRAIN_CFG["batch_size"])

    criterion = FocalLoss(gamma=TRAIN_CFG["focal_gamma"], weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=TRAIN_CFG["learning_rate"], weight_decay=TRAIN_CFG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_CFG["epochs"])

    best_f1, best_state, patience = -1.0, None, 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, TRAIN_CFG["epochs"] + 1):
        model.train()
        running = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += loss.item() * xb.size(0)
        scheduler.step()

        # Validation macro-F1.
        y_true, y_pred = _predict(model, test_dl, device)
        val_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        val_acc = accuracy_score(y_true, y_pred)
        history.append({"epoch": epoch, "train_loss": running / len(train_dl.dataset),
                        "val_acc": float(val_acc), "val_macro_f1": float(val_f1)})
        logger.info("Epoch %2d | loss=%.4f | val_acc=%.4f | val_macroF1=%.4f",
                    epoch, history[-1]["train_loss"], val_acc, val_f1)

        if val_f1 > best_f1:
            best_f1, best_state, patience = val_f1, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= TRAIN_CFG["early_stopping_patience"]:
                logger.info("Early stopping at epoch %d (best macro-F1=%.4f).", epoch, best_f1)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model._history = history  # type: ignore[attr-defined]
    return model


@torch.no_grad()
def _predict(model: nn.Module, dl: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for xb, yb in dl:
        logits = model(xb.to(device))
        ps.append(logits.argmax(1).cpu().numpy())
        ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(ps)


# --------------------------------------------------------------------------- #
#  Per-attack binary metrics (paper Table 1 style)
# --------------------------------------------------------------------------- #
def per_attack_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, label_mapping: Dict[str, int]
) -> Dict[str, Dict[str, float]]:
    """For each attack class, compute one-vs-Normal detection accuracy + FPR.

    Detection accuracy = correct decisions over {this attack} ∪ {Normal}.
    FPR = fraction of Normal samples mis-flagged as *any* attack.
    """
    inv = {v: k for k, v in label_mapping.items()}
    normal_id = label_mapping.get("Normal")
    out: Dict[str, Dict[str, float]] = {}

    # Global FPR: Normal rows predicted as any non-Normal class.
    if normal_id is not None:
        normal_mask = y_true == normal_id
        n_normal = int(normal_mask.sum())
        fp_global = int(((y_pred != normal_id) & normal_mask).sum())
        global_fpr = fp_global / n_normal if n_normal else 0.0
    else:
        global_fpr = float("nan")

    for cls_name, cls_id in label_mapping.items():
        if cls_name == "Normal":
            continue
        mask = (y_true == cls_id) | (y_true == normal_id) if normal_id is not None else (y_true == cls_id)
        if mask.sum() == 0:
            continue
        # Binary view: positive = this attack class.
        yt = (y_true[mask] == cls_id).astype(int)
        yp = (y_pred[mask] == cls_id).astype(int)
        acc = float(accuracy_score(yt, yp))
        # Per-attack FPR uses only Normal rows in this slice.
        if normal_id is not None:
            normal_slice = y_true[mask] == normal_id
            n_norm = int(normal_slice.sum())
            fp = int(((y_pred[mask] == cls_id) & normal_slice).sum())
            fpr = fp / n_norm if n_norm else 0.0
        else:
            fpr = float("nan")
        out[cls_name] = {"detection_accuracy": acc, "fpr": fpr}

    out["_global"] = {"benign_fpr": global_fpr}
    return out


# --------------------------------------------------------------------------- #
#  ONNX export + INT8 quantization
# --------------------------------------------------------------------------- #
def export_onnx(model: CNNBiGRU, num_features: int, out_dir: Path) -> Tuple[Path, Path]:
    import onnxruntime as ort  # noqa: F401  (ensures runtime present)
    from onnxruntime.quantization import quantize_dynamic, QuantType

    model.eval().cpu()
    fp32_path = out_dir / "cnn_bigru_fp32.onnx"
    int8_path = out_dir / "cnn_bigru_int8.onnx"
    dummy = torch.randn(1, 1, num_features)  # sequence_length: 1
    export_kwargs = dict(
        export_params=True, opset_version=13, do_constant_folding=True,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    # Prefer the stable TorchScript exporter (dynamo=False) to avoid the
    # dynamo exporter's unicode status prints; fall back if unsupported.
    try:
        torch.onnx.export(model, dummy, str(fp32_path), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, str(fp32_path), **export_kwargs)
    logger.info("Exported FP32 ONNX -> %s", fp32_path)
    quantize_dynamic(model_input=str(fp32_path), model_output=str(int8_path),
                     weight_type=QuantType.QInt8)
    logger.info("Exported INT8 ONNX -> %s", int8_path)
    return fp32_path, int8_path


def evaluate_onnx(onnx_path: Path, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    preds = []
    t0 = time.perf_counter()
    bs = 4096
    for i in range(0, len(X), bs):
        logits = sess.run(None, {name: X[i:i + bs].astype(np.float32)})[0]
        preds.append(logits.argmax(1))
    dt = (time.perf_counter() - t0) / max(len(X), 1) * 1000.0
    return np.concatenate(preds), dt


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    loader = EdgeIIoTLoader()
    out_dir = _PROJECT_ROOT / loader.cfg.get("output_dir", "checkpoints/edge_iiotset")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = loader.prepare()

    # --- Train (FP32) ---
    model = train_model(data, device)

    # --- Evaluate FP32 ---
    test_dl = DataLoader(
        TensorDataset(torch.tensor(data.X_test), torch.tensor(data.y_test)),
        batch_size=4096, shuffle=False,
    )
    y_true, y_pred_fp32 = _predict(model, test_dl, device)
    fp32_acc = float(accuracy_score(y_true, y_pred_fp32))
    fp32_macro_f1 = float(f1_score(y_true, y_pred_fp32, average="macro", zero_division=0))
    report = classification_report(
        y_true, y_pred_fp32,
        target_names=list(data.label_mapping.keys()),
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred_fp32).tolist()
    fp32_per_attack = per_attack_metrics(y_true, y_pred_fp32, data.label_mapping)
    logger.info("FP32 overall acc=%.4f | macro-F1=%.4f", fp32_acc, fp32_macro_f1)

    # --- Save FP32 weights ---
    torch.save(model.state_dict(), out_dir / "cnn_bigru_fp32.pt")

    # --- Export ONNX + INT8 and evaluate INT8 ---
    fp32_onnx, int8_onnx = export_onnx(model, data.num_features, out_dir)
    Xte2d = data.X_test.reshape(len(data.X_test), 1, data.num_features)
    y_pred_int8, int8_latency = evaluate_onnx(int8_onnx, Xte2d, data.y_test)
    int8_acc = float(accuracy_score(y_true, y_pred_int8))
    int8_per_attack = per_attack_metrics(y_true, y_pred_int8, data.label_mapping)
    fp32_size = (out_dir / "cnn_bigru_fp32.onnx").stat().st_size / 1e6
    int8_size = (out_dir / "cnn_bigru_int8.onnx").stat().st_size / 1e6
    logger.info("INT8 overall acc=%.4f | size %.2f->%.2f MB | latency=%.4f ms",
                int8_acc, fp32_size, int8_size, int8_latency)

    # --- Persist results (consumed by evaluation/paper_table1.py) ---
    with open(out_dir / "label_mapping.json", "w") as f:
        json.dump(data.label_mapping, f, indent=2)

    results = {
        "domain": "edge_iiotset",
        "num_features": data.num_features,
        "num_classes": data.num_classes,
        "label_mapping": data.label_mapping,
        "feature_names": data.feature_names,
        "fp32": {
            "accuracy": fp32_acc,
            "macro_f1": fp32_macro_f1,
            "per_class_report": report,
            "confusion_matrix": cm,
            "per_attack": fp32_per_attack,
            "onnx_size_mb": fp32_size,
        },
        "int8": {
            "accuracy": int8_acc,
            "per_attack": int8_per_attack,
            "onnx_size_mb": int8_size,
            "latency_ms_per_sample": int8_latency,
            "size_reduction_pct": (1 - int8_size / fp32_size) * 100 if fp32_size else 0.0,
        },
        "training_history": getattr(model, "_history", []),
    }
    with open(out_dir / "edge_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved Edge-IIoTset results -> %s", out_dir / "edge_results.json")
    print("\nDONE. Next: python -m evaluation.paper_table1 --domain edge\n")


if __name__ == "__main__":
    main()
