# -*- coding: utf-8 -*-
"""
Paper Table 1 Generator  (per-domain, datasets kept SEPARATE)
=============================================================

Reproduces **Table 1** of the paper — per-attack detection performance
(FP32 vs INT8) plus False-Positive Rate — for EITHER domain independently:

  * ``--domain cic``   -> CICIoMT2024 (medical), from checkpoints/training_results.json
  * ``--domain edge``  -> Edge-IIoTset (industrial), from checkpoints/edge_iiotset/edge_results.json

The two domains are NEVER combined; each prints its own table. This is the
honest "cross-domain" reporting the paper requires.

Outputs:
  * A console table.
  * A markdown table written to ``results/table1_<domain>.md``.

Usage::

    python -m evaluation.paper_table1 --domain edge
    python -m evaluation.paper_table1 --domain cic
    python -m evaluation.paper_table1 --domain both
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[2]

# Paper Table 1 targets (for reference columns).
PAPER_TARGETS = {
    "DDoS": {"fp32": 99.4, "int8": 99.1, "fpr": 0.05},
    "Spoofing": {"fp32": 98.7, "int8": 98.2, "fpr": 0.10},
    "MITM": {"fp32": 97.9, "int8": 97.1, "fpr": 0.15},
}


# --------------------------------------------------------------------------- #
#  CIC (medical) — derive per-attack accuracy + FPR from the confusion matrix
# --------------------------------------------------------------------------- #
def _cic_rows(results_path: Path) -> Tuple[List[List[str]], Dict]:
    data = json.loads(results_path.read_text())
    tr = data.get("test_results", data)
    cm = tr.get("confusion_matrix")
    report = tr.get("per_class_report", {})
    # Label order from training (Benign=0 ...).
    labels = ["Benign", "DDoS", "DoS", "MITM", "Reconnaissance", "Spoofing"]
    int8_acc = data.get("quantization", {}).get("int8_accuracy")

    rows: List[List[str]] = []
    if cm:
        n = len(cm)
        total = sum(sum(r) for r in cm)
        benign_id = 0
        for i, name in enumerate(labels[:n]):
            if name == "Benign":
                continue
            tp = cm[i][i]
            fp = sum(cm[j][i] for j in range(n)) - tp
            fn = sum(cm[i]) - tp
            tn = total - tp - fp - fn
            # one-vs-benign detection accuracy
            det_acc = (tp + cm[benign_id][benign_id]) / (
                tp + fn + sum(cm[benign_id]) ) if (tp + fn) else 0.0
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            # use per-class recall as the FP32 "accuracy" proxy if available
            rec = report.get(name, {}).get("recall")
            fp32_acc = rec * 100 if rec is not None else det_acc * 100
            rows.append([
                name, f"{fp32_acc:.2f}",
                f"{int8_acc*100:.2f}" if int8_acc else "n/a",
                f"{fpr*100:.4f}",
            ])
    return rows, data


# --------------------------------------------------------------------------- #
#  Edge (industrial) — read the structured per_attack block we already compute
# --------------------------------------------------------------------------- #
def _edge_rows(results_path: Path) -> Tuple[List[List[str]], Dict]:
    data = json.loads(results_path.read_text())
    fp32_pa = data.get("fp32", {}).get("per_attack", {})
    int8_pa = data.get("int8", {}).get("per_attack", {})
    rows: List[List[str]] = []
    for cls, m in fp32_pa.items():
        if cls.startswith("_"):
            continue
        fp32_acc = m["detection_accuracy"] * 100
        int8_acc = int8_pa.get(cls, {}).get("detection_accuracy")
        fpr = m["fpr"] * 100
        rows.append([
            cls, f"{fp32_acc:.2f}",
            f"{int8_acc*100:.2f}" if int8_acc is not None else "n/a",
            f"{fpr:.4f}",
        ])
    return rows, data


# --------------------------------------------------------------------------- #
#  Rendering
# --------------------------------------------------------------------------- #
def _render(domain: str, rows: List[List[str]], extra: Dict) -> str:
    header = ["Attack Vector", "FP32 Acc (%)", "INT8 Acc (%)", "FPR (%)"]
    lines = [f"# Table 1 — {domain.upper()} domain (per-attack detection)", ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    lines.append("")
    # Overall + sizes
    if domain == "edge":
        f = extra.get("fp32", {}); q = extra.get("int8", {})
        lines.append(f"- Overall FP32 accuracy: **{f.get('accuracy', 0)*100:.2f}%**, "
                     f"macro-F1: **{f.get('macro_f1', 0):.4f}**")
        lines.append(f"- Overall INT8 accuracy: **{q.get('accuracy', 0)*100:.2f}%**")
        lines.append(f"- Model size: {f.get('onnx_size_mb', 0):.2f} MB -> "
                     f"{q.get('onnx_size_mb', 0):.2f} MB "
                     f"({q.get('size_reduction_pct', 0):.1f}% smaller)")
        lines.append(f"- INT8 latency: {q.get('latency_ms_per_sample', 0):.4f} ms/sample")
        g = extra.get("fp32", {}).get("per_attack", {}).get("_global", {})
        if g:
            lines.append(f"- Benign FPR (global): {g.get('benign_fpr', 0)*100:.4f}%")
    else:
        tr = extra.get("test_results", {})
        lines.append(f"- Overall accuracy (FP32): **{tr.get('accuracy', 0)*100:.2f}%**")
        q = extra.get("quantization", {})
        if q:
            lines.append(f"- INT8 accuracy: **{q.get('int8_accuracy', 0)*100:.2f}%**, "
                         f"size {q.get('fp32_size_mb', 0):.2f}->{q.get('int8_size_mb', 0):.2f} MB")
    lines.append("")
    lines.append("### Paper targets (reference)")
    lines.append("| Attack | FP32 target | INT8 target | FPR target |")
    lines.append("|---|---|---|---|")
    for k, v in PAPER_TARGETS.items():
        lines.append(f"| {k} | {v['fp32']}% | {v['int8']}% | <{v['fpr']}% |")
    return "\n".join(lines)


def run_domain(domain: str) -> Optional[str]:
    if domain == "cic":
        path = _ROOT / "checkpoints" / "training_results.json"
        if not path.exists():
            print(f"[cic] results not found: {path}")
            return None
        rows, extra = _cic_rows(path)
    elif domain == "edge":
        path = _ROOT / "checkpoints" / "edge_iiotset" / "edge_results.json"
        if not path.exists():
            print(f"[edge] results not found: {path}. Run train_edge_iiotset.py first.")
            return None
        rows, extra = _edge_rows(path)
    else:
        raise ValueError(domain)

    md = _render(domain, rows, extra)
    out_dir = _ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"table1_{domain}.md"
    out_file.write_text(md, encoding="utf-8")
    print(md)
    print(f"\n[Saved] {out_file}\n")
    return md


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate paper Table 1 per domain.")
    ap.add_argument("--domain", choices=["cic", "edge", "both"], default="both")
    args = ap.parse_args()
    if args.domain == "both":
        run_domain("cic")
        run_domain("edge")
    else:
        run_domain(args.domain)


if __name__ == "__main__":
    main()
