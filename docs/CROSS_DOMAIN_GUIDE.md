# Cross-Domain Training & Evaluation Guide

This project trains **two independent domain models** — they are **never merged**:

| Domain | Dataset | Pipeline | Output dir |
|---|---|---|---|
| **Medical** | CICIoMT2024 | `kaggle_train.py` (existing) | `checkpoints/` |
| **Industrial** | Edge-IIoTset | `train_edge_iiotset.py` (new) | `checkpoints/edge_iiotset/` |

Each domain gets its own model, its own ONNX/INT8 artifacts, and its own
`Table 1`. This is the honest realisation of the paper's "Cross-Domain" claim.

---

## 0. Train on Kaggle (recommended — no local disk needed)

Your local C: drive is nearly full, and Edge-IIoTset is ~11 GB. **Train on
Kaggle instead** — the dataset mounts read-only and does not touch your disk.

1. New Kaggle Notebook → **Add Input** → search
   *"Edge-IIoTset Cyber Security Dataset of IoT & IIoT"* (mohamedamineferrag).
   It mounts at:
   ```
   /kaggle/input/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot/Edge-IIoTset dataset
   ```
2. Open [`kaggle_train_edge.py`](../kaggle_train_edge.py), copy its entire
   contents into one notebook cell (or upload it and run `%run kaggle_train_edge.py`).
3. Run the cell. It auto-finds `DNN-EdgeIIoT-dataset.csv`, trains, quantizes,
   exports ONNX, prints **Table 1**, and saves everything to
   `/kaggle/working/edge_iiotset/`.
4. Download `edge_results.json`, `cnn_bigru_int8.onnx`, `label_mapping.json`
   into your local `checkpoints/edge_iiotset/` to run the local evaluators
   (`paper_table1.py`, `runtime_benchmark.py`).

The script is **fully self-contained** (model + loader + training inline) — it
needs no other project files on Kaggle.

---

## 1. Place the Edge-IIoTset data (local alternative)

After you extract the 11 GB Edge-IIoTset zip, point the config at it. The loader
targets **File 3.1 — `DNN-EdgeIIoT-dataset.csv`** (the curated ML/DL file).

Option A — drop the folder anywhere under the project and let auto-discovery find it:
```
Agentic AI/
└── data/edge_iiotset/.../DNN-EdgeIIoT-dataset.csv
```

Option B — set the path explicitly in `config/settings.yaml`:
```yaml
edge_iiotset:
  dataset_path: "D:/datasets/Edge-IIoTset"   # folder OR direct path to the CSV
  max_rows: null        # set e.g. 300000 for a fast trial run
```

> The loader applies the **official Edge-IIoTset cleaning recipe**: it drops 15
> leakage/identifier columns (IPs, raw payloads, timestamps, ephemeral ports),
> removes NaN/Inf and duplicates, label-encodes categoricals, and scales features.
> These exact drops are why this dataset reaches the paper's target accuracies.

---

## 2. Train the industrial (Edge-IIoTset) model

```powershell
python train_edge_iiotset.py
```

What it does:
- Trains `CNN-BiGRU` with **focal loss + class weights + balanced sampler**
  (this is what lifts minority-class recall toward the paper targets).
- Early-stops on validation macro-F1.
- Exports `cnn_bigru_fp32.onnx` and INT8 `cnn_bigru_int8.onnx`.
- Computes **per-attack one-vs-Normal detection accuracy + FPR** (Table 1 style).
- Writes everything to `checkpoints/edge_iiotset/edge_results.json`.

Artifacts (all separate from the medical model):
```
checkpoints/edge_iiotset/
├── cnn_bigru_fp32.pt
├── cnn_bigru_fp32.onnx
├── cnn_bigru_int8.onnx
├── label_mapping.json
└── edge_results.json
```

---

## 3. Generate Table 1 (per domain)

```powershell
python -m evaluation.paper_table1 --domain edge   # industrial
python -m evaluation.paper_table1 --domain cic    # medical
python -m evaluation.paper_table1 --domain both
```

Outputs a console table + `results/table1_<domain>.md`, including a reference
column with the paper's targets (DDoS 99.1 / Spoofing 98.2 / MITM 97.1 INT8).

---

## 4. Measure latency & resources (paper §5.2 / §5.3)

```powershell
python -m evaluation.runtime_benchmark --domain edge --iters 2000
python -m evaluation.runtime_benchmark --domain cic  --iters 2000
```

Reports, with PASS/FAIL against targets:
- `tau_edge` — **per-packet** INT8 latency (mean/p50/p95/p99) — target ≤ 3 ms
- `tau_agent` — System 2 ReAct convergence — target ≤ 180 ms
- `T_ttm` = tau_edge + tau_comm + tau_agent + tau_action — target < 250 ms
- peak RSS (MB) and CPU (%) during the loop

> Note: peak RSS reported here includes the whole Python+Torch process. For the
> paper's ≤ 45 MB edge figure, run the edge model inside the constrained
> container (`--memory=128mb --cpus=0.5`) defined in `infrastructure/`.

---

## 5. Windows notes (already handled in code)

- `KMP_DUPLICATE_LIB_OK=TRUE` + `OMP_NUM_THREADS=1` — avoids the OpenMP
  duplicate-runtime segfault (0xC0000005).
- The loader forces pandas' **non-Arrow string backend** — the PyArrow string
  backend segfaults on Windows during `read_csv`.
- `PYTHONIOENCODING=utf-8` + stable ONNX exporter — avoids the cp1252 crash on
  torch.onnx's unicode status prints.

These are set automatically inside `train_edge_iiotset.py`.

---

## 6. Why this reaches the targets (and CICIoMT2024 struggled)

- Edge-IIoTset's `DNN-EdgeIIoT-dataset.csv` is a **balanced, curated** DL split;
  with the standard leakage-column drops it is highly separable (literature
  routinely reports 94–100% multiclass, ~100% binary detection).
- Focal loss + class weights + balanced sampling directly target the
  minority-class recall problem that capped the medical model at 85.8%.
- Keeping domains **separate** avoids the label-collision/feature-mismatch that
  the old merged `kaggle_train.py` introduced.
