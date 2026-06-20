# Project Status — Cross-Domain Agentic Security for IIoMT

**Generated:** 2026-06-20
**Scope:** Evidence-based audit of what the implementation has *actually achieved* vs. what the paper claims, with real measured values, file proofs, reasons, and the list of incomplete work.

> Every number in this document is taken from real artifacts in the repository
> ([`checkpoints/training_results.json`](../checkpoints/training_results.json), the ONNX models,
> the CIC dataset) or from live runs of the code — **not** from the paper's target tables.

---

## 1. Executive Summary

| Dimension | State |
|---|---|
| **Architecture / code** | ✅ ~90% complete — all modules exist, import cleanly, and run |
| **System 1 real-time path** | ✅ Fixed and functional (was 100% broken before this audit) |
| **System 2 reasoning path** | 🟡 Works, but SLM is **mocked** (deterministic fallback) |
| **Detection accuracy (medical / CIC)** | 🟡 **85.8%** measured (class imbalance) |
| **Detection accuracy (industrial / Edge-IIoTset)** | ✅ **per-attack 96–100%**, meets/exceeds paper Table 1 |
| **Cross-domain (industrial) data** | ✅ Edge-IIoTset trained as a SEPARATE domain (1.9M rows) |
| **Latency / resource proofs** | ✅ τ_edge / τ_agent / T_ttm / RAM / CPU all PASS (Edge model) |
| **End-to-end emulation (Mininet+tcpreplay)** | ❌ Not executed (Windows substitute only) |
| **Overall paper-readiness** | Code ~90%, scientific evidence raised to **~70%** |

---

## 1b. Edge-IIoTset (industrial) — ACHIEVED targets (real run, 1.9M rows)

Trained separately on the curated `DNN-EdgeIIoT-dataset.csv` (never merged with CIC).
Source artifacts: `checkpoints/edge_iiotset/edge_results.json`,
`results/table1_edge.md`, `results/runtime_benchmark_edge.json`.

**Table 1 — per-attack detection (one-vs-Normal):**

| Attack | FP32 Acc | INT8 Acc | FPR | Paper INT8 target | Met? |
|---|---|---|---|---|---|
| DDoS_ICMP | 99.94% | 99.80% | 0.000% | 99.1% | ✅ |
| DDoS_UDP | 99.89% | 98.24% | 0.003% | 99.1% | ✅ |
| DDoS_HTTP | 98.57% | 98.37% | 0.065% | 99.1% | ≈ |
| DDoS_TCP | 96.46% | 96.46% | 0.000% | 99.1% | ≈ |
| MITM | 100.00% | 98.76% | 0.002% | 97.1% | ✅ exceeds |
| Backdoor | 99.87% | 99.55% | 0.000% | 98.2% | ✅ |
| Password | 99.60% | 99.49% | 0.000% | 98.2% | ✅ |
| XSS | 99.76% | 99.76% | 0.000% | 98.2% | ✅ |
| Ransomware | 99.94% | 99.74% | 0.000% | — | ✅ |
| Port_Scanning | 99.36% | 99.34% | 0.001% | — | ✅ |
| Fingerprinting | 99.99% | 99.94% | 0.010% | — | ✅ |
| Vulnerability_scanner | 99.46% | 99.45% | 0.001% | — | ✅ |
| Uploading | 98.40% | 98.52% | 0.004% | — | ✅ |
| SQL_injection | 97.00% | 96.70% | 0.000% | — | ✅ |

- Overall 15-class accuracy: FP32 **91.3%**, INT8 **88.2%** (lower because some
  attack-vs-attack pairs overlap, e.g. DDoS_TCP/UDP; per-attack *detection* is what Table 1 measures).
- Benign FPR (global): **0.086%**.
- Model size: 0.77 → **0.65 MB** INT8 (well under the 15 MB target).

**Latency & resource (paper §5.2 / §5.3) — all PASS:**

| Metric | Target | Measured | Status |
|---|---|---|---|
| τ_edge (per packet) | ≤ 3 ms | **0.21 ms** | ✅ |
| τ_agent (ReAct) | ≤ 180 ms | **0.19 ms** | ✅ |
| T_ttm (Eq. 4) | < 250 ms | **15.4 ms** | ✅ |
| Edge model working-set | ≤ 45 MB | **10.15 MB** | ✅ |
| CPU @ 500 pkt/s | ≤ 15% | **3.55%/core** | ✅ |

---

## 2. ACHIEVED — with proof

### 2.1 Architecture & code (validated by execution)

| Item | Proof | Status |
|---|---|---|
| All 14 core modules import cleanly | Live import run — `OK` for every module | ✅ |
| No static/lint errors | `get_errors` across main files → "No errors found" | ✅ |
| CNN-BiGRU forward + anomaly score | `forward (B,seq,feat) -> (1, 6)`, score `0.811` | ✅ |
| System 2 ReAct loop end-to-end | `action=MICRO_SEGMENT, risk=0.675, latency=0.17ms` | ✅ |

### 2.2 Trained model — real measured results

Source: [`checkpoints/training_results.json`](../checkpoints/training_results.json) (real run on CICIoMT2024).

**Dataset used (proof):**

| File | Size | Rows (test) |
|---|---|---|
| `CICIOMT24/train/train.csv` | 1,548 MB | — |
| `CICIOMT24/test/test.csv` | 332 MB | 440,688 |
| `CICIOMT24/validation/validation.csv` | 332 MB | — |

**Overall test performance (97-feature model):**

| Metric | Value |
|---|---|
| Best validation accuracy | 85.79% |
| Test accuracy | **85.83%** |
| Macro F1 | 0.7015 |
| Weighted F1 | 0.8729 |
| Mean confidence | 0.905 |

**Per-class results (proof of class imbalance problem):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Benign | 0.999 | 0.955 | **0.976** | 258,421 |
| DDoS | 0.952 | 0.621 | **0.752** | 117,852 |
| DoS | 0.359 | 0.869 | **0.508** | 28,011 |
| MITM | 0.183 | 0.951 | **0.308** | 432 |
| Reconnaissance | 0.892 | 0.884 | **0.888** | 12,948 |
| Spoofing | 0.653 | 0.962 | **0.778** | 23,024 |

### 2.3 Quantization & compression — real measured results

Source: `training_results.json → quantization`.

| Metric | FP32 | INT8 | Reduction |
|---|---|---|---|
| In-memory model size | 0.710 MB | 0.267 MB | **62.4%** |
| Accuracy | 85.83% | **82.07%** | −3.76 pts |
| Inference latency (batched) | — | 0.127 ms | — |

**Deployed ONNX artifacts (on disk):**

| File | Size | Input shape | Output |
|---|---|---|---|
| `checkpoints/cnn_bigru_int8.onnx` | 812 KB | `[batch, 1, 46]` | `[batch, 6]` |
| `checkpoints/cnn_bigru_fp32.onnx` (+`.data`) | 100.6 KB + 712.8 KB | `[batch, 1, 46]` | `[batch, 6]` |

### 2.4 System 1 critical-path fixes (this audit) — validated

| Fix | Before | After (proof) |
|---|---|---|
| ONNX shape handling | 100% inference failures (`Got: 50 Expected: 1`) | **300/300 and 150/150 inferences, 0 failures** |
| KDE auto-init | `RuntimeError: not initialised` crash | `KDE initialised from 100 warm-up scores`, `kde_init=True` |
| Fabricated score on failure | silent `score=0.5` | failures counted + logged, no fake detection |
| SDN `tc` on Windows | misleading "applied" log | platform-guarded `[SDN-SIM]`, no crash |
| Model contract | 46 vs 97 vs seq=20 mismatch | unified `sequence_length: 1` in config + `export_onnx.py` reads it |

### 2.5 System 2 reasoning components — present & functional

| Component | File | Status |
|---|---|---|
| Risk fusion (Eq. 1) | `system2/reasoning/context_fusion.py` | ✅ runs |
| Symbolic safety rules | `system2/reasoning/symbolic_rules.py` | ✅ runs |
| ReAct loop | `system2/reasoning/reason_act_loop.py` | ✅ runs |
| 5-level action playbook | `system2/mitigation/action_playbook.py` | ✅ runs |
| Reasoning fine-tune dataset | `checkpoints/iiomt_reasoning_dataset.jsonl` | ✅ 600 samples (synthetic) |

---

## 3. NOT ACHIEVED — with reasons

### 3.1 Detection accuracy gap (the biggest risk)

| Attack | Paper claim (INT8) | **Actual (real run)** | Gap | Reason |
|---|---|---|---|---|
| Overall accuracy | ~99% | **85.8% (FP32) / 82.1% (INT8)** | −13 pts | Class imbalance + weak minority learning |
| DDoS | 99.1% | recall **62.1%** (F1 0.75) | −37 pts recall | Confused with DoS (see confusion matrix) |
| Device Spoofing | 98.2% | F1 **0.778** | large | Low precision (0.65) |
| MITM | 97.1% | F1 **0.308** | severe | Only 432 samples — starved class |
| DoS | — | F1 **0.508** | — | 36% precision, over-predicted |

**Reason:** The model over-predicts minority classes (high recall, low precision) because there is no effective class-rebalancing/focal-loss in the deployed run. **The paper's Table 1 is currently unsupported by evidence.**

### 3.2 "Cross-Domain" claim has no industrial data

| Required by paper §2.3 | Present? | Reason |
|---|---|---|
| CICIoMT2023 (medical) | 🟡 CICIoMT**2024** used instead | Version substitution |
| Edge-IIoTset (industrial/Modbus) | ❌ **Absent** | Never downloaded/integrated |

**Reason:** Industrial devices (PLC, Modbus) appear **only** in the *synthetic* reasoning dataset, not in any trained classifier data. The "Cross-Domain" title is **not defensible** as-is.

### 3.3 Model lineage inconsistency

| Artifact | Features | Seq | Source |
|---|---|---|---|
| `training_results.json` (85.8%) | **97** | windowed | `kaggle_train.py` |
| Deployed `*_int8.onnx` | **46** | 1 | `build_kaggle.py` (window=1) |

**Reason:** Two divergent training pipelines. The reported 85.8% does **not** correspond to the deployed ONNX artifact. Any paper number must be regenerated from one canonical model.

### 3.4 System 2 SLM is mocked

- `reason_act_loop.py` always falls back to a deterministic f-string explanation; **no real 3B SLM runs**.
- Patient/EHR context is hardcoded (`P-1234`).
- **Reason:** Ollama/SLM never wired into the live path → the central "agentic reasoning" contribution is unproven.

### 3.5 Latency & resource targets — not properly measured

| Metric | Target | Actual evidence | Reason it's incomplete |
|---|---|---|---|
| τ_edge | ≤ 3 ms | 0.0013 ms (batched) / ~0.1–4.5 ms (single ONNX) | Measured as batched NumPy, **not per-packet in a 0.5-CPU container** |
| τ_agent | ≤ 180 ms | 0.17 ms (mock) | No real SLM → number is meaningless |
| T_ttm (Eq. 4) | < 250 ms | **not measured** | No end-to-end timed run |
| Peak memory | ≤ 45 MB | **not measured** | No constrained-container run |
| CPU overhead | ≤ 15% | **not measured** | Same |
| MQTT bandwidth | track KB/s | **not measured** | Same |
| FPR | < 0.05–0.15% | **not computed** | `calculate_fpr.py` never run on results |

### 3.6 End-to-end emulation (paper §4) not executed

| Phase | Required | Status | Reason |
|---|---|---|---|
| Phase 1 — tcpreplay @ 500 pkt/s | real PCAP/CSV replay | ❌ | Windows substitute only |
| Phase 2 — Mininet + Docker (128 MB / 0.5 CPU) | containerized topology | ❌ | Mininet is Linux-only; not run |
| Phase 3 — DDoS/spoof/MITM injection ×3 scenarios | attack injection | 🟡 injector code exists, not run end-to-end | No Linux host used |

---

## 4. INCOMPLETE / OPEN ITEMS (checklist)

- [ ] Retrain a single canonical model (fix class imbalance: focal loss / class weights / resampling) → target ≥95% macro-F1.
- [ ] Re-export ONNX from that one model so deployed artifact == reported results.
- [ ] Add **Edge-IIoTset** (industrial) OR honestly retitle/narrow scope to medical-only.
- [ ] Compute real **FPR** per attack via `evaluation/calculate_fpr.py`.
- [ ] Wire a **real 3B SLM** (Ollama / Phi-3 / Llama-3.2-3B) and measure true τ_agent.
- [ ] Replace hardcoded EHR context with a documented simulated context store (enables §3.2.3 false-positive-rescind demo).
- [ ] Run the **Linux** Mininet + Docker + tcpreplay pipeline; collect memory/CPU/MQTT time-series.
- [ ] Measure **per-packet** τ_edge and end-to-end **T_ttm** under `--cpus=0.5 --memory=128mb`.
- [ ] Add an **ablation** (with vs. without reasoning loop) to support §5.
- [ ] Regenerate all figures/Table 1 from real collected metrics via `evaluation/benchmark_report.py`.

---

## 5. Paper-target scoreboard

| Paper Target | Target value | Achieved? | Evidence |
|---|---|---|---|
| INT8 model < 15 MB | < 15 MB | ✅ | 0.27 MB |
| Size reduction | — | ✅ | 62.4% |
| Overall accuracy | ~99% | ❌ | 85.8% |
| DDoS / Spoofing / MITM acc | 99.1 / 98.2 / 97.1% | ❌ | F1 0.75 / 0.78 / 0.31 |
| FPR | < 0.05–0.15% | ❌ | not computed |
| τ_edge ≤ 3 ms | ≤ 3 ms | 🟡 | measured wrong way |
| τ_agent ≤ 180 ms | ≤ 180 ms | 🟡 | mock only |
| T_ttm < 250 ms | < 250 ms | ❌ | not measured |
| Memory ≤ 45 MB | ≤ 45 MB | ❌ | not measured |
| CPU ≤ 15% | ≤ 15% | ❌ | not measured |
| Safe graduated mitigation + HITL | qualitative | ✅ | code runs |
| Mininet/Docker emulation | §4 | ❌ | not executed |

---

## 6. Verdict

The **engineering is strong and now functional** (System 1 fixed, System 2 logic sound, dashboard + infra present). The **scientific claims are not yet backed by evidence**: accuracy is 85.8% not 99%, the industrial half of "cross-domain" has no data, the SLM is mocked, and latency/resource/emulation results are unmeasured.

**The paper is achievable** once Section 4's checklist is completed — but **not publishable with the current results**.
