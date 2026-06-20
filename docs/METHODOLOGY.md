# Project Methodology

## 1. Objective and Research Questions

This project implements and validates a dual-process neuro-symbolic security architecture for Industrial Medical IoT (IIoMT):

- System 1 (edge reflex): low-latency detection and immediate containment triggers
- System 2 (gateway reasoning): risk-aware, rule-constrained mitigation decisions
- Human-in-the-loop (HITL): clinician/operator review and overrides for safety-critical actions

Primary validation questions:

1. Can the edge classifier provide high detection quality under strict runtime constraints?
2. Can the gateway reasoning loop converge fast enough for real-time mitigation?
3. Can mitigation remain safe for medical/industrial constraints (no unsafe auto-quarantine)?
4. Can all reported values be traced back to raw artifacts and reproduced?

## 2. Experimental Design

### 2.1 Domain Strategy

The project uses two domain tracks that are evaluated independently:

- Medical domain: CICIoMT2024 pipeline
- Industrial domain: Edge-IIoTset pipeline

For publication-style Table 1 and runtime validation, the industrial track is the primary evidence source in this repository state.

### 2.2 Data Pipeline

Industrial data source:

- Dataset: Edge-IIoTset (curated CSV variant)
- Raw file: `datasets/DNN-EdgeIIoT-dataset.csv`
- Result artifact source: `checkpoints/edge_iiotset/edge_results.json`

Pipeline stages:

1. Load and sanitize tabular traffic data
2. Remove unstable/leakage columns and invalid rows
3. Encode categorical fields and normalize features
4. Split into train/validation/test
5. Train FP32 CNN-BiGRU model
6. Export and quantize to INT8 ONNX
7. Benchmark latency/resource behavior and aggregate paper-style metrics

## 3. Model and Inference Method

### 3.1 System 1 (Edge Reflex)

- Architecture: CNN-BiGRU classifier
- Input shape in deployed artifacts: `[batch, 1, 46]`
- Output: 15-class logits (industrial domain labels)
- Quantization: post-training INT8 ONNX for edge runtime

### 3.2 Dynamic Thresholding and Brake

- KDE-style anomaly thresholding is used for adaptive boundarying
- Emergency brake performs fast micro-mitigation signaling for suspicious traffic bursts

## 4. Gateway Reasoning Method (System 2)

System 2 follows a structured loop:

- OBSERVE -> THINK -> PLAN -> VALIDATE -> ACT -> EXPLAIN

Risk fusion uses weighted context components:

$$
Risk = \alpha \cdot ClfConf + \beta \cdot Criticality + \gamma \cdot HistoricalDensity
$$

Actions are selected from a graduated 5-level playbook:

1. LOG_ONLY
2. THROTTLE
3. MICRO_SEGMENT
4. RE_AUTHENTICATE
5. QUARANTINE

Safety rules constrain actions on high-criticality devices.

## 5. Metrics, Targets, and Mathematical Definitions

### 5.1 Detection Metrics

Per-class accuracy:

$$
Accuracy = \frac{TP}{Support}
$$

Binary false positive rate against benign traffic:

$$
FPR = \frac{FP}{FP + TN} \times 100\%
$$

### 5.2 Runtime and System Metrics

- Edge latency: $\tau_{edge}$ (ms/sample)
- Agent latency: $\tau_{agent}$ (ms/iteration)
- Time-to-mitigation:

$$
T_{ttm} = \tau_{edge} + \tau_{comm} + \tau_{agent} + \tau_{action}
$$

- Edge memory working set (MB)
- CPU steady-state utilization per core (%)

## 6. Reproducibility Protocol

### 6.1 Environment

```bash
pip install -e .
```

### 6.2 Recompute Main Report Outputs

```bash
python -m evaluation.paper_table1 --domain edge
python -m evaluation.runtime_benchmark --domain edge
```

### 6.3 Evidence Files

Primary raw evidence:

- `checkpoints/edge_iiotset/edge_results.json`
- `results/table1_edge.md`
- `results/runtime_benchmark_edge.json`

Proof/audit references:

- `docs/TABLE1_MATHEMATICAL_AUDIT.md`
- `docs/VERIFICATION_GUIDE.md`
- `docs/FINAL_VERIFICATION_REPORT.md`

## 7. Results Interpretation Policy

To avoid over-claiming:

- Always publish target-vs-actual deltas
- Do not replace measured values with rounded target values
- Keep all misses explicit (for this repo state: DDoS aggregate INT8 is below 99.1% target)
- Prioritize verifiable operational metrics (FPR, latency, memory, CPU) with direct file provenance

## 8. Current Verified Outcome (Industrial Track)

From `results/table1_edge.md` and `results/runtime_benchmark_edge.json`:

- DDoS aggregate INT8 detection: 98.38% (target 99.1%)
- Spoofing aggregate INT8 detection: 99.55% (target 98.2%)
- MITM INT8 detection: 98.76% (target 97.1%)
- DDoS FPR: 0.0026%
- Spoofing FPR: 0.023%
- MITM FPR: 0.0022%
- $\tau_{edge}$ mean: 0.229 ms
- $\tau_{agent}$ mean: 0.167 ms
- $T_{ttm}$: 15.396 ms
- Edge model working set: 9.96 MB
- CPU steady per core: 2.83%

These values are reproducible from the listed artifacts and scripts.

## 9. Limitations and Next Validation Step

- DDoS aggregate detection is 0.72 percentage points below the paper target in the current run
- Linux-hosted Mininet/tcpreplay validation is prepared but host-dependent
- Real SLM-backed explanations can replace deterministic fallback for production-style deployment studies

Despite these limits, the core architecture and runtime viability are strongly supported by measurable evidence.
