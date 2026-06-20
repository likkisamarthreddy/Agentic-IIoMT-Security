# Cross-Domain Agentic Security for Industrial Medical IoT

A **dual-process neuro-symbolic agentic framework** that transforms lightweight deep learning perception engines into autonomous action-oriented security agents for Industrial Medical IoT (IIoMT) networks.

> **Research Implementation** — This codebase implements the architecture described in *"Cross-Domain Agentic Security for Industrial Medical IoT"*, featuring a System 1 (Edge Reflex Layer) and System 2 (Gateway Reasoning Engine) with Human-in-the-Loop (HITL) verification.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│              Heterogeneous IIoMT Traffic Stream                 │
│              MQTT, CoAP, DICOM, HL7, Bluetooth                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  SYSTEM 1: Edge Reflex Layer                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  INT8-Quantized CNN-BiGRU Classifier                     │   │
│  │  + KDE Dynamic Anomaly Threshold                         │   │
│  │  + Emergency Brake (SDN Micro-Mitigation)                │   │
│  └──────────────────────────────────────────────────────────┘   │
│  Target: τ_edge ≤ 3ms | Memory ≤ 45MB | CPU ≤ 15%             │
└──────────────────────────┬──────────────────────────────────────┘
                           │ MQTT Alert
┌──────────────────────────▼──────────────────────────────────────┐
│  SYSTEM 2: Gateway Reasoning Engine                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Context Fusion → Risk Metric (Eq. 1)                    │   │
│  │  Reason-and-Act (ReAct) Loop                             │   │
│  │  Symbolic Safety Rule Validation                         │   │
│  │  Graduated Action Playbook (5 mitigation levels)         │   │
│  │  Optional: 3B-Parameter SLM via Ollama                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│  Target: τ_agent ≤ 180ms | T_ttm < 250ms                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  HITL Dashboard                                                 │
│  Real-time alerts • NL explanations • Clinician overrides       │
│  Network topology • Latency metrics • Resource monitoring       │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

The project uses a `src/` layout. Install it in editable mode once so the
packages (`system1`, `system2`, `data`, `evaluation`, `dashboard`,
`infrastructure`) are importable from anywhere:

```bash
pip install -e .
```

This also installs the runtime dependencies listed in `requirements.txt`.

> All executable pipeline scripts live in `scripts/`. They self-bootstrap
> `src/` onto the path and anchor the working directory to the repo root, so
> you can run them from any location.

### 2. Run End-to-End Demo

```bash
python scripts/main.py demo
```

This will:
- Generate synthetic IIoMT traffic (20,000 flows)
- Train a CNN-BiGRU classifier
- Quantize to INT8
- Run System 2 reasoning on a simulated alert
- Print full metrics comparison against paper targets

### 3. Full Training Pipeline

```bash
# Train on synthetic data
python scripts/main.py train --data synthetic --epochs 50

# Quantize and prune the model
python scripts/main.py quantize

# Run simulation with attack injection
python scripts/main.py simulate --duration 60

# Generate evaluation report
python scripts/main.py evaluate --output results/
```

### 4. Train the industrial (Edge-IIoTset) domain model

```bash
# Place the dataset under datasets/ (auto-discovered), then:
python scripts/train_edge_iiotset.py

# Generate paper Table 1 and the runtime benchmark (installed packages):
python -m evaluation.paper_table1 --domain edge
python -m evaluation.runtime_benchmark --domain edge
```

### 5. Launch HITL Dashboard

```bash
python scripts/main.py dashboard
# Open http://localhost:5000 in your browser
```

## Project Structure

Source code lives **only** under `src/`; executable pipeline scripts live
**only** under `scripts/`.

```
Agentic AI/
├── pyproject.toml                   # src-layout packaging (pip install -e .)
├── conftest.py                      # makes src/ importable for pytest
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── .gitignore
│
├── config/
│   ├── settings.yaml                # Global configuration & hyperparameters
│   └── safety_policies.yaml         # Symbolic safety rules & device constraints
│
├── scripts/                         # Executable pipeline scripts (entry points)
│   ├── main.py                      # CLI entry point (6 subcommands)
│   ├── train_edge_iiotset.py        # Industrial-domain (Edge-IIoTset) trainer
│   ├── export_onnx.py               # ONNX export + INT8 quantization
│   ├── run_inference_test.py        # Deployment inference smoke test
│   ├── build_kaggle.py              # Bundle a single-file Kaggle script
│   └── kaggle_train*.py             # Self-contained Kaggle training scripts
│
├── datasets/                        # (gitignored) raw datasets
├── checkpoints/                     # (gitignored) trained model artifacts
├── results/                         # (gitignored) generated reports/metrics
├── artifacts/                       # (gitignored) zips, logs, dumps
│
└── src/                             # All importable source code
    │
    ├── data/
    │   ├── synthetic_generator.py       # Synthetic IIoMT data generator
    │   ├── preprocessor.py              # Feature engineering pipeline
    │   ├── edge_iiotset_loader.py       # Edge-IIoTset loader (industrial)
    │   └── traffic_replay.py            # Real-time traffic streaming via MQTT
    │
    ├── system1/                     # Edge Reflex Layer
│   ├── edge_agent.py                # Main edge agent orchestrator
│   ├── models/
│   │   ├── cnn_bigru.py             # CNN-BiGRU hybrid classifier
│   │   └── isolation_forest_lite.py # Lightweight Isolation Forest
│   ├── quantization/
│   │   ├── quantizer.py             # INT8 post-training quantization
│   │   └── pruner.py                # L1 structured channel pruning
│   ├── detection/
│   │   ├── kde_threshold.py         # Adaptive KDE anomaly threshold
│   │   └── emergency_brake.py       # SDN emergency brake
│   └── training/
│       └── trainer.py               # Full training pipeline
│
├── system2/                         # Gateway Reasoning Engine
│   ├── gateway_agent.py             # Main gateway orchestrator
│   ├── reasoning/
│   │   ├── context_fusion.py        # Risk metric computation (Eq. 1)
│   │   ├── symbolic_rules.py        # Safety rule validation engine
│   │   ├── reason_act_loop.py       # Structured ReAct loop
│   │   └── slm_interface.py         # Optional SLM connection (Ollama)
│   └── mitigation/
│       ├── action_playbook.py       # 5-level graduated actions
│       └── sdn_controller.py        # Simulated SDN rule executor
│
├── dashboard/                       # HITL Web Interface
│   ├── app.py                       # Flask + SocketIO server
│   ├── templates/
│   │   └── index.html               # Dashboard UI
│   └── static/
│       ├── styles.css               # Premium glassmorphism styles
│       └── dashboard.js             # Real-time dashboard logic
│
├── infrastructure/                  # Deployment & Emulation
│   ├── docker-compose.yaml          # Containerized topology
│   ├── Dockerfile.edge              # Edge node container
│   ├── Dockerfile.gateway           # Gateway container
│   └── network_emulator.py          # Windows-compatible emulator
│
└── evaluation/                      # Metrics & Reporting
    ├── metrics_collector.py         # Comprehensive metrics collection
    ├── attack_injector.py           # Phase 3 attack simulation
    └── benchmark_report.py          # Publication-quality reports
```

## Key Technical Details

### System 1: CNN-BiGRU Classifier
- **Architecture**: 2× Conv1D (64→128) + BatchNorm + ReLU + MaxPool → 2-layer BiGRU (hidden=64, bidirectional) → Dense classifier
- **Optimization**: INT8 post-training quantization + L1 structured pruning (30%)
- **Anomaly Detection**: Adaptive KDE threshold with sliding window (1000 samples)
- **Emergency Brake**: Automatic SDN micro-mitigation when anomaly score exceeds KDE threshold

### System 2: Gateway Reasoning Engine
- **Risk Metric**: `RiskMetric = α·Clf_Conf + β·Criticality_Index + γ·Historical_Density`
- **ReAct Loop**: OBSERVE → THINK → PLAN → VALIDATE → ACT → EXPLAIN
- **Safety Rules**: Device-type constraints (never auto-quarantine infusion pumps), anti-flap guards, telemetry preservation
- **5 Mitigation Levels**: LOG_ONLY → THROTTLE → MICRO_SEGMENT → RE_AUTHENTICATE → QUARANTINE

### Evaluation Targets (Paper §5)

| Metric | Target |
|--------|--------|
| DDoS Accuracy (INT8) | ≥ 99.1% |
| Spoofing Accuracy (INT8) | ≥ 98.2% |
| MITM Accuracy (INT8) | ≥ 97.1% |
| Edge Inference Latency (τ_edge) | ≤ 3 ms |
| Agent Convergence (τ_agent) | ≤ 180 ms |
| Total Time-to-Mitigation (T_ttm) | < 250 ms |
| Edge Peak Memory | ≤ 45 MB |
| Edge CPU Overhead | ≤ 15% |

## Docker Deployment (Linux)

```bash
cd infrastructure
docker-compose up --build
```

## Configuration

All parameters are centralized in `config/settings.yaml`:
- Model hyperparameters (CNN filters, GRU dimensions, dropout rates)
- KDE threshold parameters (bandwidth, percentile, window size)
- Risk metric weights (α, β, γ)
- MQTT broker settings
- Device criticality matrix
- Latency and memory targets

Safety policies in `config/safety_policies.yaml`:
- Graduated mitigation levels with risk score ranges
- Per-device-type constraints and forbidden actions
- HITL override policies and escalation chain
- Symbolic validation rules (6 rules with priority ordering)

## License

Research implementation — see accompanying paper for citation.
