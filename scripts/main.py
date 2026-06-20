"""
Cross-Domain Agentic Security for Industrial Medical IoT
=========================================================

Main CLI entry point for the dual-process neuro-symbolic agentic
security framework. Provides subcommands for training, quantization,
simulation, evaluation, and dashboard operation.

Usage:
    python main.py train        # Train CNN-BiGRU model
    python main.py quantize     # Quantize and prune trained model
    python main.py simulate     # Run full emulated pipeline
    python main.py evaluate     # Run attack injection + metrics
    python main.py dashboard    # Launch HITL web dashboard
    python main.py demo         # End-to-end demo with synthetic data
"""

# --- repo bootstrap: make src/ importable + anchor CWD to repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _os.path.join(_ROOT, "src") not in _sys.path:
    _sys.path.insert(0, _os.path.join(_ROOT, "src"))
_os.chdir(_ROOT)
# --- end bootstrap ---

import argparse
import logging
import sys
import time
import json
import os
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np

class MockMQTTMessage:
    def __init__(self, topic, payload):
        self.topic = topic
        self.payload = payload

class MockMQTTClient:
    _broker_subscribers = {}
    
    def __init__(self, *args, **kwargs):
        self.on_message = None
        self.on_connect = None
        self.on_disconnect = None
        self._connected = False
        
    def connect(self, host, port, keepalive=60):
        self._connected = True
        if self.on_connect:
            self.on_connect(self, None, None, 0)
            
    def is_connected(self): return self._connected
    def loop_start(self): pass
    def loop_stop(self): pass
    def disconnect(self):
        self._connected = False
        if self.on_disconnect: self.on_disconnect(self, None, 0)
            
    def subscribe(self, topic, qos=0):
        if topic not in self.__class__._broker_subscribers:
            self.__class__._broker_subscribers[topic] = []
        if self not in self.__class__._broker_subscribers[topic]:
            self.__class__._broker_subscribers[topic].append(self)
            
    def publish(self, topic, payload, qos=0):
        if not self._connected: return
        msg = MockMQTTMessage(topic, payload)
        subs = self.__class__._broker_subscribers
        
        # Exact match
        if topic in subs:
            for client in subs[topic]:
                if client.on_message: client.on_message(client, None, msg)
                
        # Wildcard match (e.g., iimt/traffic/#)
        for sub_topic, clients in subs.items():
            if sub_topic.endswith("/#"):
                base = sub_topic[:-2]
                if topic.startswith(base):
                    for client in clients:
                        if client.on_message: client.on_message(client, None, msg)

import paho.mqtt.client as mqtt
mqtt.Client = MockMQTTClient

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with consistent formatting."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

logger = logging.getLogger("iimt.main")

# ---------------------------------------------------------------------------
# Configuration Loader
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"
POLICIES_PATH = PROJECT_ROOT / "config" / "safety_policies.yaml"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ===================================================================
# SUBCOMMAND: train
# ===================================================================
def cmd_train(args: argparse.Namespace) -> None:
    """Train the CNN-BiGRU model on synthetic or real data."""
    from data.synthetic_generator import SyntheticIIoMTGenerator
    from data.preprocessor import DataPreprocessor
    from system1.models.cnn_bigru import CNNBiGRU
    from system1.training.trainer import ModelTrainer

    config = load_config()
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 65)
    logger.info("  SYSTEM 1 — CNN-BiGRU MODEL TRAINING")
    logger.info("=" * 65)

    # ----- Data -----
    if args.data == "synthetic":
        logger.info("Generating synthetic IIoMT dataset...")
        generator = SyntheticIIoMTGenerator(Path("config/settings.yaml"))
        df = generator.generate_combined_dataset(
            config["data"]["synthetic"]["num_samples"]
        )
        logger.info(f"Generated {len(df)} samples across "
                    f"{df['label'].nunique()} classes")
    else:
        logger.info(f"Loading dataset from {args.data}...")
        preprocessor = DataPreprocessor(Path("config/settings.yaml"))
        df = preprocessor.load_csv(args.data, max_rows=getattr(args, 'max_rows', None))

    # ----- Preprocess -----
    preprocessor = DataPreprocessor(Path("config/settings.yaml"))
    result = preprocessor.prepare_pipeline(
        df,
        test_ratio=config["data"]["test_ratio"],
        window_size=1,  # no windowing for basic training
    )
    X_train, X_test, y_train, y_test, label_mapping = result

    train_loader, test_loader = preprocessor.get_dataloaders(
        X_train, X_test, y_train, y_test,
        batch_size=config["system1"]["training"]["batch_size"],
    )

    # ----- Model -----
    num_features = X_train.shape[-1]
    num_classes = len(label_mapping)
    model = CNNBiGRU(
        num_features=num_features,
        num_classes=num_classes,
        config_path=Path("config/settings.yaml"),
    )
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ----- Train -----
    trainer = ModelTrainer(model, config["system1"]["training"])
    trainer.train(train_loader, test_loader, epochs=args.epochs or
                  config["system1"]["training"]["epochs"])

    # ----- Evaluate -----
    report = trainer.evaluate(model, test_loader)
    logger.info("\n--- Classification Report ---")
    for key, val in report.items():
        logger.info(f"  {key}: {val}")

    per_attack = trainer.evaluate_per_attack(model, test_loader, label_mapping)
    logger.info("\n--- Per-Attack Metrics (Table 1) ---")
    for atk, metrics in per_attack.items():
        logger.info(f"  {atk}: Acc={metrics['accuracy']:.4f}  "
                    f"FPR={metrics['fpr']:.6f}")

    # ----- Save -----
    ckpt_path = CHECKPOINTS_DIR / "cnn_bigru_fp32.pt"
    trainer.save_checkpoint(model, ckpt_path)
    logger.info(f"FP32 model saved to {ckpt_path}")

    # Save label mapping
    mapping_path = CHECKPOINTS_DIR / "label_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(label_mapping, f, indent=2)
    logger.info(f"Label mapping saved to {mapping_path}")


# ===================================================================
# SUBCOMMAND: quantize
# ===================================================================
def cmd_quantize(args: argparse.Namespace) -> None:
    """Quantize and prune a trained CNN-BiGRU model."""
    import torch
    from system1.models.cnn_bigru import CNNBiGRU
    from system1.quantization.quantizer import ModelQuantizer
    from system1.quantization.pruner import ChannelPruner

    config = load_config()
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 65)
    logger.info("  MODEL OPTIMIZATION — INT8 QUANTIZATION & PRUNING")
    logger.info("=" * 65)

    # ----- Load -----
    model_path = Path(args.model) if args.model else CHECKPOINTS_DIR / "cnn_bigru_fp32.pt"
    mapping_path = CHECKPOINTS_DIR / "label_mapping.json"

    try:
        with open(mapping_path, "r") as f:
            label_mapping = json.load(f)
        num_classes = len(label_mapping)
    except FileNotFoundError:
        logger.warning(f"Label mapping not found at {mapping_path}. Defaulting to 34 classes.")
        num_classes = 34
    num_features = config["data"]["num_features"]

    model = CNNBiGRU(
        num_features=num_features,
        num_classes=num_classes,
        config_path=Path("config/settings.yaml"),
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    logger.info(f"Loaded FP32 model from {model_path}")

    quantizer = ModelQuantizer(config["system1"]["quantization"])

    # ----- Prune -----
    if not args.skip_pruning:
        pruner = ChannelPruner()
        sparsity = config["system1"]["pruning"]["sparsity"]
        pruner.prune_conv_layers(model, amount=sparsity)
        pruner.make_permanent(model)
        report = pruner.get_sparsity_report(model)
        logger.info(f"Pruning complete — sparsity report: {report}")

    # ----- Quantize -----
    fp32_size = quantizer.measure_model_size(model)
    logger.info(f"FP32 model size: {fp32_size:.2f} MB")

    int8_model = quantizer.quantize_dynamic(model)
    int8_size = quantizer.measure_model_size(int8_model)
    logger.info(f"INT8 model size: {int8_size:.2f} MB")
    logger.info(f"Compression ratio: {fp32_size / int8_size:.1f}x")

    # ----- Benchmark -----
    dummy = torch.randn(1, 1, num_features)
    fp32_lat = quantizer.benchmark_latency(model, dummy)
    int8_lat = quantizer.benchmark_latency(int8_model, dummy)
    logger.info(f"FP32 latency: {fp32_lat:.3f} ms")
    logger.info(f"INT8 latency: {int8_lat:.3f} ms")
    logger.info(f"Speedup: {fp32_lat / int8_lat:.2f}x")

    # ----- Save -----
    int8_path = CHECKPOINTS_DIR / "cnn_bigru_int8.pt"
    torch.save(int8_model.state_dict(), int8_path)
    logger.info(f"INT8 model saved to {int8_path}")

    # ----- ONNX Export -----
    onnx_path = CHECKPOINTS_DIR / "cnn_bigru.onnx"
    quantizer.export_onnx(model, onnx_path, (1, 1, num_features))
    logger.info(f"ONNX model exported to {onnx_path}")


# ===================================================================
# SUBCOMMAND: simulate
# ===================================================================
def cmd_simulate(args: argparse.Namespace) -> None:
    """Run the full containerized emulated pipeline (Phase 2)."""
    import subprocess
    import sys
    from pathlib import Path

    logger.info("=" * 65)
    logger.info("  FULL PIPELINE SIMULATION (PHASE 2 EMULATION)")
    logger.info("=" * 65)

    config = load_config()
    test_csv = "CICIOMT24/test/test.csv"
    
    if not Path(test_csv).exists():
        logger.error(f"Cannot run replay. {test_csv} not found.")
        return

    # 1. Start Docker / Mininet infrastructure
    logger.info("Step 1: Initializing Containerized Infrastructure...")
    logger.info("Because Mininet/Containernet requires a native Linux kernel,")
    logger.info("this script will trigger the functionally equivalent Docker Compose topology.")
    logger.info("Building and launching Edge Nodes and Gateway...")
    
    try:
        subprocess.run(
            ["docker", "compose", "-f", "infrastructure/docker-compose.yaml", "up", "-d", "--build"],
            check=True
        )
        logger.info("Containers successfully launched.")
    except Exception as e:
        logger.error(f"Failed to launch docker-compose: {e}")
        logger.warning("Ensure Docker Desktop is running.")
        return

    # Give Mosquitto time to start
    import time
    logger.info("Waiting 5 seconds for MQTT broker to initialize...")
    time.sleep(5)

    # 2. Trigger CSV Replay
    logger.info(f"Step 2: Injecting Traffic via csv_replay.py (Rate: {config['data']['replay_rate']} pkt/s)...")
    try:
        # Run csv_replay.py
        replay_cmd = [
            sys.executable,
            "infrastructure/csv_replay.py",
            "--csv", test_csv,
            "--broker", "127.0.0.1",
            "--port", "1883",
            "--rate", str(config['data']['replay_rate']),
            "--max_rows", str(args.duration * config['data']['replay_rate'])
        ]
        
        # Run it synchronously so we can observe the simulation
        subprocess.run(replay_cmd)
        
    except KeyboardInterrupt:
        logger.info("\nSimulation interrupted by user.")
    except Exception as e:
        logger.error(f"Traffic replay failed: {e}")
        
    finally:
        # 3. Teardown
        logger.info("Step 3: Tearing down infrastructure...")
        try:
            subprocess.run(
                ["docker", "compose", "-f", "infrastructure/docker-compose.yaml", "down", "-v"],
                check=False
            )
            logger.info("Teardown complete.")
        except Exception as e:
            logger.warning(f"Error during teardown: {e}")
        
        logger.info("=" * 65)
        logger.info("  PHASE 2 EMULATION COMPLETE")
        logger.info("=" * 65)


# ===================================================================
# SUBCOMMAND: evaluate
# ===================================================================
def cmd_evaluate(args: argparse.Namespace) -> None:
    """Run attack injection and collect evaluation metrics."""
    from evaluation.attack_injector import AttackInjector
    from evaluation.metrics_collector import MetricsCollector
    from evaluation.benchmark_report import BenchmarkReport

    config = load_config()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 65)
    logger.info("  ATTACK INJECTION & EVALUATION")
    logger.info("=" * 65)

    metrics_collector = MetricsCollector()
    injector = AttackInjector(config, metrics_collector)

    # Run all attack scenarios
    logger.info("Running attack scenarios...")
    injector.run_all_scenarios()

    # Generate report
    report = BenchmarkReport(metrics_collector, config)

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = RESULTS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir.mkdir(parents=True, exist_ok=True)
    report.generate_full_report(output_dir)
    report.print_summary()

    logger.info(f"\nFull report saved to {output_dir}")


# ===================================================================
# SUBCOMMAND: dashboard
# ===================================================================
def cmd_dashboard(args: argparse.Namespace) -> None:
    """Launch the HITL web dashboard."""
    from dashboard.app import create_app

    config = load_config()

    logger.info("=" * 65)
    logger.info("  HUMAN-IN-THE-LOOP DASHBOARD")
    logger.info("=" * 65)

    app, socketio = create_app(config)

    host = args.host or config["dashboard"]["host"]
    port = args.port or config["dashboard"]["port"]

    logger.info(f"Dashboard starting at http://{host}:{port}")
    logger.info("Press Ctrl+C to stop")

    socketio.run(app, host=host, port=port, debug=args.debug,
                 allow_unsafe_werkzeug=True)


# ===================================================================
# SUBCOMMAND: demo
# ===================================================================
def cmd_demo(args: argparse.Namespace) -> None:
    """Run end-to-end demonstration with synthetic data."""
    import torch
    from data.synthetic_generator import SyntheticIIoMTGenerator
    from data.preprocessor import DataPreprocessor
    from system1.models.cnn_bigru import CNNBiGRU
    from system1.training.trainer import ModelTrainer
    from system1.quantization.quantizer import ModelQuantizer
    from system1.detection.kde_threshold import AdaptiveKDEThreshold
    from system2.reasoning.context_fusion import ContextFusionEngine
    from system2.reasoning.reason_act_loop import ReActLoop
    from system2.reasoning.symbolic_rules import SymbolicRuleEngine
    from system2.mitigation.action_playbook import ActionPlaybook
    from evaluation.metrics_collector import MetricsCollector, compute_ecr, compute_fer, compute_gci, compute_ri2, compute_cas

    config = load_config()
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 65)
    logger.info("  END-TO-END DEMO — Agentic IIoMT Security")
    logger.info("=" * 65)

    # ---- Step 1: Generate Data ----
    logger.info("\n[Step 1/6] Generating synthetic IIoMT data...")
    generator = SyntheticIIoMTGenerator("config/settings.yaml")
    df = generator.generate_combined_dataset(num_samples=20000)
    logger.info(f"  Generated {len(df)} samples")
    logger.info(f"  Class distribution:\n{df['label'].value_counts().to_string()}")

    # ---- Step 2: Preprocess ----
    logger.info("\n[Step 2/6] Preprocessing data...")
    preprocessor = DataPreprocessor("config/settings.yaml")
    result = preprocessor.prepare_pipeline(df, test_ratio=0.2, window_size=1)
    X_train, X_test, y_train, y_test, label_mapping = result
    train_loader, test_loader = preprocessor.get_dataloaders(
        X_train, X_test, y_train, y_test, batch_size=256
    )
    logger.info(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    logger.info(f"  Features: {X_train.shape[-1]} | Classes: {len(label_mapping)}")

    # ---- Step 3: Train ----
    logger.info("\n[Step 3/6] Training CNN-BiGRU model...")
    num_features = X_train.shape[-1]
    model = CNNBiGRU(
        num_features=num_features,
        num_classes=len(label_mapping),
        config_path=Path("config/settings.yaml")
    )
    trainer = ModelTrainer(model, config["system1"]["training"])
    trainer.train(train_loader, test_loader, epochs=10)

    # ---- Step 4: Evaluate + Quantize ----
    logger.info("\n[Step 4/6] Evaluating and quantizing...")
    report = trainer.evaluate(model, test_loader)
    logger.info(f"  FP32 Accuracy: {report.get('accuracy', 'N/A')}")

    per_attack = trainer.evaluate_per_attack(model, test_loader, label_mapping)
    logger.info("  Per-attack performance:")
    for atk, m in per_attack.items():
        logger.info(f"    {atk}: Acc={m['accuracy']:.4f} FPR={m['fpr']:.6f}")

    # --- Agentic Governance Metrics Calculation ---
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    benign_cls = label_mapping.get("Benign", 0)
    policy_ok_count = 0
    total_constrained = len(all_preds)
    total_escalations = 0
    false_escalation_count = 0
    
    segments = np.array_split(np.arange(total_constrained), 5)
    ecr_segments = []
    
    for seg_indices in segments:
        seg_policy_ok = 0
        for i in seg_indices:
            pred = all_preds[i]
            true_label = all_targets[i]
            conf = np.max(all_probs[i])
            if conf >= 0.85:
                seg_policy_ok += 1
                policy_ok_count += 1
            if pred != benign_cls:
                total_escalations += 1
                if true_label == benign_cls:
                    false_escalation_count += 1
        ecr_segments.append(compute_ecr(seg_policy_ok, len(seg_indices)))

    ecr = compute_ecr(policy_ok_count, total_constrained)
    fer = compute_fer(false_escalation_count, total_escalations)
    ri2 = compute_ri2(ecr_segments)
    gci = compute_gci([ecr, 1.0 - fer], [0.6, 0.4])
    cas = compute_cas(ecr, fer, previous_ecr=None, delta_t=0)
    # ----------------------------------------------

    quantizer = ModelQuantizer(Path("config/settings.yaml"))
    fp32_size = quantizer.measure_model_size(model)
    int8_model = quantizer.quantize_dynamic(model)
    int8_size = quantizer.measure_model_size(int8_model)
    logger.info(f"  FP32: {fp32_size:.2f} MB -> INT8: {int8_size:.2f} MB "
                f"({fp32_size/int8_size:.1f}x compression)")

    dummy = torch.randn(1, 1, num_features)
    fp32_lat = quantizer.benchmark_latency(model, dummy)
    int8_lat = quantizer.benchmark_latency(int8_model, dummy)
    logger.info(f"  FP32 latency: {fp32_lat['mean_latency_ms']:.3f} ms -> INT8: {int8_lat['mean_latency_ms']:.3f} ms")

    # ---- Step 5: System 2 Demo ----
    logger.info("\n[Step 5/6] Demonstrating System 2 gateway reasoning...")
    context_engine = ContextFusionEngine(**config["system2"]["risk_metric"], config_path=Path("config/settings.yaml"))
    rule_engine = SymbolicRuleEngine(POLICIES_PATH)
    playbook = ActionPlaybook(POLICIES_PATH)
    react_loop = ReActLoop(
        context_engine=context_engine,
        rule_engine=rule_engine,
        action_playbook=playbook,
        config=config["system2"]["reasoning"],
    )

    # Simulate an alert
    sample_alert = {
        "alert_id": "ALT-001",
        "device_id": "dev-001",
        "device_type": "infusion_pump",
        "attack_type": "DDoS",
        "classifier_confidence": 0.92,
        "anomaly_score": 0.88,
        "timestamp": datetime.now().isoformat(),
        "source_agent": "edge-1",
    }
    logger.info(f"  Simulating alert: {sample_alert['attack_type']} on "
                f"{sample_alert['device_type']}")

    react_result = react_loop.execute(sample_alert)
    logger.info(f"  Risk Score: {react_result.risk_score:.4f}")
    logger.info(f"  Action: {react_result.action_taken}")
    logger.info(f"  Reasoning Latency: {react_result.latency_ms:.2f} ms")
    logger.info(f"  Explanation: {react_result.explanation_nl}")
    logger.info("  Reasoning Trace:")
    for step in react_result.reasoning_trace:
        logger.info(f"    [{step.get('phase', '?')}] {step.get('detail', '')}")

    # ---- Step 6: Summary ----
    logger.info("\n[Step 6/6] Final Summary")
    logger.info("=" * 65)
    logger.info(f"  Detection Accuracy (FP32): {report.get('accuracy', 'N/A')}")
    logger.info(f"  Model Size: {fp32_size:.2f} MB -> {int8_size:.2f} MB")
    logger.info(f"  Edge Latency: {int8_lat['mean_latency_ms']:.3f} ms  "
                f"(target <= {config['system1']['latency']['tau_edge_target']} ms)")
    logger.info(f"  Agent Latency: {react_result.latency_ms:.2f} ms  "
                f"(target <= {config['system2']['latency']['tau_agent_target']} ms)")
    logger.info("--- Agentic Governance Metrics ---")
    logger.info(f"  Ethical Compliance Rate (ECR): {ecr:.4f}")
    logger.info(f"  False Escalation Rate (FER): {fer:.4f}")
    logger.info(f"  Governance Compliance Index (GCI): {gci:.4f}")
    logger.info(f"  Resilience Index (RI2): {ri2:.4f}")
    logger.info(f"  Cyber-Adaptive Score (CAS): {cas:.4f}")
    logger.info("=" * 65)
    logger.info("  [OK] Demo complete!")

    # Save checkpoint
    trainer.save_checkpoint(model, CHECKPOINTS_DIR / "cnn_bigru_fp32.pt")
    with open(CHECKPOINTS_DIR / "label_mapping.json", "w") as f:
        native_mapping = {k: int(v) for k, v in label_mapping.items()}
        json.dump(native_mapping, f, indent=2)


# ===================================================================
# CLI Argument Parser
# ===================================================================
def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="iimt-security",
        description="Cross-Domain Agentic Security for Industrial Medical IoT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py demo                        # End-to-end demo
  python main.py train --data synthetic      # Train on synthetic data
  python main.py quantize                    # Quantize trained model
  python main.py simulate --duration 60      # 60-second simulation
  python main.py evaluate --output results/  # Run evaluation suite
  python main.py dashboard                   # Launch HITL dashboard
        """,
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- train ---
    p_train = subparsers.add_parser("train", help="Train CNN-BiGRU model")
    p_train.add_argument("--data", default="synthetic",
                         help="Data source: 'synthetic' or path to CSV")
    p_train.add_argument("--epochs", type=int, default=None,
                         help="Override training epochs")
    p_train.add_argument("--max-rows", type=int, default=None,
                         help="Maximum rows to load from CSV (for quick tests)")

    # --- quantize ---
    p_quant = subparsers.add_parser("quantize", help="Quantize trained model")
    p_quant.add_argument("--model", default=None,
                         help="Path to FP32 model checkpoint")
    p_quant.add_argument("--skip-pruning", action="store_true",
                         help="Skip channel pruning step")

    # --- simulate ---
    p_sim = subparsers.add_parser("simulate", help="Run full emulated pipeline")
    p_sim.add_argument("--duration", type=int, default=30,
                       help="Simulation duration in seconds (default: 30)")
    p_sim.add_argument("--attacks", default="ddos,spoofing,mitm",
                       help="Comma-separated attack types to inject")

    # --- evaluate ---
    p_eval = subparsers.add_parser("evaluate", help="Run evaluation suite")
    p_eval.add_argument("--output", default=None,
                        help="Output directory for results")

    # --- dashboard ---
    p_dash = subparsers.add_parser("dashboard", help="Launch HITL dashboard")
    p_dash.add_argument("--host", default=None, help="Dashboard host")
    p_dash.add_argument("--port", type=int, default=None, help="Dashboard port")
    p_dash.add_argument("--debug", action="store_true", help="Enable debug mode")

    # --- demo ---
    subparsers.add_parser("demo", help="End-to-end demo with synthetic data")

    return parser


# ===================================================================
# Entry Point
# ===================================================================
def main() -> None:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(args.log_level)

    logger.info("========================================================")
    logger.info("|  Cross-Domain Agentic Security for IIoMT             |")
    logger.info("|  Dual-Process Neuro-Symbolic Architecture            |")
    logger.info("========================================================")

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "train": cmd_train,
        "quantize": cmd_quantize,
        "simulate": cmd_simulate,
        "evaluate": cmd_evaluate,
        "dashboard": cmd_dashboard,
        "demo": cmd_demo,
    }

    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        logger.info("\nOperation interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
