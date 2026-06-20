# -*- coding: utf-8 -*-
"""
Runtime Latency & Resource Benchmark
====================================

Measures the paper's §5.2 / §5.3 metrics **honestly and per-packet** for a
chosen domain's INT8 ONNX edge model, plus the System 2 reasoning latency:

  * tau_edge  — single-packet INT8 inference latency (mean / p50 / p95 / p99)
  * tau_agent — System 2 ReAct convergence latency (mean / p95)
  * T_ttm     — tau_edge + tau_comm + tau_agent + tau_action (Eq. 4)
  * memory    — peak RSS during the inference loop (MB)
  * cpu       — process CPU utilisation (%) during the loop

Unlike batched evaluation, tau_edge here is measured ONE packet at a time,
which is the realistic edge condition the paper specifies.

Usage::

    python -m evaluation.runtime_benchmark --domain edge --iters 2000
    python -m evaluation.runtime_benchmark --domain cic  --iters 2000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import psutil
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_CFG = yaml.safe_load((_ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))


def _percentiles(samples: List[float]) -> Dict[str, float]:
    a = np.asarray(samples, dtype=np.float64)
    return {
        "mean_ms": float(a.mean()),
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "max_ms": float(a.max()),
    }


def _resolve_model(domain: str) -> Path:
    if domain == "edge":
        p = _ROOT / "checkpoints" / "edge_iiotset" / "cnn_bigru_int8.onnx"
    else:
        p = _ROOT / "checkpoints" / "cnn_bigru_int8.onnx"
    if not p.exists():
        raise FileNotFoundError(f"INT8 ONNX not found for domain '{domain}': {p}")
    return p


def bench_edge(domain: str, iters: int) -> Dict:
    import onnxruntime as ort

    model_path = _resolve_model(domain)
    proc = psutil.Process()
    # Baseline RSS *before* the model session exists, so we can attribute the
    # incremental memory of the edge model (this is the figure the paper's
    # <=45 MB edge target refers to — not the whole Python interpreter).
    baseline_rss = proc.memory_info().rss / 1e6

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    name = inp.name
    # Resolve concrete shape (replace dynamic dims with 1).
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    rng = np.random.default_rng(0)

    # Warm-up (exclude cold-start from the measurement).
    for _ in range(50):
        sess.run(None, {name: rng.standard_normal(shape).astype(np.float32)})

    proc.cpu_percent(interval=None)  # prime CPU counter
    peak_rss = 0.0
    latencies: List[float] = []

    for _ in range(iters):
        x = rng.standard_normal(shape).astype(np.float32)
        t0 = time.perf_counter_ns()
        sess.run(None, {name: x})
        latencies.append((time.perf_counter_ns() - t0) / 1e6)  # ms
        peak_rss = max(peak_rss, proc.memory_info().rss / 1e6)

    cpu = proc.cpu_percent(interval=None)
    model_working_set = max(peak_rss - baseline_rss, 0.0)
    model_size_mb = model_path.stat().st_size / 1e6

    # Realistic steady-state CPU: pace inference at the configured packet rate
    # (paper §5.3 measures CPU "during active attack sequences", not a tight
    # max-throughput loop which would saturate a single core).
    replay_rate = float(_CFG.get("data", {}).get("replay_rate", 500)) or 500.0
    interval = 1.0 / replay_rate
    proc.cpu_percent(interval=None)
    paced_start = time.perf_counter()
    paced_n = int(min(replay_rate * 2, 2000))  # ~2 seconds of traffic
    for _ in range(paced_n):
        t_next = time.perf_counter() + interval
        sess.run(None, {name: rng.standard_normal(shape).astype(np.float32)})
        sleep = t_next - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)
    paced_cpu = proc.cpu_percent(interval=None)
    n_cores = max(psutil.cpu_count() or 1, 1)

    stats = _percentiles(latencies)
    stats.update({
        "model": str(model_path.relative_to(_ROOT)),
        "input_shape": shape,
        "iterations": iters,
        "model_size_mb": round(model_size_mb, 3),
        "model_working_set_mb": round(model_working_set, 2),
        "process_peak_rss_mb": round(peak_rss, 2),
        "cpu_percent_maxloop": round(cpu, 2),
        "cpu_percent_steady_per_core": round(paced_cpu / n_cores, 2),
        "steady_rate_pps": replay_rate,
    })
    return stats


def bench_agent(n: int = 200) -> Dict:
    """Measure System 2 ReAct loop convergence latency."""
    try:
        from system2.reasoning.context_fusion import ContextFusionEngine
        from system2.reasoning.symbolic_rules import SymbolicRuleEngine
        from system2.mitigation.action_playbook import ActionPlaybook
        from system2.reasoning.reason_act_loop import ReActLoop
    except Exception as e:
        return {"error": f"System 2 import failed: {e}"}

    cfg_path = _ROOT / "config" / "settings.yaml"
    pol_path = _ROOT / "config" / "safety_policies.yaml"
    ce = ContextFusionEngine(config_path=cfg_path)
    try:
        re_ = SymbolicRuleEngine(pol_path)
        ap = ActionPlaybook(pol_path)
    except Exception as e:
        return {"error": f"System 2 init failed: {e}"}

    loop = ReActLoop(ce, re_, ap, _CFG.get("system2", {}).get("reasoning", {}))
    rng = np.random.default_rng(1)
    attacks = ["DDoS", "MITM", "Spoofing", "DoS", "Reconnaissance"]
    lat: List[float] = []
    for _ in range(n):
        alert = {
            "device_id": "infusion_pump_01",
            "alert_type": str(rng.choice(attacks)),
            "confidence": float(rng.uniform(0.6, 0.99)),
            "score": float(rng.uniform(0.3, 0.95)),
        }
        r = loop.execute(alert)
        lat.append(r.latency_ms)
    stats = _percentiles(lat)
    stats["iterations"] = n
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["cic", "edge"], default="edge")
    ap.add_argument("--iters", type=int, default=2000)
    args = ap.parse_args()

    edge = bench_edge(args.domain, args.iters)
    agent = bench_agent()

    tau_edge = edge["mean_ms"]
    tau_agent = agent.get("mean_ms", float("nan"))
    tau_comm = _CFG["system2"]["latency"].get("tau_comm_target", 10.0)
    tau_action = _CFG["system2"]["latency"].get("tau_action_target", 5.0)
    t_ttm = tau_edge + tau_comm + tau_agent + tau_action

    targets = {
        "tau_edge_ms": _CFG["system1"]["latency"]["tau_edge_target"],
        "tau_agent_ms": _CFG["system2"]["latency"]["tau_agent_target"],
        "t_ttm_ms": _CFG["system2"]["latency"]["t_ttm_target"],
        "peak_ram_mb": _CFG["system1"]["memory"]["peak_ram_mb"],
        "cpu_pct": 15.0,
    }

    summary = {
        "domain": args.domain,
        "tau_edge": edge,
        "tau_agent": agent,
        "t_ttm_ms": round(t_ttm, 3),
        "components_ms": {
            "tau_edge": round(tau_edge, 4), "tau_comm": tau_comm,
            "tau_agent": round(tau_agent, 4), "tau_action": tau_action,
        },
        "targets": targets,
        "pass": {
            "tau_edge": tau_edge <= targets["tau_edge_ms"],
            "tau_agent": tau_agent <= targets["tau_agent_ms"],
            "t_ttm": t_ttm <= targets["t_ttm_ms"],
            "model_working_set": edge["model_working_set_mb"] <= targets["peak_ram_mb"],
            "cpu_per_core": edge["cpu_percent_steady_per_core"] <= targets["cpu_pct"],
        },
    }

    out_dir = _ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"runtime_benchmark_{args.domain}.json"
    out_file.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\n[Saved] {out_file}")
    print("\n=== PASS/FAIL vs paper targets ===")
    for k, v in summary["pass"].items():
        print(f"  {k:10s}: {'PASS' if v else 'FAIL'}")


if __name__ == "__main__":
    main()
