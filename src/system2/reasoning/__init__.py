# -*- coding: utf-8 -*-
"""
system2.reasoning — Reasoning Sub-package
==========================================

Contains the deliberative reasoning components of the Gateway Agent:

* **ContextFusionEngine** — aggregates edge alerts, device metadata,
  patient context, and historical logs into a unified risk assessment.
* **SymbolicRuleEngine** — evaluates proposed mitigations against
  safety policies defined in ``safety_policies.yaml``.
* **ReActLoop** — structured Reason-and-Act loop that drives the
  observe → think → plan → validate → act → explain cycle.
* **SLMInterface** — optional integration with a Small Language Model
  (via Ollama) for threat assessment and natural-language explanations.
"""

from system2.reasoning.context_fusion import ContextFusionEngine
from system2.reasoning.symbolic_rules import SymbolicRuleEngine
from system2.reasoning.reason_act_loop import ReActLoop
from system2.reasoning.slm_interface import SLMInterface

__all__ = [
    "ContextFusionEngine",
    "SymbolicRuleEngine",
    "ReActLoop",
    "SLMInterface",
]
