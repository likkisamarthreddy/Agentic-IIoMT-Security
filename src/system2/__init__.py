# -*- coding: utf-8 -*-
"""
System 2 — Gateway Reasoning Engine
====================================

The deliberative reasoning layer of the Cross-Domain Agentic Security
framework for Industrial Medical IoT (IIoMT).  System 2 receives
coarse-grained alerts from multiple System 1 edge agents, performs
context fusion, symbolic rule validation, graduated mitigation
selection, and optional SLM-augmented reasoning via a structured
Reason-and-Act (ReAct) loop.

Sub-packages
------------
reasoning
    Context fusion, symbolic rule engine, ReAct loop, SLM interface.
mitigation
    Graduated action playbook and simulated SDN controller.

Top-level Modules
-----------------
gateway_agent
    Main orchestrator that wires all System 2 components together and
    communicates with edge agents over MQTT.
"""

from system2.gateway_agent import GatewayAgent
from system2.reasoning.context_fusion import ContextFusionEngine
from system2.reasoning.symbolic_rules import SymbolicRuleEngine
from system2.reasoning.reason_act_loop import ReActLoop
from system2.reasoning.slm_interface import SLMInterface
from system2.mitigation.action_playbook import ActionPlaybook
from system2.mitigation.sdn_controller import SDNController

__all__ = [
    "GatewayAgent",
    "ContextFusionEngine",
    "SymbolicRuleEngine",
    "ReActLoop",
    "SLMInterface",
    "ActionPlaybook",
    "SDNController",
]
