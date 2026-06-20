import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import time

logger = logging.getLogger("iimt.system2.reason_act_loop")

@dataclass
class ReActResult:
    action_taken: str
    risk_score: float
    reasoning_trace: List[Dict[str, str]]
    explanation_nl: str
    latency_ms: float

class ReActLoop:
    """Structured Reason-and-Act (ReAct) loop for System 2 Gateway Reasoning Engine."""
    
    def __init__(self, context_engine, rule_engine, action_playbook, config: dict, slm_interface=None):
        self.context_engine = context_engine
        self.rule_engine = rule_engine
        self.action_playbook = action_playbook
        self.config = config
        self.slm_interface = slm_interface
        self.max_iterations = config.get("max_iterations", 5)

    def execute(self, alert_event: dict) -> ReActResult:
        """Execute the ReAct loop for an incoming alert event."""
        start_time = time.perf_counter()
        trace = []
        
        try:
            # 1. OBSERVE
            device_id = alert_event.get("device_id")
            trace.append({"phase": "OBSERVE", "detail": f"Received alert for {device_id}"})
            
            # 2. THINK
            device_id = alert_event.get("device_id", "unknown")
            # Fetch device info from context engine's loaded registry
            device_info = self.context_engine._device_registry.get(device_id, {"id": device_id, "type": "unknown", "criticality": "MEDIUM"})
            
            # Fetch patient context (Mocking EHR integration)
            patient_context = {"patient_id": "P-1234", "active_procedure": "surgery" if device_info.get("criticality") == "LIFE_CRITICAL" else "monitoring"}
            
            context = self.context_engine.aggregate_context(alert_event, device_info, patient_context, [])
            risk_score = getattr(context, "risk_score", 0.0)
            trace.append({"phase": "THINK", "detail": f"Computed risk score: {risk_score:.4f} for {device_info.get('criticality')} device"})
            
            # 3. PLAN
            action_level = self.action_playbook.select_action(risk_score, device_info)
            trace.append({"phase": "PLAN", "detail": f"Selected mitigation level {action_level.level}: {action_level.action_name}"})
            
            # 4. VALIDATE
            validation = self.rule_engine.validate_action(action_level, device_info, context)
            trace.append({"phase": "VALIDATE", "detail": f"Validation result: {validation.is_valid} ({validation.message})"})
            
            # 5. ACT
            action_taken = action_level.action_name if validation.is_valid else validation.suggested_alternative
            trace.append({"phase": "ACT", "detail": f"Final action decided: {action_taken}"})
            
            # 6. EXPLAIN
            if self.slm_interface and self.slm_interface.is_available():
                explanation = self.slm_interface.generate_explanation(alert_event, action_taken, context)
            else:
                explanation = f"Autonomous mitigation: applied {action_taken} due to risk score {risk_score:.2f}."
            trace.append({"phase": "EXPLAIN", "detail": "Generated NL explanation."})
            
            latency = (time.perf_counter() - start_time) * 1000
            
            return ReActResult(
                action_taken=action_taken,
                risk_score=risk_score,
                reasoning_trace=trace,
                explanation_nl=explanation,
                latency_ms=latency
            )
            
        except Exception as e:
            logger.error(f"Error in ReAct loop: {e}", exc_info=True)
            return ReActResult("LOG_ONLY", 0.0, trace, f"Error: {e}", 0.0)
