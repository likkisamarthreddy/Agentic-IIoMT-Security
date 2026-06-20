# -*- coding: utf-8 -*-
"""
SLM (Small Language Model) Interface
=====================================

Optional integration with a local SLM (e.g. **Phi-3 Mini**) served
via `Ollama <https://ollama.ai>`_.  Provides:

* **Threat assessment** — structured prompt → risk analysis.
* **Natural-language explanations** — human-readable summaries of
  threats and mitigations for clinicians (HITL).

Falls back gracefully to deterministic rule-based explanations when the
SLM is unavailable or disabled in ``settings.yaml``.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

# HTTP library — optional at import time so the module can be imported
# even if ``requests`` is not yet installed.
try:
    import requests  # type: ignore[import-untyped]
except ImportError:
    requests = None  # type: ignore[assignment]
    logger.debug("requests library not available — SLM calls will be disabled")


# ------------------------------------------------------------------ #
#  Prompt templates                                                   #
# ------------------------------------------------------------------ #

_TEMPLATES: Dict[str, str] = {
    "threat_assessment": (
        "You are a cybersecurity analyst for a medical IoT network.\n"
        "Analyze the following security alert and provide a structured threat assessment.\n\n"
        "ALERT DATA:\n"
        "- Device ID: {device_id}\n"
        "- Device Type: {device_type}\n"
        "- Alert Type: {alert_type}\n"
        "- Classifier Confidence: {confidence:.2f}\n"
        "- Risk Score: {risk_score:.3f}\n"
        "- Device Criticality: {criticality}\n"
        "- Historical Alert Density: {historical_density:.3f}\n"
        "- Timestamp: {timestamp}\n\n"
        "Respond in JSON with keys: threat_level (LOW/MEDIUM/HIGH/CRITICAL), "
        "assessment (1-2 sentences), recommended_action (string), "
        "confidence (0-1 float)."
    ),

    "explanation": (
        "You are a clinical security advisor explaining a network security "
        "event to a clinician.\n\n"
        "THREAT SUMMARY:\n"
        "- Device: {device_name} ({device_type})\n"
        "- Attack Type: {alert_type}\n"
        "- Risk Score: {risk_score:.1%}\n\n"
        "ACTION TAKEN:\n"
        "- Mitigation: {action_name} (Level {action_level})\n"
        "- Description: {action_description}\n\n"
        "Write a concise, jargon-free explanation (3-5 sentences) suitable "
        "for a clinician.  Focus on patient-safety impact and what the "
        "clinician should do next."
    ),
}


# ------------------------------------------------------------------ #
#  SLM Interface                                                      #
# ------------------------------------------------------------------ #

class SLMInterface:
    """Interface to a local Small Language Model served by Ollama.

    Args:
        config: Either a dict with SLM configuration keys or a
            :class:`pathlib.Path` to ``settings.yaml``.  Expected
            keys: ``enabled``, ``model_name``, ``ollama_host``,
            ``timeout_ms``, ``max_tokens``.
    """

    def __init__(self, config: Dict[str, Any] | Path) -> None:
        if isinstance(config, Path):
            config = self._load_config(config)

        self.enabled: bool = bool(config.get("enabled", False))
        self.model_name: str = str(config.get("model_name", "phi3:mini"))
        self.ollama_host: str = str(config.get("ollama_host", "http://localhost:11434"))
        self.timeout_sec: float = float(config.get("timeout_ms", 5000)) / 1000.0
        self.max_tokens: int = int(config.get("max_tokens", 512))

        self.response_cache: Dict[str, Any] = {}

        logger.info(
            "SLMInterface initialised (enabled=%s, model=%s, host=%s)",
            self.enabled,
            self.model_name,
            self.ollama_host,
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def is_available(self) -> bool:
        """Check whether the Ollama server is running and the model is loaded.

        Returns:
            ``True`` if the SLM is enabled, the ``requests`` library is
            present, and the Ollama health-check succeeds.
        """
        if not self.enabled or requests is None:
            return False

        try:
            resp = requests.get(
                f"{self.ollama_host}/api/tags",
                timeout=2.0,
            )
            if resp.status_code != 200:
                logger.debug("Ollama health-check returned %d", resp.status_code)
                return False

            models = resp.json().get("models", [])
            available = any(
                m.get("name", "").startswith(self.model_name.split(":")[0])
                for m in models
            )
            if not available:
                logger.debug("Model %s not found in Ollama", self.model_name)
            return available
        except Exception as exc:  # noqa: BLE001
            logger.debug("Ollama not reachable: %s", exc)
            return False

    def _cache_key(self, context: Dict[str, Any]) -> str:
        """Generate MD5 hash for cache key based on core alert signature."""
        import hashlib
        key_fields = {
            'alert_type': context.get('alert_type'),
            'risk_score': round(float(context.get('risk_score', 0)), 1), # Quantize risk for cache hits
            'device_type': context.get('device_type')
        }
        return hashlib.md5(json.dumps(key_fields, sort_keys=True).encode()).hexdigest()

    def assess_threat(
        self,
        alert_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Send a structured prompt to the SLM for threat assessment.

        Falls back to a deterministic assessment when the SLM is
        unavailable.
        """
        if not self.is_available():
            return self._rule_based_assessment(alert_context)
            
        c_key = self._cache_key(alert_context) + "_assess"
        if c_key in self.response_cache:
            return self.response_cache[c_key]

        prompt = self._build_prompt("threat_assessment", alert_context)
        raw = self._call_ollama(prompt)
        if raw is None:
            return self._rule_based_assessment(alert_context)

        parsed = self._parse_response(raw)
        parsed["source"] = "slm"
        self.response_cache[c_key] = parsed
        return parsed

    def generate_explanation(
        self,
        alert: Dict[str, Any],
        action: Dict[str, Any],
        context: Dict[str, Any],
    ) -> str:
        """Generate a natural-language explanation of a mitigation action."""
        merged = {**context, **alert, **action}

        if not self.is_available():
            return self._rule_based_explanation(merged)

        c_key = self._cache_key(merged) + "_explain"
        if c_key in self.response_cache:
            return self.response_cache[c_key]

        prompt = self._build_prompt("explanation", merged)
        raw = self._call_ollama(prompt)
        if raw is None:
            return self._rule_based_explanation(merged)

        result = raw.strip()
        self.response_cache[c_key] = result
        return result

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _build_prompt(
        self,
        template_name: str,
        context: Dict[str, Any],
    ) -> str:
        """Construct a structured prompt from a named template.

        Args:
            template_name: Key in the ``_TEMPLATES`` dict.
            context: Placeholder values.

        Returns:
            Formatted prompt string.
        """
        template = _TEMPLATES.get(template_name, "")
        if not template:
            logger.warning("Unknown prompt template: %s", template_name)
            return str(context)

        try:
            return template.format_map(_SafeDict(context))
        except Exception as exc:  # noqa: BLE001
            logger.error("Prompt formatting failed: %s", exc)
            return template

    def _parse_response(self, raw_response: str) -> Dict[str, Any]:
        """Extract structured action data from SLM text output.

        Attempts JSON parsing first; falls back to key-value extraction.

        Args:
            raw_response: Raw text from the SLM.

        Returns:
            Parsed dict with threat assessment keys.
        """
        # Try to find a JSON block in the response
        try:
            # Look for ```json ... ``` blocks
            if "```json" in raw_response:
                json_str = raw_response.split("```json")[1].split("```")[0]
            elif "{" in raw_response:
                start = raw_response.index("{")
                end = raw_response.rindex("}") + 1
                json_str = raw_response[start:end]
            else:
                json_str = raw_response

            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError, IndexError):
            logger.debug("Could not parse SLM JSON response; returning raw")
            return {
                "threat_level": "UNKNOWN",
                "assessment": raw_response.strip()[:200],
                "recommended_action": "REVIEW",
                "confidence": 0.5,
            }

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Call the Ollama REST API to generate text.

        Args:
            prompt: Input prompt text.

        Returns:
            Generated text string, or ``None`` on failure.
        """
        if requests is None:
            return None

        url = f"{self.ollama_host}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json" if "json" in prompt.lower() else "",
            "options": {
                "num_predict": self.max_tokens,
                "temperature": 0.1,
            },
        }

        try:
            t0 = time.perf_counter()
            resp = requests.post(url, json=payload, timeout=self.timeout_sec)
            latency_ms = (time.perf_counter() - t0) * 1000

            if resp.status_code != 200:
                logger.warning(
                    "Ollama returned %d: %s", resp.status_code, resp.text[:200],
                )
                return None

            body = resp.json()
            text = body.get("response", "")
            logger.info(
                "SLM response received in %.1fms (%d chars)",
                latency_ms,
                len(text),
            )
            return text
        except Exception as exc:  # noqa: BLE001
            logger.error("Ollama request failed: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    #  Deterministic fallbacks                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _rule_based_assessment(ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Deterministic threat assessment when SLM is unavailable.

        Args:
            ctx: Alert context dict.

        Returns:
            Assessment dict.
        """
        risk = float(ctx.get("risk_score", 0.0))
        if risk >= 0.85:
            level = "CRITICAL"
        elif risk >= 0.7:
            level = "HIGH"
        elif risk >= 0.5:
            level = "MEDIUM"
        else:
            level = "LOW"

        return {
            "threat_level": level,
            "assessment": (
                f"Rule-based assessment: risk score {risk:.3f} "
                f"for {ctx.get('alert_type', 'unknown')} on "
                f"device {ctx.get('device_id', 'unknown')}."
            ),
            "recommended_action": ctx.get("recommended_action", "MONITOR"),
            "confidence": min(risk + 0.1, 1.0),
            "source": "rule_based",
        }

    @staticmethod
    def _rule_based_explanation(ctx: Dict[str, Any]) -> str:
        """Deterministic natural-language explanation fallback.

        Args:
            ctx: Merged alert/action/context dict.

        Returns:
            Human-readable explanation string.
        """
        device_name = ctx.get("device_name", ctx.get("device_id", "A device"))
        alert_type = ctx.get("alert_type", "security event")
        risk_score = float(ctx.get("risk_score", 0.0))
        action_name = ctx.get("action_name", "monitoring")

        return (
            f"A {alert_type} event was detected on {device_name} "
            f"with a risk score of {risk_score:.1%}. "
            f"The system has applied '{action_name}' mitigation to protect "
            f"the network while preserving critical patient-care data streams. "
            f"Please review the alert dashboard for details and approve or "
            f"override the action as needed."
        )

    @staticmethod
    def _load_config(config_path: Path) -> Dict[str, Any]:
        """Load SLM section from settings.yaml.

        Args:
            config_path: Path to settings.yaml.

        Returns:
            SLM config dict.
        """
        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
            return cfg.get("system2", {}).get("slm", {})
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load SLM config: %s", exc)
            return {}


# ------------------------------------------------------------------ #
#  Helper: safe dict for format_map                                    #
# ------------------------------------------------------------------ #

class _SafeDict(dict):  # type: ignore[type-arg]
    """Dict subclass that returns the key itself for missing format keys."""

    def __missing__(self, key: str) -> str:
        return f"{{{key}}}"


# ------------------------------------------------------------------ #
#  Standalone smoke test                                              #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    project_root = Path(__file__).resolve().parents[3]
    settings = project_root / "config" / "settings.yaml"

    slm = SLMInterface(settings)
    print(f"SLM available: {slm.is_available()}")

    ctx = {
        "device_id": "dev-001",
        "device_type": "infusion_pump",
        "alert_type": "DDoS",
        "confidence": 0.92,
        "risk_score": 0.78,
        "criticality": "LIFE_CRITICAL",
        "historical_density": 0.4,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    assessment = slm.assess_threat(ctx)
    print(f"Assessment: {assessment}")

    explanation = slm.generate_explanation(
        alert={"alert_type": "DDoS", "risk_score": 0.78},
        action={"action_name": "MICRO_SEGMENT", "action_level": 2,
                "action_description": "Isolate to read-only VLAN"},
        context={"device_name": "Infusion Pump A", "device_type": "infusion_pump"},
    )
    print(f"Explanation: {explanation}")
