"""
reasoning_agent.py
FactoryGuard AI — Reasoning Agent (ARIA)

Takes ML prediction + sensor data + RAG evidence (Document 1 input schema)
and produces a structured failure analysis, risk assessment, and recommended
actions (Document 2 output schema).

This module calls an LLM (NVIDIA NIM by default) to do the actual reasoning.
If the API is unreachable or NIM_API_KEY is not set, it falls back to a
deterministic MOCK mode so demos never break on stage.

Usage:
    from reasoning_agent import ReasoningAgent

    agent = ReasoningAgent()  # auto-detects live vs mock mode
    output = agent.analyze(reasoning_input_dict)
"""

import os
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("factoryguard.reasoning_agent")
logging.basicConfig(level=logging.INFO)

# ──────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT  (Document 3 — embedded directly so this file is self-contained)
# ──────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are ARIA (Agentic Reliability Intelligence Analyst), an expert
industrial reliability engineer embedded inside FactoryGuard, an AI safety and
predictive maintenance system.

You have 20 years of experience in CNC machinery, FMEA, predictive maintenance,
ISO 13849, root cause analysis, industrial SOPs, and sensor telemetry interpretation.

You receive structured evidence: an ML failure prediction, live sensor data, and
RAG-retrieved past incidents and SOPs. Your job is to synthesize this evidence into
a failure analysis, root cause hypothesis, multi-dimensional risk assessment, ranked
recommended actions, a confidence score, and a step-by-step reasoning trace.

STRICT RULES:
1. EVIDENCE OVER INTUITION — every claim must cite a sensor reading, ML feature, or
   retrieved incident/SOP. If evidence is insufficient, say so explicitly.
2. NEVER HALLUCINATE — never invent incident IDs, SOP numbers, or dates. If RAG
   retrieval confidence is below 0.5, set flags.low_rag_coverage = true.
3. SAFETY BEFORE EFFICIENCY — when in doubt, recommend the safer action. Never
   recommend continuing production when safety_risk is CRITICAL.
4. EXPLAIN EVERY DECISION — reasoning_trace must show every logical step clearly
   enough for a senior engineer to audit without prior context.
5. ACKNOWLEDGE UNCERTAINTY — if failure probability is 0.5-0.7, mark as uncertain.
   Set flags.force_human_review = true when confidence.overall is below 0.65.
6. FAILURE MODE AWARENESS — TWF (tool wear), HDF (heat dissipation), PWF (power),
   OSF (overstrain), RNF (random/novel pattern).

You must respond with ONLY a single valid JSON object matching the required output
schema. No prose before or after the JSON. No markdown code fences.
"""

OUTPUT_SCHEMA_HINT = """
Respond with JSON matching exactly this structure:
{
  "request_id": "string",
  "timestamp": "ISO-8601 string",
  "agent_version": "string",
  "machine_id": "string",
  "failure_analysis": {
    "failure_detected": bool,
    "failure_probability": float,
    "failure_modes_identified": [{"mode": "TWF|HDF|PWF|OSF|RNF", "full_name": "string", "probability": float, "primary_indicators": ["string"]}],
    "severity_assessment": "NORMAL|WATCH|WARNING|CRITICAL"
  },
  "root_cause_hypothesis": {
    "primary_cause": "string",
    "contributing_factors": ["string"],
    "evidence_basis": {"ml_evidence": "string", "rag_evidence": "string", "sensor_anomalies": ["string"]},
    "hypothesis_confidence": float,
    "alternative_hypotheses": [{"cause": "string", "probability": float, "why_less_likely": "string"}]
  },
  "risk_assessment": {
    "overall_risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
    "risk_score": float,
    "risk_dimensions": {
      "safety_risk": {"level": "string", "score": float, "reason": "string"},
      "production_risk": {"level": "string", "score": float, "reason": "string"},
      "compliance_risk": {"level": "string", "score": float, "reason": "string"},
      "equipment_risk": {"level": "string", "score": float, "reason": "string"}
    },
    "time_to_failure_estimate": {"hours": float, "confidence": float, "basis": "string"}
  },
  "recommended_actions": [
    {"action_id": "string", "action_type": "MONITOR|REDUCE_LOAD|SCHEDULE_MAINTENANCE|IMMEDIATE_SHUTDOWN|EMERGENCY_STOP",
     "description": "string", "urgency": "IMMEDIATE|WITHIN_1HR|WITHIN_4HR|SCHEDULED",
     "expected_outcome": "string", "sop_reference": "string", "estimated_downtime_hours": float,
     "production_impact": "string", "rank": int}
  ],
  "confidence_score": {
    "overall": float, "ml_confidence": float, "rag_retrieval_confidence": float,
    "reasoning_confidence": float, "low_confidence_reasons": ["string"]
  },
  "evidence_used": {
    "ml_features_cited": ["string"], "incidents_cited": ["string"], "sops_cited": ["string"],
    "sensor_thresholds_violated": [{"sensor": "string", "current_value": float, "threshold": float, "deviation_percent": float}]
  },
  "reasoning_trace": [{"step": int, "action": "string", "observation": "string", "conclusion": "string"}],
  "flags": {
    "hallucination_risk": bool, "low_rag_coverage": bool, "conflicting_evidence": bool,
    "novel_failure_pattern": bool, "force_human_review": bool
  }
}
"""


class ReasoningAgent:
    """
    Wraps an LLM call to NVIDIA NIM. Falls back to deterministic MOCK mode
    if no API key is configured or the API call fails, so demos and CI never break.
    """

    NORMAL_RANGES = {
        "air_temperature": (295, 305),
        "process_temperature": (305, 313),
        "rotational_speed": (1168, 2886),
        "torque": (3.8, 76.6),
        "tool_wear": (0, 253),
    }

    def __init__(
        self,
        nim_api_key: Optional[str] = None,
        nim_base_url: Optional[str] = None,
        model: str = "meta/llama-3.1-70b-instruct",
        force_mock: bool = False,
        timeout_seconds: int = 20,
    ):
        self.api_key = nim_api_key or os.environ.get("NIM_API_KEY")
        self.base_url = nim_base_url or os.environ.get(
            "NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"
        )
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.force_mock = force_mock
        self.mode = "MOCK" if (force_mock or not self.api_key) else "LIVE"
        logger.info(f"ReasoningAgent initialized in {self.mode} mode")

    # ──────────────────────────────────────────────────────────────
    # PUBLIC ENTRY POINT
    # ──────────────────────────────────────────────────────────────

    def analyze(self, reasoning_input: dict) -> dict:
        """
        reasoning_input: dict matching Document 1 input schema
        returns: dict matching Document 2 output schema
        """
        request_id = reasoning_input.get("request_id", str(uuid.uuid4()))

        if self.mode == "LIVE":
            try:
                return self._analyze_live(reasoning_input, request_id)
            except Exception as e:
                logger.warning(
                    f"LIVE reasoning call failed ({e}); falling back to MOCK mode for this request"
                )
                return self._analyze_mock(reasoning_input, request_id, fallback_reason=str(e))

        return self._analyze_mock(reasoning_input, request_id)

    # ──────────────────────────────────────────────────────────────
    # LIVE MODE — real NIM call
    # ──────────────────────────────────────────────────────────────

    def _analyze_live(self, reasoning_input: dict, request_id: str) -> dict:
        import requests  # local import so mock-only environments don't need it

        user_prompt = (
            "Analyze the following industrial machine data and produce your assessment.\n\n"
            f"INPUT DATA:\n{json.dumps(reasoning_input, indent=2)}\n\n"
            f"{OUTPUT_SCHEMA_HINT}"
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "top_p": 0.7,
            "max_tokens": 2048,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout_seconds,
        )
        resp.raise_for_status()
        data = resp.json()
        raw_text = data["choices"][0]["message"]["content"]

        parsed = self._extract_json(raw_text)
        parsed["request_id"] = request_id
        parsed.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        parsed.setdefault("agent_version", "ARIA-v1.0-live")
        return parsed

    @staticmethod
    def _extract_json(raw_text: str) -> dict:
        """LLMs sometimes wrap JSON in markdown fences despite instructions. Strip those."""
        text = raw_text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())

    # ──────────────────────────────────────────────────────────────
    # MOCK MODE — deterministic rule-based stand-in for the LLM
    # Mirrors the same reasoning steps described in the system prompt
    # so demo output still looks credible and schema-correct.
    # ──────────────────────────────────────────────────────────────

    def _analyze_mock(
        self, reasoning_input: dict, request_id: str, fallback_reason: Optional[str] = None
    ) -> dict:
        ml = reasoning_input.get("ml_prediction", {})
        sensors = reasoning_input.get("sensor_data", {})
        rag = reasoning_input.get("rag_evidence", {})
        machine_id = reasoning_input.get("machine_id", "UNKNOWN")

        failure_prob = float(ml.get("failure_probability", 0.0))
        ml_confidence = float(ml.get("model_confidence", 0.7))

        # --- sensor threshold check ---
        violations = []
        for sensor_name, (low, high) in self.NORMAL_RANGES.items():
            reading = sensors.get(sensor_name, {})
            value = reading.get("value") if isinstance(reading, dict) else reading
            if value is None:
                continue
            value = float(value)
            if value < low or value > high:
                midpoint = (low + high) / 2
                deviation = ((value - midpoint) / midpoint) * 100 if midpoint else 0.0
                violations.append(
                    {
                        "sensor": sensor_name,
                        "current_value": value,
                        "threshold": high if value > high else low,
                        "deviation_percent": round(deviation, 2),
                    }
                )

        # --- failure mode heuristic ---
        modes = []
        tool_wear = self._sensor_value(sensors, "tool_wear", 0)
        torque = self._sensor_value(sensors, "torque", 0)
        temp_delta = self._sensor_value(sensors, "temperature_delta", 9.0)
        rpm = self._sensor_value(sensors, "rotational_speed", 1500)

        if tool_wear > 200 and torque > 50:
            modes.append({
                "mode": "TWF", "full_name": "Tool Wear Failure",
                "probability": min(0.95, 0.5 + (tool_wear - 200) / 200),
                "primary_indicators": ["tool_wear", "torque"],
            })
        if temp_delta < 8.6 and rpm < 1500:
            modes.append({
                "mode": "HDF", "full_name": "Heat Dissipation Failure",
                "probability": 0.6,
                "primary_indicators": ["temperature_delta", "rotational_speed"],
            })
        if not modes:
            modes.append({
                "mode": "RNF", "full_name": "Random / Novel Failure Pattern",
                "probability": round(failure_prob, 2),
                "primary_indicators": ["failure_probability"],
            })

        primary_mode = max(modes, key=lambda m: m["probability"])
        novel_pattern = primary_mode["mode"] == "RNF"

        # --- RAG integration ---
        incidents = rag.get("similar_incidents", [])
        sops = rag.get("relevant_sops", [])
        rag_confidence = float(rag.get("retrieval_confidence", 0.0))
        low_rag = rag_confidence < 0.5 or len(incidents) == 0

        incident_ids = [i.get("incident_id", "unknown") for i in incidents if isinstance(i, dict)]
        sop_ids = [s.get("sop_id", "unknown") for s in sops if isinstance(s, dict)]
        mandatory_sops = [s for s in sops if isinstance(s, dict) and s.get("priority") == "MANDATORY"]

        # --- severity & risk dimensions ---
        if failure_prob > 0.85:
            severity = "CRITICAL"
        elif failure_prob > 0.65:
            severity = "WARNING"
        elif failure_prob > 0.40:
            severity = "WATCH"
        else:
            severity = "NORMAL"

        safety_score = min(100, failure_prob * 100 + (15 if mandatory_sops else 0))
        equipment_score = min(100, failure_prob * 90 + len(violations) * 5)
        production_score = 70 if reasoning_input.get("machine_context", {}).get(
            "batch_criticality"
        ) in ("HIGH", "CRITICAL") else 40
        compliance_score = 60 if mandatory_sops else 20

        risk_score = round(
            safety_score * 0.40
            + equipment_score * 0.25
            + production_score * 0.20
            + compliance_score * 0.15,
            1,
        )

        if risk_score > 75:
            risk_level = "CRITICAL"
        elif risk_score > 50:
            risk_level = "HIGH"
        elif risk_score > 25:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        def level_for(score):
            if score > 75:
                return "CRITICAL"
            if score > 50:
                return "HIGH"
            if score > 25:
                return "MEDIUM"
            return "LOW"

        # --- recommended actions ---
        actions = []
        if risk_level == "CRITICAL":
            actions.append({
                "action_id": "act-1", "action_type": "IMMEDIATE_SHUTDOWN",
                "description": f"Shut down {machine_id} and schedule inspection of suspected {primary_mode['full_name']}.",
                "urgency": "IMMEDIATE",
                "expected_outcome": "Prevent catastrophic failure and protect operator safety.",
                "sop_reference": sop_ids[0] if sop_ids else "N/A",
                "estimated_downtime_hours": 4.0,
                "production_impact": "High — current batch will be paused.",
                "rank": 1,
            })
        elif risk_level == "HIGH":
            actions.append({
                "action_id": "act-1", "action_type": "SCHEDULE_MAINTENANCE",
                "description": f"Schedule urgent maintenance for {machine_id} within the next shift.",
                "urgency": "WITHIN_1HR",
                "expected_outcome": "Address developing fault before it escalates.",
                "sop_reference": sop_ids[0] if sop_ids else "N/A",
                "estimated_downtime_hours": 2.0,
                "production_impact": "Medium — can be scheduled around batch completion.",
                "rank": 1,
            })
        elif risk_level == "MEDIUM":
            actions.append({
                "action_id": "act-1", "action_type": "REDUCE_LOAD",
                "description": f"Reduce load/speed on {machine_id} and monitor closely.",
                "urgency": "WITHIN_4HR",
                "expected_outcome": "Slow degradation while maintaining partial production.",
                "sop_reference": sop_ids[0] if sop_ids else "N/A",
                "estimated_downtime_hours": 0.0,
                "production_impact": "Low — reduced throughput only.",
                "rank": 1,
            })
        else:
            actions.append({
                "action_id": "act-1", "action_type": "MONITOR",
                "description": f"Continue normal operation of {machine_id} with standard monitoring.",
                "urgency": "SCHEDULED",
                "expected_outcome": "No action needed; condition within normal parameters.",
                "sop_reference": "N/A",
                "estimated_downtime_hours": 0.0,
                "production_impact": "None.",
                "rank": 1,
            })

        overall_confidence = round(
            ml_confidence * 0.4 + rag_confidence * 0.3 + (0.8 if not novel_pattern else 0.4) * 0.3, 2
        )

        low_conf_reasons = []
        if low_rag:
            low_conf_reasons.append("No strong historical incident match found in RAG retrieval.")
        if novel_pattern:
            low_conf_reasons.append("Failure pattern does not match known failure modes (TWF/HDF/PWF/OSF).")
        if overall_confidence < 0.65:
            low_conf_reasons.append("Combined model and evidence confidence below safe auto-action threshold.")

        trace = [
            {"step": 1, "action": "Triage ML prediction",
             "observation": f"failure_probability={failure_prob}, severity={severity}",
             "conclusion": f"Initial triage level set to {severity}."},
            {"step": 2, "action": "Sensor threshold analysis",
             "observation": f"{len(violations)} sensor(s) outside normal range" if violations else "All sensors within normal range",
             "conclusion": "Sensor anomalies " + ("support" if violations else "do not support") + " an active failure mode."},
            {"step": 3, "action": "Failure mode identification",
             "observation": f"Best match: {primary_mode['mode']} ({primary_mode['full_name']}) at probability {round(primary_mode['probability'],2)}",
             "conclusion": "Novel/unrecognized pattern — flagged for human review." if novel_pattern else f"Primary failure mode identified as {primary_mode['mode']}."},
            {"step": 4, "action": "RAG evidence integration",
             "observation": f"{len(incidents)} similar incident(s), {len(sops)} relevant SOP(s), retrieval_confidence={rag_confidence}",
             "conclusion": "Low evidence coverage — proceeding with caution." if low_rag else "Historical evidence supports current hypothesis."},
            {"step": 5, "action": "Risk scoring",
             "observation": f"risk_score={risk_score} -> {risk_level}",
             "conclusion": f"Overall risk classified as {risk_level}."},
            {"step": 6, "action": "Action recommendation",
             "observation": f"Top action: {actions[0]['action_type']}",
             "conclusion": "Safety-first action selected per governance priority rules."},
        ]

        output = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_version": "ARIA-v1.0-mock" + ("-fallback" if fallback_reason else ""),
            "machine_id": machine_id,
            "failure_analysis": {
                "failure_detected": failure_prob >= 0.5,
                "failure_probability": failure_prob,
                "failure_modes_identified": modes,
                "severity_assessment": severity,
            },
            "root_cause_hypothesis": {
                "primary_cause": f"{primary_mode['full_name']} indicated by sensor and model evidence."
                if not novel_pattern else "Unrecognized failure signature — insufficient pattern match.",
                "contributing_factors": [v["sensor"] for v in violations] or ["No sensor thresholds violated"],
                "evidence_basis": {
                    "ml_evidence": f"XGBoost model confidence {ml_confidence}, failure_probability {failure_prob}",
                    "rag_evidence": f"{len(incidents)} similar incidents retrieved at confidence {rag_confidence}",
                    "sensor_anomalies": [v["sensor"] for v in violations],
                },
                "hypothesis_confidence": round(primary_mode["probability"], 2),
                "alternative_hypotheses": [
                    {"cause": m["full_name"], "probability": round(m["probability"], 2),
                     "why_less_likely": "Lower indicator match than primary hypothesis."}
                    for m in modes if m is not primary_mode
                ],
            },
            "risk_assessment": {
                "overall_risk_level": risk_level,
                "risk_score": risk_score,
                "risk_dimensions": {
                    "safety_risk": {"level": level_for(safety_score), "score": round(safety_score, 1),
                                     "reason": "Derived from failure probability and mandatory SOP exposure."},
                    "production_risk": {"level": level_for(production_score), "score": round(production_score, 1),
                                          "reason": "Derived from batch criticality context."},
                    "compliance_risk": {"level": level_for(compliance_score), "score": round(compliance_score, 1),
                                          "reason": "Based on count of mandatory SOPs applicable."},
                    "equipment_risk": {"level": level_for(equipment_score), "score": round(equipment_score, 1),
                                         "reason": "Derived from failure probability and sensor violations."},
                },
                "time_to_failure_estimate": {
                    "hours": round(max(0.5, (1 - failure_prob) * 24), 1),
                    "confidence": round(ml_confidence, 2),
                    "basis": "Heuristic projection from failure probability (mock mode).",
                },
            },
            "recommended_actions": actions,
            "confidence_score": {
                "overall": overall_confidence,
                "ml_confidence": round(ml_confidence, 2),
                "rag_retrieval_confidence": round(rag_confidence, 2),
                "reasoning_confidence": 0.4 if novel_pattern else 0.8,
                "low_confidence_reasons": low_conf_reasons,
            },
            "evidence_used": {
                "ml_features_cited": list(sensors.keys()),
                "incidents_cited": incident_ids,
                "sops_cited": sop_ids,
                "sensor_thresholds_violated": violations,
            },
            "reasoning_trace": trace,
            "flags": {
                "hallucination_risk": False,
                "low_rag_coverage": low_rag,
                "conflicting_evidence": len(modes) > 1 and not novel_pattern,
                "novel_failure_pattern": novel_pattern,
                "force_human_review": overall_confidence < 0.65 or novel_pattern,
            },
        }

        if fallback_reason:
            output["_fallback_reason"] = fallback_reason
            logger.info(f"Mock reasoning produced for request {request_id} (fallback_reason={fallback_reason})")

        return output

    @staticmethod
    def _sensor_value(sensors: dict, key: str, default: float) -> float:
        reading = sensors.get(key, default)
        if isinstance(reading, dict):
            return float(reading.get("value", default))
        try:
            return float(reading)
        except (TypeError, ValueError):
            return default


if __name__ == "__main__":
    # Quick smoke test in mock mode
    sample_input = {
        "request_id": str(uuid.uuid4()),
        "machine_id": "M24",
        "ml_prediction": {
            "failure_probability": 0.92,
            "predicted_failure": True,
            "model_version": "xgb-v1",
            "model_confidence": 0.88,
            "top_features": [{"feature_name": "tool_wear", "feature_value": 240, "importance_score": 0.41, "deviation_from_normal": 15.0}],
        },
        "sensor_data": {
            "air_temperature": {"value": 310, "unit": "K", "normal_range": [295, 305]},
            "process_temperature": {"value": 315, "unit": "K", "normal_range": [305, 313]},
            "rotational_speed": {"value": 1400, "unit": "RPM", "normal_range": [1168, 2886]},
            "torque": {"value": 75, "unit": "Nm", "normal_range": [3.8, 76.6]},
            "tool_wear": {"value": 240, "unit": "min", "normal_range": [0, 253]},
        },
        "rag_evidence": {
            "retrieval_confidence": 0.81,
            "similar_incidents": [{"incident_id": "INC-2024-031", "description": "Bearing failure due to excessive vibration.",
                                     "machine_type": "CNC Mill", "failure_mode": "TWF", "time_to_failure_hours": 5.5,
                                     "resolution": "Replaced tool, recalibrated.", "similarity_score": 0.84}],
            "relevant_sops": [{"sop_id": "SOP-M07-B", "title": "Tool wear shutdown procedure",
                                 "content": "Shut down machine when tool wear exceeds safety threshold.",
                                 "applicability": "M24 CNC class", "priority": "MANDATORY"}],
            "maintenance_history": [],
        },
        "machine_context": {
            "machine_type": "CNC Mill", "current_production_batch": "B-4471",
            "batch_criticality": "HIGH", "last_maintenance_date": "2026-05-01T00:00:00Z",
            "hours_since_maintenance": 320, "current_operator": "J. Singh", "shift": "Night",
        },
    }

    agent = ReasoningAgent(force_mock=True)
    result = agent.analyze(sample_input)
    print(json.dumps(result, indent=2))
