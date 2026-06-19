from __future__ import annotations

from uuid import uuid4


ACTION_MAP = {
    "inspect_bearing": "SCHEDULE_MAINTENANCE",
    "schedule_maintenance": "SCHEDULE_MAINTENANCE",
    "reduce_load": "REDUCE_LOAD",
    "monitor": "MONITOR",
    "immediate_shutdown": "IMMEDIATE_SHUTDOWN",
    "emergency_stop": "EMERGENCY_STOP",
}


URGENCY_MAP = {
    "low": "SCHEDULED",
    "medium": "WITHIN_4HR",
    "high": "WITHIN_1HR",
    "critical": "IMMEDIATE",
}


def map_risk_from_confidence(confidence: float, urgency: str) -> str:
    if urgency == "critical":
        return "CRITICAL"
    if urgency == "high" and confidence >= 0.75:
        return "HIGH"
    if urgency in ["medium", "high"]:
        return "MEDIUM"
    return "LOW"


def build_governance_input(
    diagnosis: dict,
    action_proposal: dict,
    machine_context: dict | None = None,
    policy_overrides: dict | None = None,
) -> dict:
    machine_context = machine_context or {
        "machine_id": "UNKNOWN",
        "current_production_batch": "BATCH-DEMO",
        "batch_criticality": "HIGH",
        "operator_present": True,
        "shift_supervisor_available": True,
        "time_since_last_human_review": 0.0,
    }

    policy_overrides = policy_overrides or {
        "force_human_review_all": False,
        "emergency_mode_active": False,
        "maintenance_window_active": False,
    }

    actions = action_proposal.get("actions", [])
    top_action = actions[0] if actions else {
        "action": "monitor",
        "rationale": "No action proposed, continue monitoring.",
        "urgency": "low",
    }

    raw_action = top_action.get("action", "monitor")
    raw_urgency = top_action.get("urgency", "low")
    confidence = float(diagnosis.get("confidence", 0.5))

    action_type = ACTION_MAP.get(raw_action, "SCHEDULE_MAINTENANCE")
    urgency = URGENCY_MAP.get(raw_urgency, "WITHIN_4HR")
    risk_level = map_risk_from_confidence(confidence, raw_urgency)

    reasoning_output = {
        "request_id": str(uuid4()),
        "machine_id": machine_context.get("machine_id", "UNKNOWN"),
        "failure_analysis": {
            "failure_detected": risk_level in ["MEDIUM", "HIGH", "CRITICAL"],
            "failure_probability": confidence,
            "failure_modes_identified": [
                {
                    "mode": diagnosis.get("failure_mode", "unknown"),
                    "full_name": diagnosis.get("failure_mode", "unknown"),
                    "probability": confidence,
                    "primary_indicators": diagnosis.get("supporting_evidence", []),
                }
            ],
            "severity_assessment": risk_level,
        },
        "root_cause_hypothesis": {
            "primary_cause": diagnosis.get("root_cause", "Unknown root cause"),
            "contributing_factors": diagnosis.get("supporting_evidence", []),
            "evidence_basis": {
                "ml_evidence": "Provided by anomaly detection module",
                "rag_evidence": ", ".join(diagnosis.get("supporting_evidence", [])),
                "sensor_anomalies": [],
            },
            "hypothesis_confidence": confidence,
            "alternative_hypotheses": [],
        },
        "risk_assessment": {
            "overall_risk_level": risk_level,
            "risk_score": {
                "LOW": 20,
                "MEDIUM": 45,
                "HIGH": 70,
                "CRITICAL": 90,
            }[risk_level],
            "risk_dimensions": {
                "safety_risk": {
                    "level": risk_level,
                    "score": 90 if risk_level == "CRITICAL" else 60,
                    "reason": "Mapped from diagnosis confidence and action urgency.",
                },
                "production_risk": {
                    "level": risk_level,
                    "score": 70 if risk_level in ["HIGH", "CRITICAL"] else 40,
                    "reason": "Production impact estimated from proposed action.",
                },
                "compliance_risk": {
                    "level": "MEDIUM",
                    "score": 50,
                    "reason": "Requires SOP review before execution.",
                },
                "equipment_risk": {
                    "level": risk_level,
                    "score": 80 if risk_level in ["HIGH", "CRITICAL"] else 40,
                    "reason": diagnosis.get("reasoning", ""),
                },
            },
        },
        "recommended_actions": [
            {
                "action_id": "ACT-001",
                "action_type": action_type,
                "description": top_action.get("rationale", action_proposal.get("summary", "")),
                "urgency": urgency,
                "expected_outcome": action_proposal.get("summary", ""),
                "sop_reference": ", ".join(diagnosis.get("supporting_evidence", [])),
                "estimated_downtime_hours": 1.0,
                "production_impact": "Requires operational review.",
                "rank": 1,
            }
        ],
        "confidence_score": {
            "overall": confidence,
            "ml_confidence": confidence,
            "rag_retrieval_confidence": confidence,
            "reasoning_confidence": confidence,
            "low_confidence_reasons": [] if confidence >= 0.65 else ["Diagnosis confidence below governance threshold."],
        },
        "evidence_used": {
            "ml_features_cited": [],
            "incidents_cited": diagnosis.get("supporting_evidence", []),
            "sops_cited": diagnosis.get("supporting_evidence", []),
            "sensor_thresholds_violated": [],
        },
        "reasoning_trace": [
            {
                "step": 1,
                "action": "Read diagnosis and action proposal",
                "observation": diagnosis.get("reasoning", ""),
                "conclusion": f"Mapped proposed action '{raw_action}' to governance action '{action_type}'.",
            }
        ],
        "flags": {
            "hallucination_risk": False,
            "low_rag_coverage": len(diagnosis.get("supporting_evidence", [])) == 0,
            "conflicting_evidence": False,
            "novel_failure_pattern": len(diagnosis.get("supporting_evidence", [])) == 0,
            "force_human_review": confidence < 0.65,
        },
    }

    return {
        "request_id": reasoning_output["request_id"],
        "reasoning_output": reasoning_output,
        "machine_context": machine_context,
        "policy_overrides": policy_overrides,
    }