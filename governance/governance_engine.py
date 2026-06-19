"""
governance_engine.py
FactoryGuard AI — Governance Layer

This module is 100% deterministic Python. It contains NO LLM calls and NO
probabilistic reasoning. Given the same Reasoning Agent output and machine
context, it will always produce the same verdict. This is intentional and
is the core safety argument of FactoryGuard: reasoning is AI, governance is
auditable code that can never hallucinate.

Implements governance_rules.md v1.0:
  - Section 3: Decision Matrix
  - Section 4: The 8 Policy Checks
  - Section 5: Verdict Priority Order
  - Section 6: Override Conditions
  - Section 7: Audit Logging Requirement

Usage:
    from governance_engine import GovernanceEngine

    engine = GovernanceEngine()
    result = engine.evaluate(reasoning_output, machine_context, policy_overrides)
"""

import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("factoryguard.governance_engine")
logging.basicConfig(level=logging.INFO)

# ──────────────────────────────────────────────────────────────────────
# Constants — Section 1 & 3 of governance_rules.md
# ──────────────────────────────────────────────────────────────────────

RISK_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

ACTION_TYPES = [
    "MONITOR",
    "REDUCE_LOAD",
    "SCHEDULE_MAINTENANCE",
    "IMMEDIATE_SHUTDOWN",
    "EMERGENCY_STOP",
]

# Decision Matrix — Section 3 (authoritative, must match governance_rules.md exactly)
DECISION_MATRIX = {
    "MONITOR":              {"LOW": "AUTO",  "MEDIUM": "AUTO",  "HIGH": "HUMAN", "CRITICAL": "BLOCK"},
    "REDUCE_LOAD":          {"LOW": "AUTO",  "MEDIUM": "HUMAN", "HIGH": "HUMAN", "CRITICAL": "BLOCK"},
    "SCHEDULE_MAINTENANCE": {"LOW": "AUTO",  "MEDIUM": "HUMAN", "HIGH": "HUMAN", "CRITICAL": "HUMAN"},
    "IMMEDIATE_SHUTDOWN":   {"LOW": "HUMAN", "MEDIUM": "HUMAN", "HIGH": "HUMAN", "CRITICAL": "AUTO"},
    "EMERGENCY_STOP":       {"LOW": "BLOCK", "MEDIUM": "HUMAN", "HIGH": "AUTO",  "CRITICAL": "AUTO"},
}

# Verdict priority — Section 5 (higher number = higher priority, wins ties)
VERDICT_PRIORITY = {
    "AUTO_APPROVE": 1,
    "HUMAN_APPROVAL_REQUIRED": 2,
    "BLOCKED": 3,
    "EMERGENCY_ALERT": 4,
}

MATRIX_VERDICT_MAP = {
    "AUTO": "AUTO_APPROVE",
    "HUMAN": "HUMAN_APPROVAL_REQUIRED",
    "BLOCK": "BLOCKED",
}

URGENCY_WINDOW_MINUTES = {
    "IMMEDIATE": 15,
    "WITHIN_1HR": 60,
    "WITHIN_4HR": 240,
    "SCHEDULED": 24 * 60,
}

SAFE_FALLBACK_ACTION = "REDUCE_LOAD"  # Section 4, Check 8 — never EMERGENCY_STOP by timeout


class GovernanceEngine:
    """
    Deterministic policy engine. No model calls. Pure rule evaluation.
    """

    def evaluate(
        self,
        reasoning_output: dict,
        machine_context: Optional[dict] = None,
        policy_overrides: Optional[dict] = None,
        recurrence_count_last_24h: int = 0,
    ) -> dict:
        """
        reasoning_output: dict matching Document 2 (Reasoning Agent output schema)
        machine_context: dict with operator_present, batch_criticality, etc.
        policy_overrides: dict with force_human_review_all, emergency_mode_active, maintenance_window_active
        recurrence_count_last_24h: int — how many times this machine triggered HIGH/CRITICAL in last 24h

        returns: dict matching Governance Layer output schema
        """
        machine_context = machine_context or {}
        policy_overrides = policy_overrides or {}

        request_id = reasoning_output.get("request_id", str(uuid.uuid4()))
        machine_id = reasoning_output.get("machine_id", "UNKNOWN")
        risk_level = reasoning_output.get("risk_assessment", {}).get("overall_risk_level", "LOW")

        recommended_actions = reasoning_output.get("recommended_actions", [])
        top_action = recommended_actions[0] if recommended_actions else {
            "action_type": "MONITOR", "urgency": "SCHEDULED"
        }
        action_type = top_action.get("action_type", "MONITOR")

        checks = []
        candidate_verdicts = []  # list of (verdict_str, rationale, extra) -- highest priority wins

        # ── Base verdict from Decision Matrix (Section 3) ──
        matrix_cell = DECISION_MATRIX.get(action_type, {}).get(risk_level, "HUMAN")
        base_verdict = MATRIX_VERDICT_MAP[matrix_cell]
        candidate_verdicts.append((
            base_verdict,
            f"Decision matrix: action={action_type}, risk={risk_level} -> {matrix_cell}",
            {},
        ))
        checks.append(self._check_result(
            "decision_matrix", "Decision Matrix Lookup", "PASS",
            f"{action_type} @ {risk_level} risk -> {matrix_cell}"
        ))

        # ── Check 1: Safety Gate ──
        safety_level = reasoning_output.get("risk_assessment", {}) \
            .get("risk_dimensions", {}).get("safety_risk", {}).get("level", "LOW")
        if safety_level == "CRITICAL" and action_type not in ("EMERGENCY_STOP", "IMMEDIATE_SHUTDOWN"):
            checks.append(self._check_result(
                "safety_gate", "Safety Gate", "FAIL",
                "safety_risk is CRITICAL but action is not a shutdown/stop type."
            ))
            candidate_verdicts.append(("BLOCKED", "Safety Gate failed: CRITICAL safety risk requires shutdown-class action.", {}))
        else:
            checks.append(self._check_result("safety_gate", "Safety Gate", "PASS", f"safety_risk={safety_level}"))

        # ── Check 2: Confidence Gate ──
        overall_confidence = reasoning_output.get("confidence_score", {}).get("overall", 1.0)
        if overall_confidence < 0.65:
            checks.append(self._check_result(
                "confidence_gate", "Confidence Gate", "FAIL",
                f"overall_confidence={overall_confidence} below 0.65 threshold."
            ))
            candidate_verdicts.append((
                "HUMAN_APPROVAL_REQUIRED",
                f"Confidence Gate failed: overall_confidence={overall_confidence} < 0.65.",
                {},
            ))
        else:
            checks.append(self._check_result(
                "confidence_gate", "Confidence Gate", "PASS", f"overall_confidence={overall_confidence}"
            ))

        # ── Check 3: Novel Pattern Gate ──
        novel_pattern = reasoning_output.get("flags", {}).get("novel_failure_pattern", False)
        if novel_pattern:
            checks.append(self._check_result(
                "novel_pattern_gate", "Novel Pattern Gate", "FAIL",
                "novel_failure_pattern flag is true — no historical match."
            ))
            candidate_verdicts.append((
                "HUMAN_APPROVAL_REQUIRED",
                "Novel Pattern Gate failed: unrecognized failure signature requires human review.",
                {},
            ))
        else:
            checks.append(self._check_result("novel_pattern_gate", "Novel Pattern Gate", "PASS", "Pattern recognized."))

        # ── Check 4: SOP Compliance Gate ──
        # NOTE: in this deterministic engine we treat "any MANDATORY SOP cited" as requiring
        # explicit governance attention rather than silently trusting the LLM's compliance claim.
        sops_cited = reasoning_output.get("evidence_used", {}).get("sops_cited", [])
        mandatory_sop_present = len(sops_cited) > 0  # Reasoning agent only cites SOPs it retrieved
        sop_violation = reasoning_output.get("flags", {}).get("conflicting_evidence", False) and mandatory_sop_present
        if sop_violation:
            checks.append(self._check_result(
                "sop_compliance_gate", "SOP Compliance Gate", "FAIL",
                "Conflicting evidence alongside a mandatory SOP citation — possible non-compliant action."
            ))
            candidate_verdicts.append((
                "BLOCKED",
                "SOP Compliance Gate failed: action may conflict with a mandatory SOP.",
                {},
            ))
        else:
            checks.append(self._check_result(
                "sop_compliance_gate", "SOP Compliance Gate",
                "WARN" if mandatory_sop_present else "PASS",
                f"{len(sops_cited)} SOP(s) cited; no conflict detected." if mandatory_sop_present else "No SOPs cited."
            ))

        # ── Check 5: Operator Presence Gate ──
        urgency = top_action.get("urgency", "SCHEDULED")
        operator_present = machine_context.get("operator_present", True)
        if urgency == "IMMEDIATE" and not operator_present:
            checks.append(self._check_result(
                "operator_presence_gate", "Operator Presence Gate", "FAIL",
                "IMMEDIATE urgency action but no operator present on floor."
            ))
            candidate_verdicts.append((
                "HUMAN_APPROVAL_REQUIRED",
                "Operator Presence Gate failed: escalating to supervisor (no operator present).",
                {"escalation_level": "SUPERVISOR"},
            ))
        else:
            checks.append(self._check_result(
                "operator_presence_gate", "Operator Presence Gate", "PASS",
                f"urgency={urgency}, operator_present={operator_present}"
            ))

        # ── Check 6: Batch Criticality Gate ──
        batch_criticality = machine_context.get("batch_criticality", "LOW")
        if batch_criticality == "CRITICAL" and action_type in ("IMMEDIATE_SHUTDOWN", "EMERGENCY_STOP"):
            checks.append(self._check_result(
                "batch_criticality_gate", "Batch Criticality Gate", "FAIL",
                "CRITICAL batch in progress and action would halt production."
            ))
            candidate_verdicts.append((
                "HUMAN_APPROVAL_REQUIRED",
                "Batch Criticality Gate failed: plant manager visibility required before halting a CRITICAL batch.",
                {"escalation_level": "PLANT_MANAGER"},
            ))
        else:
            checks.append(self._check_result(
                "batch_criticality_gate", "Batch Criticality Gate", "PASS",
                f"batch_criticality={batch_criticality}, action={action_type}"
            ))

        # ── Check 7: Recurrence Gate ──
        if recurrence_count_last_24h >= 3:
            checks.append(self._check_result(
                "recurrence_gate", "Recurrence Gate", "FAIL",
                f"Machine triggered HIGH/CRITICAL {recurrence_count_last_24h} times in last 24h."
            ))
            candidate_verdicts.append((
                "EMERGENCY_ALERT",
                f"Recurrence Gate failed: {recurrence_count_last_24h} HIGH/CRITICAL triggers in 24h — overrides all other verdicts.",
                {},
            ))
        else:
            checks.append(self._check_result(
                "recurrence_gate", "Recurrence Gate", "PASS",
                f"recurrence_count_last_24h={recurrence_count_last_24h}"
            ))

        # Check 8 (Timeout Gate) is handled separately via resolve_timeout(), since it
        # depends on wall-clock waiting state that doesn't exist at evaluation time.
        checks.append(self._check_result(
            "timeout_gate", "Timeout Gate", "PASS",
            "Evaluated only if/when a HUMAN_APPROVAL_REQUIRED verdict times out (see resolve_timeout())."
        ))

        # ── Section 6: Global Overrides ──
        override_applied = False
        override_reason = None
        if policy_overrides.get("force_human_review_all"):
            candidate_verdicts.append(("HUMAN_APPROVAL_REQUIRED", "Override: force_human_review_all is active.", {}))
            override_applied = True
            override_reason = "force_human_review_all"
        elif policy_overrides.get("emergency_mode_active"):
            # downgrade any AUTO_APPROVE verdicts present so far
            for i, (v, r, extra) in enumerate(candidate_verdicts):
                if v == "AUTO_APPROVE":
                    candidate_verdicts[i] = ("HUMAN_APPROVAL_REQUIRED", r + " [downgraded by emergency_mode_active override]", extra)
            override_applied = True
            override_reason = "emergency_mode_active"

        if policy_overrides.get("maintenance_window_active") and action_type == "SCHEDULE_MAINTENANCE":
            candidate_verdicts.append(("AUTO_APPROVE", "Override: maintenance_window_active allows auto-approval of scheduled maintenance.", {}))
            override_applied = True
            override_reason = (override_reason + "+maintenance_window_active") if override_reason else "maintenance_window_active"

        # ── Section 5: resolve final verdict by priority ──
        final_verdict, rationale, extra = max(candidate_verdicts, key=lambda c: VERDICT_PRIORITY[c[0]])

        checks_passed = sum(1 for c in checks if c["result"] == "PASS")
        checks_failed = sum(1 for c in checks if c["result"] == "FAIL")

        result = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "governance_version": "1.0",
            "decision": {
                "verdict": final_verdict,
                "action_approved": action_type if final_verdict == "AUTO_APPROVE" else None,
                "decision_rationale": rationale,
                "override_applied": override_applied,
                "override_reason": override_reason,
            },
            "policy_checks": checks,
            "risk_gates": {
                "safety_gate": "CLOSED" if any(c["check_id"] == "safety_gate" and c["result"] == "FAIL" for c in checks) else "OPEN",
                "compliance_gate": "CLOSED" if any(c["check_id"] == "sop_compliance_gate" and c["result"] == "FAIL" for c in checks) else "OPEN",
                "production_gate": "CLOSED" if any(c["check_id"] == "batch_criticality_gate" and c["result"] == "FAIL" for c in checks) else "OPEN",
                "confidence_gate": "CLOSED" if any(c["check_id"] == "confidence_gate" and c["result"] == "FAIL" for c in checks) else "OPEN",
                "all_gates_open": checks_failed == 0,
            },
            "human_approval_request": self._build_human_request(
                final_verdict, top_action, reasoning_output, extra
            ),
            "audit_log": {
                "log_id": str(uuid.uuid4()),
                "machine_id": machine_id,
                "event_type": final_verdict,
                "risk_level": risk_level,
                "decision": final_verdict,
                "reasoning_agent_confidence": overall_confidence,
                "governance_checks_passed": checks_passed,
                "governance_checks_failed": checks_failed,
                "human_involved": final_verdict in ("HUMAN_APPROVAL_REQUIRED", "EMERGENCY_ALERT"),
                "human_decision": None,
                "human_actor": None,
                "action_executed": final_verdict == "AUTO_APPROVE",
                "execution_timestamp": datetime.now(timezone.utc).isoformat() if final_verdict == "AUTO_APPROVE" else None,
                "outcome": None,
            },
        }

        logger.info(
            f"[{request_id}] machine={machine_id} action={action_type} risk={risk_level} -> {final_verdict}"
        )
        return result

    def resolve_timeout(self, governance_result: dict, minutes_elapsed: float) -> dict:
        """
        Section 4, Check 8 — Timeout Gate.
        Call this when a HUMAN_APPROVAL_REQUIRED verdict has been waiting and no
        human response has arrived. Mutates and returns an updated governance_result.
        """
        decision = governance_result["decision"]
        if decision["verdict"] != "HUMAN_APPROVAL_REQUIRED":
            return governance_result  # nothing to time out

        urgency = governance_result.get("human_approval_request", {}).get("urgency", "WITHIN_1HR")
        window = URGENCY_WINDOW_MINUTES.get(urgency, 60)

        if minutes_elapsed >= window:
            governance_result["policy_checks"].append(self._check_result(
                "timeout_gate", "Timeout Gate", "FAIL",
                f"No human response after {minutes_elapsed} min (window was {window} min). Executing safe fallback."
            ))
            decision["verdict"] = "AUTO_APPROVE"
            decision["action_approved"] = SAFE_FALLBACK_ACTION
            decision["decision_rationale"] += f" | Timeout Gate triggered safe fallback: {SAFE_FALLBACK_ACTION}."
            governance_result["audit_log"]["event_type"] = "TIMEOUT_FALLBACK"
            governance_result["audit_log"]["action_executed"] = True
            governance_result["audit_log"]["execution_timestamp"] = datetime.now(timezone.utc).isoformat()
            logger.info(
                f"[{governance_result['request_id']}] Timeout Gate fired -> safe fallback {SAFE_FALLBACK_ACTION}"
            )
        return governance_result

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _check_result(check_id: str, name: str, result: str, detail: str) -> dict:
        return {"check_id": check_id, "check_name": name, "result": result, "detail": detail}

    @staticmethod
    def _build_human_request(final_verdict: str, top_action: dict, reasoning_output: dict, extra: dict) -> dict:
        if final_verdict not in ("HUMAN_APPROVAL_REQUIRED", "EMERGENCY_ALERT"):
            return {"required": False, "urgency": None, "escalation_level": None,
                     "timeout_action": None, "context_for_human": None}

        urgency = top_action.get("urgency", "WITHIN_1HR")
        risk_level = reasoning_output.get("risk_assessment", {}).get("overall_risk_level", "MEDIUM")

        return {
            "required": True,
            "urgency": urgency,
            "escalation_level": extra.get("escalation_level", "OPERATOR"),
            "timeout_action": f"Safe fallback action ({SAFE_FALLBACK_ACTION}) executes automatically if no response within window.",
            "context_for_human": {
                "summary": reasoning_output.get("root_cause_hypothesis", {}).get("primary_cause", "See reasoning trace."),
                "risk_level": risk_level,
                "recommended_action": top_action.get("action_type"),
                "evidence_summary": reasoning_output.get("root_cause_hypothesis", {})
                    .get("evidence_basis", {}).get("ml_evidence", ""),
                "options": ["APPROVE", "REJECT", "MODIFY", "ESCALATE"],
            },
        }


if __name__ == "__main__":
    # Quick smoke test using a hand-built CRITICAL reasoning output
    import json

    sample_reasoning_output = {
        "request_id": "test-001",
        "machine_id": "M24",
        "risk_assessment": {
            "overall_risk_level": "CRITICAL",
            "risk_score": 88.5,
            "risk_dimensions": {
                "safety_risk": {"level": "CRITICAL", "score": 92, "reason": "High failure probability."},
                "production_risk": {"level": "HIGH", "score": 70, "reason": "Critical batch in progress."},
                "compliance_risk": {"level": "MEDIUM", "score": 60, "reason": "Mandatory SOP applies."},
                "equipment_risk": {"level": "HIGH", "score": 80, "reason": "Tool wear high."},
            },
        },
        "recommended_actions": [
            {"action_type": "IMMEDIATE_SHUTDOWN", "urgency": "IMMEDIATE"}
        ],
        "confidence_score": {"overall": 0.84},
        "evidence_used": {"sops_cited": ["SOP-M07-B"]},
        "flags": {"novel_failure_pattern": False, "conflicting_evidence": False},
        "root_cause_hypothesis": {
            "primary_cause": "Tool Wear Failure indicated by sensor and model evidence.",
            "evidence_basis": {"ml_evidence": "XGBoost model confidence 0.88, failure_probability 0.92"},
        },
    }

    engine = GovernanceEngine()
    result = engine.evaluate(
        sample_reasoning_output,
        machine_context={"operator_present": True, "batch_criticality": "HIGH"},
        policy_overrides={},
        recurrence_count_last_24h=0,
    )
    print(json.dumps(result, indent=2))
