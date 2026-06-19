from integration.governance_adapter import build_governance_input
from governance.governance_engine import GovernanceEngine


diagnosis = {
    "root_cause": "Bearing wear due to overdue lubrication",
    "failure_mode": "bearing_wear",
    "confidence": 0.88,
    "supporting_evidence": ["SOP-BEARING-01", "LOG-M07-2026"],
    "reasoning": "High vibration and temperature pattern matches previous bearing wear incidents."
}

action_proposal = {
    "actions": [
        {
            "action": "inspect_bearing",
            "rationale": "Assess the extent of wear and determine if replacement is necessary",
            "urgency": "high"
        },
        {
            "action": "schedule_maintenance",
            "rationale": "Address overdue lubrication and perform repairs",
            "urgency": "medium"
        }
    ],
    "summary": "Remedial actions for bearing wear due to overdue lubrication"
}

machine_context = {
    "machine_id": "M-07",
    "current_production_batch": "BATCH-DEMO-01",
    "batch_criticality": "HIGH",
    "operator_present": True,
    "shift_supervisor_available": True,
    "time_since_last_human_review": 0.0
}

gov_input = build_governance_input(
    diagnosis=diagnosis,
    action_proposal=action_proposal,
    machine_context=machine_context
)

engine = GovernanceEngine()

result = engine.evaluate(
    gov_input["reasoning_output"],
    gov_input["machine_context"],
    gov_input["policy_overrides"]
)

print("\n=== GOVERNANCE ADAPTER TEST ===")
print("Diagnosis:", diagnosis["failure_mode"])
print("Top action:", action_proposal["actions"][0]["action"])
print("Governance verdict:", result["decision"]["verdict"])
print("Rationale:", result["decision"]["decision_rationale"])
print("Human required:", result["human_approval_request"]["required"])