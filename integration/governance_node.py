from __future__ import annotations

from langchain_core.messages import AIMessage

from governance.governance_engine import GovernanceEngine
from integration.governance_adapter import build_governance_input


def governance_node(state: dict) -> dict:
    diagnosis = state.get("diagnosis")
    action_proposal = state.get("action_proposal")
    alert = state.get("alert", {})

    if not diagnosis or not action_proposal:
        return {
            "governance_decision": {
                "error": "Missing diagnosis or action_proposal for governance evaluation."
            },
            "messages": [
                AIMessage(content="[Governance] Missing diagnosis/action proposal. Cannot evaluate.")
            ],
        }

    machine_context = {
        "machine_id": alert.get("machine_id", "UNKNOWN"),
        "current_production_batch": "BATCH-DEMO-01",
        "batch_criticality": "HIGH",
        "operator_present": True,
        "shift_supervisor_available": True,
        "time_since_last_human_review": 0.0,
    }

    gov_input = build_governance_input(
        diagnosis=diagnosis,
        action_proposal=action_proposal,
        machine_context=machine_context,
    )

    engine = GovernanceEngine()

    governance_decision = engine.evaluate(
        gov_input["reasoning_output"],
        gov_input["machine_context"],
        gov_input["policy_overrides"],
    )

    verdict = governance_decision["decision"]["verdict"]

    return {
        "governance_decision": governance_decision,
        "messages": [
            AIMessage(content=f"[Governance] Final verdict: {verdict}")
        ],
    }