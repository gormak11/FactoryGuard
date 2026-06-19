"""
test_pipeline.py
FactoryGuard AI — End-to-end pipeline test

Runs sample inputs through ReasoningAgent (mock mode, no API key needed) and
then through GovernanceEngine, printing the full trace so you can visually
verify the reasoning -> governance handoff matches the schemas and rules.

This is NOT a pytest-style assertion suite (though a few sanity asserts are
included) — it's primarily a readable demo script you can run live in front
of judges or teammates to show the pipeline working end to end.

Run:
    python test_pipeline.py
"""

import json
import uuid
from reasoning_agent.reasoning_agent import ReasoningAgent
from governance.governance_engine import GovernanceEngine


def make_input(
    machine_id="M24",
    failure_probability=0.92,
    model_confidence=0.88,
    air_temp=310,
    process_temp=315,
    rpm=1400,
    torque=75,
    tool_wear=240,
    retrieval_confidence=0.81,
    similar_incidents=None,
    relevant_sops=None,
    batch_criticality="HIGH",
):
    """Builds a Document-1-shaped reasoning input. Defaults mirror Yashraj's sample ML output."""
    if similar_incidents is None:
        similar_incidents = [{
            "incident_id": "INC-2024-031",
            "description": "Bearing failure due to excessive vibration.",
            "machine_type": "CNC Mill",
            "failure_mode": "TWF",
            "time_to_failure_hours": 5.5,
            "resolution": "Replaced tool, recalibrated.",
            "similarity_score": 0.84,
        }]
    if relevant_sops is None:
        relevant_sops = [{
            "sop_id": "SOP-M07-B",
            "title": "Tool wear shutdown procedure",
            "content": "Shut down machine when tool wear exceeds safety threshold.",
            "applicability": "M24 CNC class",
            "priority": "MANDATORY",
        }]

    return {
        "request_id": str(uuid.uuid4()),
        "machine_id": machine_id,
        "ml_prediction": {
            "failure_probability": failure_probability,
            "predicted_failure": failure_probability >= 0.5,
            "model_version": "xgb-v1",
            "model_confidence": model_confidence,
            "top_features": [
                {"feature_name": "tool_wear", "feature_value": tool_wear,
                 "importance_score": 0.41, "deviation_from_normal": 15.0}
            ],
        },
        "sensor_data": {
            "air_temperature": {"value": air_temp, "unit": "K", "normal_range": [295, 305]},
            "process_temperature": {"value": process_temp, "unit": "K", "normal_range": [305, 313]},
            "rotational_speed": {"value": rpm, "unit": "RPM", "normal_range": [1168, 2886]},
            "torque": {"value": torque, "unit": "Nm", "normal_range": [3.8, 76.6]},
            "tool_wear": {"value": tool_wear, "unit": "min", "normal_range": [0, 253]},
        },
        "rag_evidence": {
            "retrieval_confidence": retrieval_confidence,
            "similar_incidents": similar_incidents,
            "relevant_sops": relevant_sops,
            "maintenance_history": [],
        },
        "machine_context": {
            "machine_type": "CNC Mill",
            "current_production_batch": "B-4471",
            "batch_criticality": batch_criticality,
            "last_maintenance_date": "2026-05-01T00:00:00Z",
            "hours_since_maintenance": 320,
            "current_operator": "J. Singh",
            "shift": "Night",
        },
    }


def run_scenario(title, reasoning_input, machine_context_override=None, policy_overrides=None,
                  recurrence_count_last_24h=0, agent=None, engine=None):
    print("\n" + "=" * 78)
    print(f"SCENARIO: {title}")
    print("=" * 78)

    agent = agent or ReasoningAgent(force_mock=True)
    engine = engine or GovernanceEngine()

    reasoning_output = agent.analyze(reasoning_input)

    print("\n--- Reasoning Agent Output (summary) ---")
    print(f"  machine_id:        {reasoning_output['machine_id']}")
    print(f"  severity:          {reasoning_output['failure_analysis']['severity_assessment']}")
    print(f"  primary_cause:     {reasoning_output['root_cause_hypothesis']['primary_cause']}")
    print(f"  risk_level:        {reasoning_output['risk_assessment']['overall_risk_level']} "
          f"(score={reasoning_output['risk_assessment']['risk_score']})")
    print(f"  top_action:        {reasoning_output['recommended_actions'][0]['action_type']} "
          f"(urgency={reasoning_output['recommended_actions'][0]['urgency']})")
    print(f"  overall_confidence:{reasoning_output['confidence_score']['overall']}")
    print(f"  flags:             {reasoning_output['flags']}")

    machine_context = machine_context_override or reasoning_input.get("machine_context", {})
    governance_result = engine.evaluate(
        reasoning_output,
        machine_context=machine_context,
        policy_overrides=policy_overrides or {},
        recurrence_count_last_24h=recurrence_count_last_24h,
    )

    print("\n--- Governance Engine Output (summary) ---")
    print(f"  VERDICT:           {governance_result['decision']['verdict']}")
    print(f"  rationale:         {governance_result['decision']['decision_rationale']}")
    print(f"  override_applied:  {governance_result['decision']['override_applied']} "
          f"({governance_result['decision']['override_reason']})")
    print(f"  checks_passed:     {governance_result['audit_log']['governance_checks_passed']}")
    print(f"  checks_failed:     {governance_result['audit_log']['governance_checks_failed']}")
    if governance_result['human_approval_request']['required']:
        print(f"  escalation_level:  {governance_result['human_approval_request']['escalation_level']}")
        print(f"  urgency:           {governance_result['human_approval_request']['urgency']}")

    print("\n--- Policy Check Detail ---")
    for c in governance_result["policy_checks"]:
        print(f"  [{c['result']:4}] {c['check_name']}: {c['detail']}")

    return reasoning_output, governance_result


def main():
    agent = ReasoningAgent(force_mock=True)  # force mock so this runs with zero API setup
    engine = GovernanceEngine()

    # ── Scenario 1: CRITICAL failure, mandatory SOP, operator present ──
    # Expect: matrix says IMMEDIATE_SHUTDOWN @ CRITICAL -> AUTO, and all gates pass -> AUTO_APPROVE
    inp1 = make_input(failure_probability=0.92, model_confidence=0.88, retrieval_confidence=0.81)
    run_scenario(
        "CRITICAL tool wear failure, good evidence, operator present -> expect AUTO_APPROVE",
        inp1,
        machine_context_override={"operator_present": True, "batch_criticality": "HIGH"},
        agent=agent, engine=engine,
    )

    # ── Scenario 2: Same as above but NO operator present + IMMEDIATE urgency ──
    # Expect: Operator Presence Gate fails -> escalate to SUPERVISOR
    inp2 = make_input(failure_probability=0.92, model_confidence=0.88, retrieval_confidence=0.81)
    run_scenario(
        "Same as Scenario 1 but NO operator present -> expect HUMAN_APPROVAL_REQUIRED (supervisor)",
        inp2,
        machine_context_override={"operator_present": False, "batch_criticality": "HIGH"},
        agent=agent, engine=engine,
    )

    # ── Scenario 3: MEDIUM risk, low confidence -> Confidence Gate should force human review ──
    inp3 = make_input(
        failure_probability=0.55, model_confidence=0.4, retrieval_confidence=0.3,
        similar_incidents=[], relevant_sops=[],
        air_temp=300, process_temp=308, rpm=1800, torque=20, tool_wear=50,
    )
    run_scenario(
        "MEDIUM risk but LOW confidence + no RAG matches -> expect HUMAN_APPROVAL_REQUIRED",
        inp3,
        machine_context_override={"operator_present": True, "batch_criticality": "LOW"},
        agent=agent, engine=engine,
    )

    # ── Scenario 4: Healthy machine -> low-moderate risk depending on mock scoring ──
    inp4 = make_input(
        failure_probability=0.10, model_confidence=0.95, retrieval_confidence=0.9,
        air_temp=300, process_temp=309, rpm=1800, torque=20, tool_wear=30,
    )
    run_scenario(
        "Healthy machine, low failure probability -> verify matrix routes correctly for whatever risk tier the mock scorer assigns",
        inp4,
        machine_context_override={"operator_present": True, "batch_criticality": "LOW"},
        agent=agent, engine=engine,
    )

    # ── Scenario 5: CRITICAL risk + CRITICAL batch + shutdown action -> Batch Criticality Gate ──
    inp5 = make_input(failure_probability=0.95, model_confidence=0.9, retrieval_confidence=0.85)
    run_scenario(
        "CRITICAL risk during a CRITICAL batch -> expect HUMAN_APPROVAL_REQUIRED (plant manager)",
        inp5,
        machine_context_override={"operator_present": True, "batch_criticality": "CRITICAL"},
        agent=agent, engine=engine,
    )

    # ── Scenario 6: Recurrence Gate -> overrides everything to EMERGENCY_ALERT ──
    inp6 = make_input(failure_probability=0.92, model_confidence=0.88, retrieval_confidence=0.81)
    run_scenario(
        "Same machine triggered HIGH/CRITICAL 3x in 24h -> expect EMERGENCY_ALERT",
        inp6,
        machine_context_override={"operator_present": True, "batch_criticality": "HIGH"},
        recurrence_count_last_24h=3,
        agent=agent, engine=engine,
    )

    # ── Scenario 7: force_human_review_all override active ──
    inp7 = make_input(
        failure_probability=0.10, model_confidence=0.95, retrieval_confidence=0.9,
        air_temp=300, process_temp=309, rpm=1800, torque=20, tool_wear=30,
    )
    run_scenario(
        "LOW risk but force_human_review_all override active -> expect HUMAN_APPROVAL_REQUIRED",
        inp7,
        machine_context_override={"operator_present": True, "batch_criticality": "LOW"},
        policy_overrides={"force_human_review_all": True},
        agent=agent, engine=engine,
    )

    # ── Scenario 8: Timeout Gate resolution ──
    print("\n" + "=" * 78)
    print("SCENARIO: Timeout Gate — human never responds, safe fallback executes")
    print("=" * 78)
    inp8 = make_input(
        failure_probability=0.55, model_confidence=0.4, retrieval_confidence=0.3,
        similar_incidents=[], relevant_sops=[],
    )
    reasoning_output_8 = agent.analyze(inp8)
    governance_result_8 = engine.evaluate(
        reasoning_output_8,
        machine_context={"operator_present": True, "batch_criticality": "LOW"},
    )
    print(f"  Initial verdict:   {governance_result_8['decision']['verdict']}")
    urgency = governance_result_8["human_approval_request"]["urgency"]
    print(f"  Urgency window:    {urgency}")

    timed_out_result = engine.resolve_timeout(governance_result_8, minutes_elapsed=999)
    print(f"  After timeout:     {timed_out_result['decision']['verdict']} "
          f"-> action_approved={timed_out_result['decision']['action_approved']}")
    print(f"  rationale:         {timed_out_result['decision']['decision_rationale']}")

    # ── Sanity assertions (lightweight, not exhaustive) ──
    print("\n" + "=" * 78)
    print("Running lightweight sanity assertions...")
    print("=" * 78)

    r1, g1 = run_scenario(
        "[assert] re-run Scenario 1", inp1,
        machine_context_override={"operator_present": True, "batch_criticality": "HIGH"},
        agent=agent, engine=engine,
    )
    assert g1["decision"]["verdict"] == "AUTO_APPROVE", "Scenario 1 should AUTO_APPROVE"

    r2, g2 = run_scenario(
        "[assert] re-run Scenario 2", inp2,
        machine_context_override={"operator_present": False, "batch_criticality": "HIGH"},
        agent=agent, engine=engine,
    )
    assert g2["decision"]["verdict"] == "HUMAN_APPROVAL_REQUIRED", "Scenario 2 should require human approval"

    r6, g6 = run_scenario(
        "[assert] re-run Scenario 6", inp6,
        machine_context_override={"operator_present": True, "batch_criticality": "HIGH"},
        recurrence_count_last_24h=3,
        agent=agent, engine=engine,
    )
    assert g6["decision"]["verdict"] == "EMERGENCY_ALERT", "Scenario 6 should escalate to EMERGENCY_ALERT"

    print("\nAll sanity assertions passed.")
    print("\nPipeline test complete. Reasoning Agent (mock mode) -> Governance Engine wired correctly.")


if __name__ == "__main__":
    main()
