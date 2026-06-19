FactoryGuard AI — Governance Layer Policy Rules

Version 1.0

Owner: AI Reasoning & Governance Lead

Purpose
This document defines the complete, authoritative rule set used by the Governance Layer to decide whether an AI-recommended action is auto-approved, escalated to a human, blocked, or treated as an emergency. These rules are deterministic — given the same input, the Governance Layer must always produce the same verdict. This is what separates FactoryGuard from "an LLM that sounds confident."

1. Risk Levels
LevelRisk Score RangeFailure Probability RangeDescriptionLOW0–25< 0.40Normal operating conditionMEDIUM26–500.40–0.65Elevated condition, watch closelyHIGH51–750.65–0.85Action needed, human review requiredCRITICAL76–100> 0.85Immediate intervention, possible emergency
risk_score is computed by the Reasoning Agent as:
risk_score = (safety_risk_score × 0.40)
           + (equipment_risk_score × 0.25)
           + (production_risk_score × 0.20)
           + (compliance_risk_score × 0.15)
Each component is scored 0–100 by the Reasoning Agent before reaching governance.

2. Action Types
Action TypeDescriptionReversible?MONITORNo physical action, continue observingYesREDUCE_LOADLower speed/load on machineYesSCHEDULE_MAINTENANCEPlan downtime window for repairYesIMMEDIATE_SHUTDOWNControlled shutdown nowPartially (production impact)EMERGENCY_STOPHard stop, safety-criticalNo (always logged as critical event)

3. Decision Matrix (Authoritative)
This matrix is the single source of truth for the verdict logic. Governance Layer code must implement this exactly.
                    RISK LEVEL
                 LOW       MEDIUM      HIGH        CRITICAL
               ┌─────────┬───────────┬───────────┬───────────┐
MONITOR        │  AUTO   │   AUTO    │  HUMAN    │  BLOCK    │
REDUCE_LOAD    │  AUTO   │  HUMAN    │  HUMAN    │  BLOCK    │
SCHEDULE_MAINT │  AUTO   │  HUMAN    │  HUMAN    │  HUMAN    │
IMMEDIATE_SHUT │  HUMAN  │  HUMAN    │  HUMAN    │   AUTO    │
EMERGENCY_STOP │  BLOCK  │  HUMAN    │   AUTO    │   AUTO    │
               └─────────┴───────────┴───────────┴───────────┘
Reading this matrix:

AUTO = execute automatically, write audit log, no human needed
HUMAN = send to operator dashboard, wait for approval/reject/modify
BLOCK = reject outright, do not execute, log reason, request re-evaluation

Why some cells look "backwards":

EMERGENCY_STOP at LOW risk = BLOCK, because stopping a healthy machine is itself a risky, costly action with no justification — this prevents the AI from being trigger-happy.
IMMEDIATE_SHUTDOWN at LOW risk = HUMAN, not AUTO, because shutting down a low-risk machine immediately is unusual enough to warrant a sanity check.
MONITOR at CRITICAL = BLOCK, because "just monitor" is never an acceptable response to a CRITICAL situation — the system refuses to let the AI under-react.


4. The 8 Policy Checks
Every recommended action passes through all 8 checks, in order. The first FAIL determines the outcome unless explicitly stated otherwise.
Check 1 — Safety Gate
IF reasoning_output.risk_assessment.risk_dimensions.safety_risk.level == "CRITICAL"
AND recommended_action.action_type != "EMERGENCY_STOP"
AND recommended_action.action_type != "IMMEDIATE_SHUTDOWN"
THEN: FAIL → verdict = BLOCKED
ELSE: PASS
Rationale: if safety risk is CRITICAL, the only acceptable actions are shutdown or emergency stop. Anything else (like MONITOR or REDUCE_LOAD) is automatically blocked regardless of what the matrix says.
Check 2 — Confidence Gate
IF reasoning_output.confidence_score.overall < 0.65
THEN: FAIL → verdict = HUMAN_APPROVAL_REQUIRED (overrides AUTO from matrix)
ELSE: PASS
Rationale: low confidence means the AI itself is unsure. Never let an unsure AI auto-execute, even if the matrix says AUTO.
Check 3 — Novel Pattern Gate
IF reasoning_output.flags.novel_failure_pattern == true
THEN: FAIL → verdict = HUMAN_APPROVAL_REQUIRED (overrides AUTO from matrix)
ELSE: PASS
Rationale: if this failure pattern has never been seen before (no RAG match), a human must review it once before the system can act on it autonomously.
Check 4 — SOP Compliance Gate
IF any sop in reasoning_output.evidence_used.sops_cited
   has priority == "MANDATORY"
AND recommended_action violates that SOP's stated procedure
THEN: FAIL → verdict = BLOCKED
ELSE: PASS
Rationale: mandatory SOPs are non-negotiable. The AI cannot reason its way around a compliance requirement.
Check 5 — Operator Presence Gate
IF recommended_action.urgency == "IMMEDIATE"
AND machine_context.operator_present == false
THEN: FAIL → verdict = HUMAN_APPROVAL_REQUIRED, escalation_level = SUPERVISOR
ELSE: PASS
Rationale: an immediate action with nobody on the floor to observe the outcome must be escalated, not silently executed.
Check 6 — Batch Criticality Gate
IF machine_context.batch_criticality == "CRITICAL"
AND recommended_action.action_type IN ["IMMEDIATE_SHUTDOWN", "EMERGENCY_STOP"]
THEN: FAIL → verdict = HUMAN_APPROVAL_REQUIRED, escalation_level = PLANT_MANAGER
ELSE: PASS
Rationale: stopping a machine mid-batch on a high-value order is a business decision, not just a technical one — it needs plant manager visibility even if it would otherwise be AUTO.
Check 7 — Recurrence Gate
IF count(machine_id triggered HIGH or CRITICAL in last 24 hours) >= 3
THEN: FAIL → verdict = EMERGENCY_ALERT (overrides everything else)
ELSE: PASS
Rationale: repeated high-risk triggers on the same machine in a short window indicates either a worsening fault or a sensor/model problem — both deserve escalation beyond normal flow.
Check 8 — Timeout Gate
IF verdict == HUMAN_APPROVAL_REQUIRED
AND time_elapsed > urgency_window (IMMEDIATE=15min, WITHIN_1HR=60min, WITHIN_4HR=240min)
AND no human response received
THEN: execute SAFE_FALLBACK_ACTION (defined per machine type, defaults to REDUCE_LOAD or MONITOR — never EMERGENCY_STOP by timeout)
ELSE: continue waiting
Rationale: the system must never silently fail open into "do nothing" or fail dangerously into an unreviewed emergency stop. Timeout always resolves to the safest non-destructive fallback.

5. Verdict Priority Order
When multiple checks fail simultaneously, the verdict with the highest priority wins:
1. EMERGENCY_ALERT       (highest priority — always wins)
2. BLOCKED
3. HUMAN_APPROVAL_REQUIRED
4. AUTO_APPROVE           (lowest priority — only if everything else passes)

6. Override Conditions
These global overrides apply regardless of the matrix or the 8 checks:
OverrideEffectpolicy_overrides.force_human_review_all == trueEvery verdict becomes HUMAN_APPROVAL_REQUIRED, no exceptionspolicy_overrides.emergency_mode_active == trueAll AUTO verdicts downgrade to HUMAN_APPROVAL_REQUIREDpolicy_overrides.maintenance_window_active == trueSCHEDULE_MAINTENANCE actions auto-approve regardless of risk level

7. Audit Logging Requirement
Every verdict — regardless of outcome — must write an audit log entry containing: request ID, machine ID, full reasoning trace, all 8 check results (pass/fail/warn with reasons), final verdict, who approved (AI or named human), timestamp of decision, timestamp of execution (if any), and outcome (if known yet). No action is exempt from logging, including BLOCKED ones.

8. Change Control
Any modification to the Decision Matrix (Section 3) or the 8 Policy Checks (Section 4) must be version-bumped and documented with rationale. This file is the single source of truth — code must reference these rules, not redefine them inline.