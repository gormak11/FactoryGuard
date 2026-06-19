from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import json
import os
from functools import lru_cache

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from state import ActionProposal, DiagnosisResult, WorkflowState
from tools import ACTION_TOOLS, DIAGNOSIS_TOOLS

MODEL = os.environ.get("AGENT_MODEL", "llama-3.3-70b-versatile")

@lru_cache(maxsize=1)
def _llm():
    # Imported lazily so the module loads even without the dependency/API key.
    from langchain_groq import ChatGroq
    return ChatGroq(model=MODEL, temperature=0, max_tokens=2000)


def _transcript(messages) -> str:
    """Flatten an agent's message list into plain text for the extractor."""
    out = []
    for m in messages:
        role = getattr(m, "type", "msg")
        content = m.content if isinstance(m.content, str) else json.dumps(m.content)
        if content.strip():
            out.append(f"{role}: {content}")
    return "\n".join(out)



# Agent 1 — Diagnosis + Evidence Agent
DIAGNOSIS_PROMPT = """You are the Diagnosis + Evidence Agent for industrial machinery.

You are given an anomaly alert with flagged sensor readings. Your job:
1. Call `get_machine_baseline` to understand normal ranges if useful.
2. Call `retrieve_evidence` with the symptom pattern to pull incidents, SOPs,
   and maintenance logs. Search more than once if the first query is too narrow.
3. Determine the single most likely ROOT CAUSE, grounded in the retrieved
   evidence. Cite the document IDs you relied on.

Be rigorous: do not guess a root cause you cannot support with retrieved
evidence. State your confidence honestly. If the evidence is thin, say so and
keep confidence low."""


def _build_diagnosis_agent():
    return create_react_agent(_llm(), DIAGNOSIS_TOOLS, prompt=DIAGNOSIS_PROMPT)


def diagnosis_node(state: WorkflowState) -> dict:
    """Run the Diagnosis agent and write a typed DiagnosisResult into state."""
    alert = state["alert"]
    attempt = state.get("diagnosis_attempts", 0) + 1

    task = (
        f"Anomaly alert on machine {alert['machine_id']} "
        f"(anomaly score {alert['anomaly_score']}).\n"
        f"Flagged sensors:\n"
        + "\n".join(
            f"  - {s['sensor']}: value={s['value']}, "
            f"normal={s['normal_range']}, z={s['z_score']}"
            for s in alert["flagged_sensors"]
        )
    )
    # On a re-run, pass the orchestrator's feedback back to the agent.
    if attempt > 1 and state.get("diagnosis"):
        task += (
            f"\n\nYour previous diagnosis had low confidence "
            f"({state['diagnosis'].get('confidence')}). Retrieve additional "
            f"evidence and refine the root cause."
        )

    agent = _build_diagnosis_agent()
    result = agent.invoke({"messages": [HumanMessage(content=task)]})

    # Distill the agent's reasoning into the typed handoff contract.
    extractor = _llm().with_structured_output(DiagnosisResult)
    diagnosis: DiagnosisResult = extractor.invoke([
        SystemMessage(content="Extract a structured diagnosis from this analysis transcript."),
        HumanMessage(content=_transcript(result["messages"])),
    ])

    return {
        "diagnosis": diagnosis.model_dump(),
        "diagnosis_attempts": attempt,
        "messages": [AIMessage(
            content=f"[Diagnosis Agent] root_cause='{diagnosis.root_cause}' "
                    f"failure_mode='{diagnosis.failure_mode}' "
                    f"confidence={diagnosis.confidence} "
                    f"evidence={diagnosis.supporting_evidence}"
        )],
    }


# Agent 2 — Action Proposal Agent
ACTION_PROMPT = """You are the Action Proposal Agent for industrial machinery.

You receive a confirmed DIAGNOSIS from the Diagnosis Agent. Your job:
1. Call `get_action_catalog` to see which actions are permitted. You may ONLY
   propose actions from this catalog.
2. Call `lookup_recommended_actions` with the diagnosis failure_mode to get the
   SOP-recommended response.
3. Propose a concise, prioritized set of remedial actions. For each action give
   a one-line rationale tied to the diagnosis, and an urgency
   (low / medium / high / critical).

Be conservative and safety-minded: never propose increasing speed or load on a
machine with an active mechanical or thermal fault."""


def _build_action_agent():
    return create_react_agent(_llm(), ACTION_TOOLS, prompt=ACTION_PROMPT)


def action_node(state: WorkflowState) -> dict:
    """Run the Action agent off the diagnosis and write a typed ActionProposal."""
    dx = state["diagnosis"]
    task = (
        "Confirmed diagnosis to act on:\n"
        f"  root_cause: {dx['root_cause']}\n"
        f"  failure_mode: {dx['failure_mode']}\n"
        f"  confidence: {dx['confidence']}\n"
        f"  evidence: {dx['supporting_evidence']}\n\n"
        "Propose the remedial actions."
    )

    agent = _build_action_agent()
    result = agent.invoke({"messages": [HumanMessage(content=task)]})

    extractor = _llm().with_structured_output(ActionProposal)
    proposal: ActionProposal = extractor.invoke([
        SystemMessage(content="Extract a structured action proposal from this transcript."),
        HumanMessage(content=_transcript(result["messages"])),
    ])

    action_list = ", ".join(f"{a.action}({a.urgency})" for a in proposal.actions)
    return {
        "action_proposal": proposal.model_dump(),
        "messages": [AIMessage(content=f"[Action Agent] proposed: {action_list}")],
    }
